from trainer.metrics import StaticMetric, VegaChartMetric, DataFrameMetric
import numpy as np
from einops import rearrange
import altair as alt
import pandas as pd
import json
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from tqdm import trange, tqdm

class GridWorldClusterMetric(VegaChartMetric):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        # Set up TSNE
        self.n_clusters = [2, 3, 4, 5, 6]
        self.n_init = self.config.get('n_init', 10)
        self.random_state = self.config.get('random_state', 0)

    def log(self, eval_output, storage, filename):
        """
        Compute t-SNE clusters of latent representations
        """
        # Get evaluated features
        features = eval_output['gridworld']['features']
        positions = eval_output['gridworld']['states']
        grid_size = min(positions[-1,:] + 1)
        
        x, y = np.meshgrid(range(grid_size), range(grid_size))
        data = pd.DataFrame({'x': x.ravel(), 'y': y.ravel()})

        n_data = features.shape[0]
        charts = []
        for idp, n_clusters in tqdm(enumerate(self.n_clusters)):
            print(f"Running K-means with n_clusters {n_clusters}")
            kmeans = KMeans(n_clusters=n_clusters,
                            n_init=self.n_init,
                            random_state=self.random_state)
            kmeans.fit(features)
            clusters = kmeans.predict(features)

            counter = 0
            heatmaps = np.zeros((grid_size * grid_size))
            for idx in range(grid_size):
                for idy in range(grid_size):
                    heatmaps[counter] = clusters[counter]
                    counter += 1
        
            data[f'clusters_{n_clusters}'] = heatmaps.ravel()

            chart = alt.Chart(data).mark_rect().encode(
                x='x:O',
                y='y:O',
                color=f'clusters_{n_clusters}:Q',
            ).properties(
                title=f"{n_clusters} clusters"
            )

            # chart.configure(title=f"{idp} clusters")
            charts.append(chart)
        charts = alt.hconcat(*charts)
        charts.save(f"{filename}.html")

        charts = charts.to_json()
        storage.save(f"{filename}.json", charts, filetype='json')
        return {'filepath': storage.storage_path(f"{filename}.json")}



        
        #     # Mesure cluster quality
        #     cluster_qualities.append(calinski_harabasz_score(features, clusters))

        #     # Measure clustering quality based on values
        #     cluster_value_centroids = []
        #     cluster_prop = []
        #     wcss = 0.
        #     for idc in range(n_clusters):
        #         cluster_values = values[clusters == idc]
        #         cluster_centroid = cluster_values.mean()
        #         # compute within-cluster separation
        #         wcss += ((cluster_values - cluster_centroid)**2).sum()
        #         # store metrics for between-cluster separation
        #         cluster_prop.append(len(cluster_values)/n_data)
        #         cluster_value_centroids.append(cluster_centroid)
        #     cluster_value_centroids = np.array(cluster_value_centroids)
            
        #     # compute between-cluster separation
        #     bcss = (cluster_value_centroids - cluster_value_centroids.mean())**2
        #     bcss = np.average(bcss, weights=cluster_prop)*n_data

        #     # value-based clustering quality
        #     cluster_qualities_value.append((bcss/(n_clusters - 1))/(wcss/(n_data - n_clusters)))
            

        

        # # Plot the TSNE embeddings
        # data = pd.DataFrame({
        #     'n_clusters': self.n_clusters,
        #     'ch_index': cluster_qualities,
        #     'ch_index_value': cluster_qualities_value
        # })
            

        # ch_index_chart = alt.Chart(data, title="CH Index").mark_bar().encode(
        #     x=alt.X('n_clusters:O'),
        #     y=alt.Y('ch_index:Q'),
        #     color=alt.Color('n_clusters:O')
        # )
        # ch_index_value_chart = alt.Chart(data, title="CH Index (Value)").mark_bar().encode(
        #     x=alt.X('n_clusters:O'),
        #     y=alt.Y('ch_index_value:Q'),
        #     color=alt.Color('n_clusters:O')
        # )
        # chart = alt.hconcat(ch_index_chart, ch_index_value_chart)
        # chart = chart.to_json()
        # storage.save(f"{filename}.json", chart, filetype='json')
        # return {'filepath': storage.storage_path(f"{filename}.json")}


class GridWorldFeaturesHeatmapMetric(VegaChartMetric):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_features_plot = 5 # Only plot first 5 features

    def log(self, eval_output, storage, filename):
        features = eval_output['gridworld']['features']
        positions = eval_output['gridworld']['states']
        grid_size = min(positions[-1,:] + 1)
        
        heatmaps = np.zeros((self.num_features_plot, grid_size * grid_size))
        for idf in range(self.num_features_plot):
            counter = 0
            for idx in range(grid_size):
                for idy in range(grid_size):
                    heatmaps[idf, counter] = features[counter, idf]
                    counter += 1
        heatmaps = rearrange(heatmaps, 'f (x y) -> f x y', x=grid_size)
        x, y = np.meshgrid(range(grid_size), range(grid_size))
        data = pd.DataFrame({'x': x.ravel(), 'y': y.ravel()})
        
        charts = []
        for idf in range(self.num_features_plot):
            data[f'feature_{idf}'] = heatmaps[idf,:].ravel()
            chart = alt.Chart(data).mark_rect().encode(
                x='x:O',
                y='y:O',
                color=f'feature_{idf}:Q'
            )
            charts.append(chart)
        charts = alt.hconcat(*charts)
        charts = charts.to_json()
        storage.save(f"{filename}.json", charts, filetype='json')
        return {'filepath': storage.storage_path(f"{filename}.json")}

class RewardsDataFrameMetric(DataFrameMetric):
    
    def log(self, eval_output, storage, filename):
        rewards = eval_output['rewards'] # (T, num_envs)
        rewards = rewards.sum(axis=0) # sum over time-steps
        df = pd.DataFrame({
            'n': np.arange(len(rewards)),
            'reward': rewards
        })
        # Save dataframe as json
        storage.save(f"{filename}.json", df.to_json(), filetype='json')
        return {'filepath': storage.storage_path(f"{filename}.json")}


class KMeansClustersMetric(VegaChartMetric):
    """
    Metric for computing K-means clusters from latent representations
    """
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        # Set up TSNE
        self.n_clusters = [2,4,6,8,10]
        self.n_init = self.config.get('n_init', 10)
        self.random_state = self.config.get('random_state', 0)

    def log(self, eval_output, storage, filename):
        """
        Compute t-SNE clusters of latent representations
        """
        # Get evaluated features
        features = eval_output['clusters']['features']
        # Normalize the features before evaluating
        features = features / np.linalg.norm(features, axis=1, keepdims=True, ord=1)
        values = eval_output['clusters']['values']
        
        n_data = features.shape[0]
        cluster_qualities = []
        cluster_qualities_value = []
        for idp, n_clusters in tqdm(enumerate(self.n_clusters)):
            print(f"Running K-means with n_clusters {n_clusters}")
            kmeans = KMeans(n_clusters=n_clusters,
                            n_init=self.n_init,
                            random_state=self.random_state)
            kmeans.fit(features)
            clusters = kmeans.predict(features)
            # Mesure cluster quality
            cluster_qualities.append(calinski_harabasz_score(features, clusters))

            # Measure clustering quality based on values
            cluster_value_centroids = []
            cluster_prop = []
            wcss = 0.
            for idc in range(n_clusters):
                cluster_values = values[clusters == idc]
                cluster_centroid = cluster_values.mean()
                # compute within-cluster separation
                wcss += ((cluster_values - cluster_centroid)**2).sum()
                # store metrics for between-cluster separation
                cluster_prop.append(len(cluster_values)/n_data)
                cluster_value_centroids.append(cluster_centroid)
            cluster_value_centroids = np.array(cluster_value_centroids)
            
            # compute between-cluster separation
            bcss = (cluster_value_centroids - cluster_value_centroids.mean())**2
            bcss = np.average(bcss, weights=cluster_prop)*n_data

            # value-based clustering quality
            cluster_qualities_value.append((bcss/(n_clusters - 1))/(wcss/(n_data - n_clusters)))
            

        

        # Plot the TSNE embeddings
        data = pd.DataFrame({
            'n_clusters': self.n_clusters,
            'ch_index': cluster_qualities,
            'ch_index_value': cluster_qualities_value
        })
            

        ch_index_chart = alt.Chart(data, title="CH Index").mark_bar().encode(
            x=alt.X('n_clusters:O'),
            y=alt.Y('ch_index:Q'),
            color=alt.Color('n_clusters:O')
        )
        ch_index_value_chart = alt.Chart(data, title="CH Index (Value)").mark_bar().encode(
            x=alt.X('n_clusters:O'),
            y=alt.Y('ch_index_value:Q'),
            color=alt.Color('n_clusters:O')
        )
        chart = alt.hconcat(ch_index_chart, ch_index_value_chart)
        chart = chart.to_json()
        storage.save(f"{filename}.json", chart, filetype='json')
        return {'filepath': storage.storage_path(f"{filename}.json")}
            


class TSNEClustersMetric(VegaChartMetric):
    """
    Metric for computing t-SNE clusters of latent representations
    """
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        # Set up TSNE
        self.n_components = self.config.get('n_components', 2)
        self.n_iter = self.config.get('n_iter', 5000)
        self.random_state = self.config.get('random_state', 0)
        self.init = self.config.get('init', 'random')
        self.perplexities = [5, 7, 9, 11, 20]


    def log(self, eval_output, storage, filename):
        """
        Compute t-SNE clusters of latent representations
        """
        # Get evaluated features
        features = eval_output['clusters']['features']
        # Normalize the features before evaluating
        features = features / np.linalg.norm(features, axis=1, keepdims=True, ord=1)
        values = eval_output['clusters']['values']
        value_min, value_max = values.min(), values.max()

        all_data = []
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        for idp, perplexity in tqdm(enumerate(self.perplexities)):
            print(f"Running TSNE with perplexity {perplexity}")

            tsne = TSNE(
                n_components=self.n_components,
                init=self.init,
                random_state=self.random_state,
                perplexity=perplexity,
                n_iter=self.n_iter
            )
        

            # Get TSNE embeddings
            tsne_features = tsne.fit_transform(features)
            # Plot the TSNE embeddings
            data = pd.DataFrame({
                'x': tsne_features[:, 0],
                'y': tsne_features[:, 1],
                'value': values.reshape(-1)
            })
            all_data.append(data)

        x_min = min([df['x'].min() for df in all_data])
        x_max = max([df['x'].max() for df in all_data])
        y_min = min([df['y'].min() for df in all_data])
        y_max = max([df['y'].max() for df in all_data])

        charts = []
        for idx, df in enumerate(all_data):
            tsne_chart = alt.Chart(df, title=f'Perplexity {self.perplexities[idx]}').mark_circle(size=60, opacity=0.5).encode(
                x=alt.X('x:Q', scale=alt.Scale(domain=[x_min, x_max])),
                y=alt.Y('y:Q', scale=alt.Scale(domain=[y_min, y_max])),
                color=alt.Color('value:Q', 
                                scale=alt.Scale(scheme='redblue', domain=[value_min, value_max]))
            )
            charts.append(tsne_chart)


        chart = alt.hconcat(*charts)
        chart = chart.to_json()
        storage.save(f"{filename}.json", chart, filetype='json')
        return {'filepath': storage.storage_path(f"{filename}.json")}
            