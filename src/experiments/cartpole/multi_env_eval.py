class MultiDistractorExperiment():
    """
    Experiment to test the generalization of RL models on multiple distrctor environments
    """


    def __init__(self, config):
        self.project = config['project']
        self.run = config['run']
        self.sweep = config.get('sweep')

        self.evaluator = config.get('evaluator')

    def train(self, config):
        pass

    def train_sweep(self, config):
        pass

    def evaluate(self, model):
        """
        Run evaluator on a single run
        output: results.pt
        """
        pass

    def analyze(self, run):
        """
        Analyze the output of evaluation run
        Convert raw data of eval run into vega-lite
        digestible format
        e.g. can be uploaded to wandb as artificat
        to make reports
        """
        pass

    def plot(self, run):
        """
        Plot the analysis of the evaluation run
        output: plot.json
        """
        pass






# Given a run
# Load the model
# Run evaluations on multiple distractor environments
# Create evaluator config with multi env
# Run evaluator
# Save output: test_output_multi_env_eval.pkl 


# Given a series of runs
# Look for test_output_multi_env_eval.pkl
# If a run does not have it, print name so we can run it


# Once you have all the results
# Run a vegalite script to plot the results
# Save the test results as an html file

