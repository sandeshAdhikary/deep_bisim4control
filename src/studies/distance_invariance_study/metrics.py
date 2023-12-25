from trainer.metrics import StaticScalarMetric
import numpy as np

class PredictionErrorMetric(StaticScalarMetric):

    def log(self, eval_output):
        """
        return MSE prediction error from eval_output
        """
        
        pred_errs = (eval_output['preds'].reshape(-1) - eval_output['y'].reshape(-1))**2
        return {
            'avg':  np.nanmean(pred_errs), # Mean across batches
            'std': np.nanstd(pred_errs) # Std across batches
        }