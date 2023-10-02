import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        # Your code here
        n_objects = x.shape[0]
        n_features = x.shape[1] if len(x.shape) > 1 else 1

        medians = np.median(x, axis = 0)
        broadcasted_medians = np.tile(medians, (n_objects, 1)) if n_features > 1 else np.full(n_objects, medians)
        print(f"x shape {x.shape}")
        print(f"broad {broadcasted_medians.shape}")
        deviations = np.add.reduce(np.abs(x - broadcasted_medians), axis=0) / n_objects
        print(f"deviations shape {deviations.shape}")
        return medians, deviations

        ####

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        self.loc, self.scale = self.mean_abs_deviation_from_median(features) # YOUR CODE HERE
        ####


    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        log_pdf = -np.log(2 * self.scale) - np.abs(values - self.loc) / self.scale
        return log_pdf
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))
