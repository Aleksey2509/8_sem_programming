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
        n_features = x.shape[1]

        medians = []
        deviations = []

        for i in range(n_features):
            ith_feature_vec = x[:, i]
            med = np.partition(ith_feature_vec, int(n_objects / 2))[int(n_objects / 2)]
            deviat = np.reduce(np.abs(ith_feature_vec - np.full(n_objects, med))) / n_objects

            medians.append(med)
            deviations.append(deviat)

        return medians, deviat

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
        return 
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))
