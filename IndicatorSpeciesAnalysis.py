import numpy as np
import pandas as pd
from joblib import Parallel, delayed

class IndicatorSpecies:

    def __init__(self, OTU_table, labels):
        # import feature names and feature data
        self.OTU_table = OTU_table
        self.feature_names = OTU_table.columns.values 
        self.labels = np.ravel(labels.values)

    def calc_IND_stat(self, labels):
        # get feature names and feature values from OTU_table
        feature_names = self.feature_names
        features = np.array(self.OTU_table.values, np.int)
        NS, NF = features.shape
        # ignore zero features
        inds = np.sum(features, 0) > 0
        # determine unique number of sites
        sites = np.unique(labels)
        indvals = np.zeros([NF, len(sites)])

        for i,label in enumerate(sites):
            # calculate A: measure of specificity, n_p / n
            # n_p_A: mean number of species in target site
            n_p_A = np.mean(features[labels==label, :], 0)
            # n: sum of mean number of species in each site
            n = 0
            for label_2 in sites:
                n += np.mean(features[labels==label_2, :], 0)
            A = n_p_A[inds] / n[inds]
            # Calculate B: measure of fidelity, n_p / N_p
            # n_p_B is the number of occurrences of species at target site group
            n_p_B = np.sum(features[labels==label, :] > 0, 0)
            N_p = np.sum(labels==label)
            B = n_p_B[inds] / N_p
            indvals[inds,i] = (A*B)**.5

        indicator_sites = np.array([sites[i] for i in np.argmax(indvals, 1)])
        indvals = np.max(indvals,1)

        return indvals, indicator_sites

    def calc_p(self, indval, nperm):
        def pv():
            randinds = np.random.permutation(len(self.labels))
            rand_labels = self.labels[randinds]
            temp_indval, site = self.calc_IND_stat(rand_labels)
            return temp_indval >= indval
        pvs = Parallel(n_jobs=4)(delayed(pv)() for _ in range(nperm))
        p_values = sum(pvs) / (1+nperm)
        return np.array(p_values)

    def run(self, max_p_value=None, save_data=False, sort_by=False, nperm=199):
        # initialize data frame to save file
        IS_results_dataframe = pd.DataFrame()
        # get indicator species stat for each feature
        indvals, indicator_sites = self.calc_IND_stat(self.labels)
        # calculate p values
        p_values = self.calc_p(indvals, nperm)
        # save data to dataframe
        IS_results_dataframe['OTUs'] = self.feature_names
        IS_results_dataframe['Stat'] = indvals
        IS_results_dataframe['Site Label'] = indicator_sites
        IS_results_dataframe['P value'] = p_values
        # remove insignificant indicator species
        if max_p_value:
            IS_results_dataframe = IS_results_dataframe.iloc[p_values<=max_p_value, :]

        # sort data frame by 'Stat'
        if sort_by:
            IS_results_dataframe = IS_results_dataframe.sort_values(by=sort_by, ascending=False)
        # save data to specified filename
        if save_data:
            IS_results_dataframe.to_csv(save_data, index=False)
        return IS_results_dataframe
