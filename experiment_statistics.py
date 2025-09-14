import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from experiment import Experiment

class ExperimentStatistics:
    def __init__(self, experiment: "Experiment"):
        """
        Parameters
        ----------
        experiment : Experiment
            An instance of the Experiment class containing regionprops and related data.
        """
        self.experiment = experiment  # Reference to the Experiment instance

    def identify_outliers(self, columns=None, q1_threshold=0.01, q3_threshold=0.99):
        regionprops = self.experiment.regionprops
        numeric_cols = regionprops.select_dtypes(include=[np.number]).columns
        if columns is not None:
            numeric_cols = numeric_cols[numeric_cols.isin(columns)]
        numeric_cols = numeric_cols[numeric_cols != 'labels']
        numeric_cols = numeric_cols[numeric_cols != 'label']
        numeric_cols = numeric_cols[numeric_cols != 'group']
        numeric_cols = numeric_cols[numeric_cols != 'sample']
        for col in numeric_cols:
            if col in regionprops.columns:
                q1 = regionprops.groupby('group')[col].quantile(q1_threshold)
                q3 = regionprops.groupby('group')[col].quantile(q3_threshold)
                iqr = q3 - q1
                outlier_mask = ~regionprops[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                column_outlier = col + '_is_outlier'
                if column_outlier not in regionprops.columns:
                    regionprops[column_outlier] = False
                regionprops.loc[outlier_mask, column_outlier] = True

    def remove_outliers(self, columns=None, q1_threshold=0.01, q3_threshold=0.99):
        regionprops = self.experiment.regionprops
        if regionprops is not None:
            numeric_cols = regionprops.select_dtypes(include=[np.number]).columns
            if columns is not None:
                numeric_cols = numeric_cols[numeric_cols.isin(columns)]
            numeric_cols = numeric_cols[numeric_cols != 'labels']
            numeric_cols = numeric_cols[numeric_cols != 'label']
            numeric_cols = numeric_cols[numeric_cols != 'group']
            numeric_cols = numeric_cols[numeric_cols != 'sample']
            for col in numeric_cols:
                if col in regionprops.columns:
                    q1 = regionprops.groupby('group')[col].transform(lambda x: x.quantile(q1_threshold))
                    q3 = regionprops.groupby('group')[col].transform(lambda x: x.quantile(q3_threshold))
                    iqr = q3 - q1
                    outlier_mask = ~regionprops[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                    self.experiment.regionprops = regionprops[~outlier_mask]

    def compare_groups(self, metric='area'):
        import scipy.stats as stats
        regionprops = self.experiment.regionprops
        self.normality_results = regionprops.groupby('group')[metric].apply(lambda x: self.normality_test(x))
        group_values = [group[metric].dropna().values for name, group in regionprops.groupby('group')]
        if len(group_values) < 2:
            self.anova_results = None
            self.kruskal_results = None
            self.tukey_results = pd.DataFrame()
            return
        self.homoscedasticity_result = stats.levene(*group_values)
        if all(result.pvalue > 0.05 for result in self.normality_results) and self.homoscedasticity_result.pvalue > 0.05:
            self.anova_results = stats.f_oneway(*group_values)
            self.posthoc_tukey(metric=metric)
            self.kruskal_results = None
        else:
            self.kruskal_results = stats.kruskal(*group_values)
            self.tukey_results = pd.DataFrame()
            self.anova_results = None

    def posthoc_tukey(self, metric='area'):
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        regionprops = self.experiment.regionprops
        if regionprops['group'].nunique() > 1:
            tukey = pairwise_tukeyhsd(endog=regionprops[metric], groups=regionprops['group'])
            self.tukey_results = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        else:
            self.tukey_results = pd.DataFrame()

    def normality_test(self, arr):
        import scipy.stats as stats
        arr = np.asarray(arr)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 5000:
            return stats.normaltest(arr)
        else:
            return stats.shapiro(arr)

