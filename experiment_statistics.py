import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests  
from typing import List, Dict, Tuple, Optional, Union, Any
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
from pathlib import Path
import warnings
import sys
import json, pickle, gzip, os, hashlib, time

if TYPE_CHECKING:
    from experiment import Experiment

@dataclass
class StatisticalTestResult:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    assumptions_met: bool = True
    sample_size: int = 0
    power: Optional[float] = None

@dataclass
class OutlierAnalysis:
    """Container for outlier analysis results"""
    method: str
    outlier_indices: List[int]
    outlier_count: int
    outlier_percentage: float
    threshold_lower: Optional[float] = None
    threshold_upper: Optional[float] = None
    z_scores: Optional[np.ndarray] = None

class ExperimentStatistics:
    def __init__(self, regionprops: pd.DataFrame):
        """
        Enhanced statistical analysis for experiment data.
        
        Parameters
        ----------
        regionprops : pd.DataFrame
            DataFrame containing regionprops data with columns for measurements
            and grouping variables.
        """
        self.regionprops = regionprops
        self.results_cache = {}
        self.outlier_results = {}
        self.normality_results = {}
        self.comparison_results = {}
        self.correlation_results = {}  
        self.posthoc_results = {}
        self.effect_sizes = {}

    def get_numeric_columns(self, exclude_cols: List[str] = None) -> List[str]:
        """Get numeric columns excluding specified ones"""
        if exclude_cols is None:
            exclude_cols = ['label', 'labels', 'group', 'sample', 'frame', 'scene', 'track_id', 
                          'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'centroid-0', 'centroid-1']
        
        numeric_cols = self.regionprops.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in exclude_cols]

    def get_categorical_columns(self, exclude_cols: List[str] = None) -> List[str]:
        """
        Get categorical columns excluding specified ones
        
        Parameters
        ----------
        exclude_cols : List[str], optional
            Columns to exclude from the categorical list
            
        Returns
        -------
        List[str]
            List of categorical column names
        """
        if exclude_cols is None:
            exclude_cols = ['label', 'labels', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 
                          'centroid-0', 'centroid-1']
        
        # Get non-numeric columns (object, category, bool)
        categorical_cols = self.regionprops.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
        
        # Also include integer columns that might be categorical (like group, sample)
        potential_categorical = ['group', 'sample', 'scene', 'frame', 'track_id', 
                               'condition', 'treatment', 'replicate']
        
        for col in potential_categorical:
            if (col in self.regionprops.columns and 
                col not in categorical_cols and 
                col not in exclude_cols):
                # Check if it has few unique values relative to total (likely categorical)
                unique_ratio = len(self.regionprops[col].unique()) / len(self.regionprops)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    categorical_cols.append(col)
        
        # Remove excluded columns
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        return categorical_cols

    def get_column_types(self) -> Dict[str, List[str]]:
        """
        Get comprehensive column type information
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary with 'numeric', 'categorical', and 'excluded' column lists
        """
        return {
            'numeric': self.get_numeric_columns(),
            'categorical': self.get_categorical_columns(),
            'excluded': ['label', 'labels', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 
                        'centroid-0', 'centroid-1'],
            'all_columns': list(self.regionprops.columns)
        }

    def describe_data_structure(self) -> pd.DataFrame:
        """
        Get detailed information about the data structure
        
        Returns
        -------
        pd.DataFrame
            DataFrame with column information
        """
        column_info = []
        
        for col in self.regionprops.columns:
            dtype = str(self.regionprops[col].dtype)
            n_unique = len(self.regionprops[col].unique())
            n_missing = self.regionprops[col].isnull().sum()
            
            # Determine column category
            if col in self.get_numeric_columns():
                category = 'numeric'
            elif col in self.get_categorical_columns():
                category = 'categorical'
            else:
                category = 'excluded'
            
            column_info.append({
                'Column': col,
                'Data Type': dtype,
                'Category': category,
                'Unique Values': n_unique,
                'Missing Values': n_missing,
                'Missing %': f"{(n_missing / len(self.regionprops)) * 100:.1f}%"
            })
        
        return pd.DataFrame(column_info)
       
    def print_column_summary(self) -> None:
        """Print a summary of column types and data structure"""
        print("📊 DATA STRUCTURE SUMMARY")
        print("=" * 50)
        
        column_types = self.get_column_types()
        
        print(f"📈 Numeric columns ({len(column_types['numeric'])}):")
        for col in column_types['numeric']:
            print(f"  • {col}")
        
        print(f"\n🏷️  Categorical columns ({len(column_types['categorical'])}):")
        for col in column_types['categorical']:
            unique_count = len(self.regionprops[col].unique())
            print(f"  • {col} ({unique_count} unique values)")
        
        print(f"\n❌ Excluded columns ({len(column_types['excluded'])}):")
        for col in column_types['excluded']:
            if col in self.regionprops.columns:
                print(f"  • {col}")
        
        print(f"\n📋 Total columns: {len(self.regionprops.columns)}")
        print(f"📋 Total observations: {len(self.regionprops)}")
        
        # Show detailed table
        print("\n🔍 DETAILED COLUMN INFORMATION:")
        detail_df = self.describe_data_structure()
        print(detail_df.to_string(index=False))

    # ===== OUTLIER DETECTION METHODS =====
    
    def detect_outliers_iqr(self, columns: List[str] = None, group_col: str = 'group', 
                           multiplier: float = 1.5) -> Dict[str, OutlierAnalysis]:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to analyze. If None, uses all numeric columns.
        group_col : str
            Column to group by for outlier detection
        multiplier : float
            IQR multiplier (typically 1.5 for mild, 3.0 for extreme outliers)
            
        Returns
        -------
        Dict[str, OutlierAnalysis]
            Dictionary mapping column names to outlier analysis results
        """
        if columns is None:
            columns = self.get_numeric_columns()
        
        outlier_results = {}
        
        for col in columns:
            if col not in self.regionprops.columns:
                continue
                
            outliers_by_group = []
            
            for group in self.regionprops[group_col].unique():
                group_data = self.regionprops[self.regionprops[group_col] == group][col].dropna()
                if len(group_data) == 0:
                    continue
                
                q1 = group_data.quantile(0.25)
                q3 = group_data.quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - multiplier * iqr
                upper_bound = q3 + multiplier * iqr
                
                group_outliers = group_data[(group_data < lower_bound) | 
                                          (group_data > upper_bound)].index.tolist()
                outliers_by_group.extend(group_outliers)
            
            outlier_results[col] = OutlierAnalysis(
                method=f"IQR (multiplier={multiplier})",
                outlier_indices=outliers_by_group,
                outlier_count=len(outliers_by_group),
                outlier_percentage=(len(outliers_by_group) / len(self.regionprops)) * 100,
                threshold_lower=lower_bound if 'lower_bound' in locals() else None,
                threshold_upper=upper_bound if 'upper_bound' in locals() else None
            )
        
        self.outlier_results['iqr'] = outlier_results
        return outlier_results
    
    def detect_outliers_zscore(self, columns: List[str] = None, group_col: str = 'group',
                              threshold: float = 3.0) -> Dict[str, OutlierAnalysis]:
        """
        Detect outliers using Z-score method.
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to analyze
        group_col : str
            Column to group by
        threshold : float
            Z-score threshold (typically 2.5 or 3.0)
            
        Returns
        -------
        Dict[str, OutlierAnalysis]
            Dictionary mapping column names to outlier analysis results
        """
        if columns is None:
            columns = self.get_numeric_columns()
        
        outlier_results = {}
        
        for col in columns:
            if col not in self.regionprops.columns:
                continue
                
            outliers_by_group = []
            all_z_scores = []
            
            for group in self.regionprops[group_col].unique():
                group_data = self.regionprops[self.regionprops[group_col] == group][col].dropna()
                if len(group_data) == 0:
                    continue
                
                z_scores = np.abs(stats.zscore(group_data))
                all_z_scores.extend(z_scores)
                
                group_outliers = group_data[z_scores > threshold].index.tolist()
                outliers_by_group.extend(group_outliers)
            
            outlier_results[col] = OutlierAnalysis(
                method=f"Z-score (threshold={threshold})",
                outlier_indices=outliers_by_group,
                outlier_count=len(outliers_by_group),
                outlier_percentage=(len(outliers_by_group) / len(self.regionprops)) * 100,
                z_scores=np.array(all_z_scores)
            )
        
        self.outlier_results['zscore'] = outlier_results
        return outlier_results
    
    def detect_outliers_isolation_forest(self, columns: List[str] = None, 
                                       contamination: float = 0.1) -> OutlierAnalysis:
        """
        Detect multivariate outliers using Isolation Forest.
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to analyze
        contamination : float
            Expected proportion of outliers
            
        Returns
        -------
        OutlierAnalysis
            Outlier analysis results
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError("scikit-learn is required for Isolation Forest outlier detection")
        
        if columns is None:
            columns = self.get_numeric_columns()
        
        regionprops = self.regionprops[columns].dropna()
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(regionprops)
        
        outlier_indices = regionprops.index[outlier_labels == -1].tolist()
        
        result = OutlierAnalysis(
            method=f"Isolation Forest (contamination={contamination})",
            outlier_indices=outlier_indices,
            outlier_count=len(outlier_indices),
            outlier_percentage=(len(outlier_indices) / len(regionprops)) * 100
        )
        
        self.outlier_results['isolation_forest'] = result
        return result

    def remove_outliers(self, method: str = 'iqr', columns: List[str] = None, 
                       inplace: bool = False, **kwargs) -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Parameters
        ----------
        method : str
            Outlier detection method ('iqr', 'zscore', 'isolation_forest')
        columns : List[str], optional
            Columns to consider for outlier detection
        inplace : bool
            Whether to modify the experiment data in place
        **kwargs
            Additional arguments for the outlier detection method
            
        Returns
        -------
        pd.DataFrame
            DataFrame with outliers removed if inplace is False
        """
        if method == 'iqr':
            outlier_results = self.detect_outliers_iqr(columns, **kwargs)
            all_outliers = set()
            for result in outlier_results.values():
                all_outliers.update(result.outlier_indices)
        elif method == 'zscore':
            outlier_results = self.detect_outliers_zscore(columns, **kwargs)
            all_outliers = set()
            for result in outlier_results.values():
                all_outliers.update(result.outlier_indices)
        elif method == 'isolation_forest':
            result = self.detect_outliers_isolation_forest(columns, **kwargs)
            all_outliers = set(result.outlier_indices)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        clean_data = self.regionprops.drop(index=list(all_outliers))
        
        if inplace:
            self.regionprops = clean_data
        
        print(f"🧹 Removed {len(all_outliers)} outliers ({len(all_outliers)/len(self.regionprops)*100:.1f}%)")
        if not inplace:
            return clean_data

    # ===== NORMALITY AND ASSUMPTION TESTING =====
    
    def test_normality(self, columns: List[str] = None, group_col: str = 'group', 
                      method: str = 'auto') -> Dict[str, Dict[str, StatisticalTestResult]]:
        """
        Test normality for each group and column.
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to test
        group_col : str
            Grouping column
        method : str
            Test method ('shapiro', 'ks', 'anderson', 'auto')
            
        Returns
        -------
        Dict[str, Dict[str, StatisticalTestResult]]
            Nested dictionary with normality test results
        """
        if columns is None:
            columns = self.get_numeric_columns()
        
        regionprops = self.regionprops
        results = {}
        
        for col in columns:
            if col not in regionprops.columns:
                continue
                
            results[col] = {}
            
            for group in regionprops[group_col].unique():
                group_data = regionprops[regionprops[group_col] == group][col].dropna()
                
                if len(group_data) < 3:
                    results[col][group] = StatisticalTestResult(
                        test_name="Insufficient data",
                        statistic=np.nan,
                        p_value=np.nan,
                        sample_size=len(group_data),
                        interpretation="Not enough data for normality testing"
                    )
                    continue
                
                # Choose test based on sample size and method
                if method == 'auto':
                    if len(group_data) <= 50:
                        test_method = 'shapiro'
                    elif len(group_data) <= 5000:
                        test_method = 'ks'
                    else:
                        test_method = 'anderson'
                else:
                    test_method = method
                
                try:
                    if test_method == 'shapiro':
                        stat, p_val = stats.shapiro(group_data)
                        test_name = "Shapiro-Wilk"
                    elif test_method == 'ks':
                        stat, p_val = stats.kstest(group_data, 'norm', 
                                                 args=(group_data.mean(), group_data.std()))
                        test_name = "Kolmogorov-Smirnov"
                    elif test_method == 'anderson':
                        result = stats.anderson(group_data, dist='norm')
                        stat = result.statistic
                        # Use 5% significance level
                        critical_val = result.critical_values[2]  # 5% level
                        p_val = 0.05 if stat > critical_val else 0.1  # Approximate
                        test_name = "Anderson-Darling"
                    
                    interpretation = "Normal distribution" if p_val > 0.05 else "Non-normal distribution"
                    
                    results[col][group] = StatisticalTestResult(
                        test_name=test_name,
                        statistic=stat,
                        p_value=p_val,
                        sample_size=len(group_data),
                        interpretation=interpretation,
                        assumptions_met=p_val > 0.05
                    )
                    
                except Exception as e:
                    results[col][group] = StatisticalTestResult(
                        test_name=f"Error: {test_method}",
                        statistic=np.nan,
                        p_value=np.nan,
                        sample_size=len(group_data),
                        interpretation=f"Test failed: {str(e)}"
                    )
        
        self.normality_results = results
        return results
    
    def test_homoscedasticity(self, columns: List[str] = None, 
                             group_col: str = 'group') -> Dict[str, StatisticalTestResult]:
        """
        Test homoscedasticity (equal variances) across groups.
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to test
        group_col : str
            Grouping column
            
        Returns
        -------
        Dict[str, StatisticalTestResult]
            Dictionary mapping column names to test results
        """
        if columns is None:
            columns = self.get_numeric_columns()
        
        regionprops = self.regionprops
        results = {}
        
        for col in columns:
            if col not in regionprops.columns:
                continue
            
            groups = [group[col].dropna().values for name, group in regionprops.groupby(group_col)]
            groups = [g for g in groups if len(g) > 0]  # Remove empty groups
            
            if len(groups) < 2:
                results[col] = StatisticalTestResult(
                    test_name="Levene's test",
                    statistic=np.nan,
                    p_value=np.nan,
                    interpretation="Insufficient groups for testing"
                )
                continue
            
            try:
                stat, p_val = stats.levene(*groups)
                interpretation = "Equal variances" if p_val > 0.05 else "Unequal variances"
                
                results[col] = StatisticalTestResult(
                    test_name="Levene's test",
                    statistic=stat,
                    p_value=p_val,
                    interpretation=interpretation,
                    assumptions_met=p_val > 0.05,
                    sample_size=sum(len(g) for g in groups)
                )
            except Exception as e:
                results[col] = StatisticalTestResult(
                    test_name="Levene's test (Error)",
                    statistic=np.nan,
                    p_value=np.nan,
                    interpretation=f"Test failed: {str(e)}"
                )
        
        return results

    # ===== GROUP COMPARISON METHODS =====
    
    def compare_groups_comprehensive(self, columns: List[str] = None, 
                                   group_col: str = 'group') -> Dict[str, Dict[str, StatisticalTestResult]]:
        """
        Comprehensive group comparison with appropriate statistical tests.
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to analyze
        group_col : str
            Grouping column
            
        Returns
        -------
        Dict[str, Dict[str, StatisticalTestResult]]
            Nested dictionary with comparison results
        """
        if columns is None:
            columns = self.get_numeric_columns()
        
        regionprops = self.regionprops
        results = {}
        
        # First, test assumptions
        normality_results = self.test_normality(columns, group_col)
        homoscedasticity_results = self.test_homoscedasticity(columns, group_col)
        
        for col in columns:
            if col not in regionprops.columns:
                continue
            
            results[col] = {}
            
            # Get group data
            groups = [group[col].dropna().values for name, group in regionprops.groupby(group_col)]
            group_names = [name for name, group in regionprops.groupby(group_col) 
                          if len(group[col].dropna()) > 0]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) < 2:
                results[col]['comparison'] = StatisticalTestResult(
                    test_name="No comparison",
                    statistic=np.nan,
                    p_value=np.nan,
                    interpretation="Less than 2 groups available"
                )
                continue
            
            # Check assumptions
            normal_groups = all(
                normality_results.get(col, {}).get(name, StatisticalTestResult("", 0, 1)).assumptions_met 
                for name in group_names
            )
            equal_variances = homoscedasticity_results.get(col, StatisticalTestResult("", 0, 1)).assumptions_met
            
            # Choose appropriate test
            if len(groups) == 2:
                if normal_groups and equal_variances:
                    # Independent t-test
                    stat, p_val = stats.ttest_ind(groups[0], groups[1])
                    test_name = "Independent t-test"
                    effect_size = self.calculate_cohens_d(groups[0], groups[1])
                elif normal_groups and not equal_variances:
                    # Welch's t-test
                    stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    test_name = "Welch's t-test"
                    effect_size = self.calculate_cohens_d(groups[0], groups[1])
                else:
                    # Mann-Whitney U test
                    stat, p_val = stats.mannwhitneyu(groups[0], groups[1])
                    test_name = "Mann-Whitney U test"
                    effect_size = self.calculate_rank_biserial_correlation(groups[0], groups[1])
            else:
                if normal_groups and equal_variances:
                    # One-way ANOVA
                    stat, p_val = stats.f_oneway(*groups)
                    test_name = "One-way ANOVA"
                    effect_size = self.calculate_eta_squared(*groups)
                else:
                    # Kruskal-Wallis test
                    stat, p_val = stats.kruskal(*groups)
                    test_name = "Kruskal-Wallis test"
                    effect_size = self.calculate_epsilon_squared(*groups)
            
            # Interpret effect size
            effect_interpretation = self.interpret_effect_size(effect_size, test_name)
            
            results[col]['comparison'] = StatisticalTestResult(
                test_name=test_name,
                statistic=stat,
                p_value=p_val,
                effect_size=effect_size,
                interpretation=f"p={p_val:.4f}, {effect_interpretation}",
                assumptions_met=normal_groups and equal_variances,
                sample_size=sum(len(g) for g in groups)
            )
            
            # Post-hoc tests if significant and more than 2 groups
            if p_val < 0.05 and len(groups) > 2:
                results[col]['posthoc'] = self.posthoc_tests(col, group_col, test_name)
        
        self.comparison_results = results
        return results
    
    def posthoc_tests(self, column: str, group_col: str = 'group', 
                     main_test: str = 'ANOVA') -> Dict[str, pd.DataFrame]:
        """
        Perform post-hoc tests after significant main test.
        
        Parameters
        ----------
        column : str
            Column to analyze
        group_col : str
            Grouping column
        main_test : str
            The main test that was significant
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with post-hoc test results
        """
        regionprops = self.regionprops
        results = {}
        
        try:
            if 'ANOVA' in main_test or 't-test' in main_test:
                # Tukey's HSD for parametric tests
                tukey = pairwise_tukeyhsd(
                    endog=regionprops[column].dropna(),
                    groups=regionprops.loc[regionprops[column].dropna(), group_col]
                )
                results['tukey'] = pd.DataFrame(data=tukey.summary().data[1:], 
                                              columns=tukey.summary().data[0])
            else:
                # Dunn's test for non-parametric tests
                try:
                    import scikit_posthocs as sp
                    dunn_results = sp.posthoc_dunn(regionprops, val_col=column, 
                                                 group_col=group_col, p_adjust='bonferroni')
                    results['dunn'] = dunn_results
                except ImportError:
                    # Fallback: Pairwise Mann-Whitney with Bonferroni correction
                    results['pairwise_mannwhitney'] = self.pairwise_mannwhitney(column, group_col)
            
        except Exception as e:
            print(f"Post-hoc test failed for {column}: {e}")
            results['error'] = f"Post-hoc test failed: {e}"
        
        self.posthoc_results[column] = results
        return results
    
    def pairwise_mannwhitney(self, column: str, group_col: str = 'group') -> pd.DataFrame:
        """Pairwise Mann-Whitney tests with Bonferroni correction"""
        regionprops = self.regionprops
        groups = regionprops.groupby(group_col)[column].apply(lambda x: x.dropna().values)
        group_names = list(groups.index)
        
        results = []
        p_values = []
        
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                group1, group2 = group_names[i], group_names[j]
                data1, data2 = groups[group1], groups[group2]
                
                if len(data1) > 0 and len(data2) > 0:
                    stat, p_val = stats.mannwhitneyu(data1, data2)
                    p_values.append(p_val)
                    results.append({
                        'group1': group1,
                        'group2': group2,
                        'statistic': stat,
                        'p_unadjusted': p_val
                    })
        
        # Bonferroni correction
        n_comparisons = len(p_values)
        p_adjusted = [min(p * n_comparisons, 1.0) for p in p_values]
        
        for i, result in enumerate(results):
            result['p_adjusted'] = p_adjusted[i]
            result['significant'] = p_adjusted[i] < 0.05
        
        return pd.DataFrame(results)

    # ===== EFFECT SIZE CALCULATIONS =====
    
    def calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def calculate_eta_squared(self, *groups) -> float:
        """Calculate eta-squared effect size for ANOVA"""
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)
        ss_total = np.sum((all_data - grand_mean) ** 2)
        
        return ss_between / ss_total if ss_total > 0 else 0
    
    def calculate_rank_biserial_correlation(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate rank-biserial correlation for Mann-Whitney U"""
        n1, n2 = len(group1), len(group2)
        u_statistic, _ = stats.mannwhitneyu(group1, group2)
        return 1 - (2 * u_statistic) / (n1 * n2)
    
    def calculate_epsilon_squared(self, *groups) -> float:
        """Calculate epsilon-squared effect size for Kruskal-Wallis"""
        n_total = sum(len(group) for group in groups)
        h_statistic, _ = stats.kruskal(*groups)
        return (h_statistic - len(groups) + 1) / (n_total - len(groups))
    
    def interpret_effect_size(self, effect_size: float, test_name: str) -> str:
        """Interpret effect size magnitude"""
        if effect_size is None or np.isnan(effect_size):
            return "Effect size not available"
        
        abs_effect = abs(effect_size)
        
        if 'Cohen' in test_name or 't-test' in test_name:
            # Cohen's d interpretation
            if abs_effect < 0.2:
                return f"Small effect (d={effect_size:.3f})"
            elif abs_effect < 0.8:
                return f"Medium effect (d={effect_size:.3f})"
            else:
                return f"Large effect (d={effect_size:.3f})"
        elif 'eta' in str(type(effect_size)).lower() or 'ANOVA' in test_name:
            # Eta-squared interpretation
            if abs_effect < 0.01:
                return f"Small effect (η²={effect_size:.3f})"
            elif abs_effect < 0.06:
                return f"Medium effect (η²={effect_size:.3f})"
            else:
                return f"Large effect (η²={effect_size:.3f})"
        else:
            return f"Effect size = {effect_size:.3f}"

    # ===== CORRELATION ANALYSIS =====
    
    def correlation_analysis(self, columns: List[str] = None, group_col: str = 'group',
                        method: str = 'pearson', alpha: float = 0.05) -> Dict[str, any]:
        """
        Comprehensive correlation analysis
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to analyze for correlations
        group_col : str
            Column for grouping analysis
        method : str
            Correlation method ('pearson', 'spearman', 'kendall')
        alpha : float
            Significance level for multiple testing correction
            
        Returns
        -------
        Dict[str, any]
            Correlation analysis results
        """
        if columns is None:
            columns = self.get_numeric_columns()
        
        # Create cache key
        cache_key = f"{method}_{hash(tuple(columns))}_{group_col}_{alpha}"
        
        # Check if results already cached
        if cache_key in self.correlation_results:
            print(f"📋 Using cached correlation results for {method} analysis")
            return self.correlation_results[cache_key]
        
        print(f"🔗 Performing {method} correlation analysis...")
        
        correlation_results = {
            'method': method,
            'columns': columns,
            'alpha': alpha,
            'overall': {},
            'by_group': {},
            'significant_correlations': []
        }
        
        # Select only numeric columns that exist
        available_columns = [col for col in columns if col in self.regionprops.columns]
        numeric_data = self.regionprops[available_columns].select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            correlation_results['error'] = "Need at least 2 numeric columns for correlation analysis"
            return correlation_results
        
        try:
            # Overall correlation analysis
            print("  📊 Computing overall correlations...")
            
            # Compute correlation matrix and p-values
            if method == 'pearson':
                corr_matrix = numeric_data.corr(method='pearson')
                # Compute p-values for Pearson correlation
                n = len(numeric_data)
                p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
                
                for i, col1 in enumerate(corr_matrix.columns):
                    for j, col2 in enumerate(corr_matrix.columns):
                        if i != j:
                            r = corr_matrix.loc[col1, col2]
                            # t-statistic for Pearson correlation
                            t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else np.inf
                            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                            p_values.loc[col1, col2] = p_val
                        else:
                            p_values.loc[col1, col2] = 0.0  # Perfect correlation with self
            
            elif method == 'spearman':
                corr_matrix = numeric_data.corr(method='spearman')
                # Compute p-values using scipy
                p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
                
                for i, col1 in enumerate(corr_matrix.columns):
                    for j, col2 in enumerate(corr_matrix.columns):
                        if i != j:
                            _, p_val = stats.spearmanr(numeric_data[col1].dropna(), 
                                                     numeric_data[col2].dropna())
                            p_values.loc[col1, col2] = p_val
                        else:
                            p_values.loc[col1, col2] = 0.0
            
            elif method == 'kendall':
                corr_matrix = numeric_data.corr(method='kendall')
                # Compute p-values using scipy
                p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
                
                for i, col1 in enumerate(corr_matrix.columns):
                    for j, col2 in enumerate(corr_matrix.columns):
                        if i != j:
                            _, p_val = stats.kendalltau(numeric_data[col1].dropna(), 
                                                      numeric_data[col2].dropna())
                            p_values.loc[col1, col2] = p_val
                        else:
                            p_values.loc[col1, col2] = 0.0
            
            # Multiple testing correction
            # Get upper triangle p-values (avoid double counting)
            upper_triangle_mask = np.triu(np.ones_like(p_values, dtype=bool), k=1)
            p_values_upper = p_values.values[upper_triangle_mask]
            
            # Apply FDR correction
            try:
                # ✅ Use correct import
                rejected, p_values_adj_array, _, _ = multipletests(p_values_upper, 
                                                                 alpha=alpha, 
                                                                 method='fdr_bh')
                
                # Create adjusted p-value matrix
                p_values_adj = p_values.copy()
                p_values_adj.values[upper_triangle_mask] = p_values_adj_array
                
                # Mirror to lower triangle
                p_values_adj = p_values_adj.fillna(p_values_adj.T)
                
            except Exception as e:
                print(f"  ⚠️ Multiple testing correction failed: {e}")
                p_values_adj = p_values
                rejected = p_values_upper < alpha
            
            # Store overall results
            correlation_results['overall'] = {
                'correlation_matrix': corr_matrix,
                'p_values': p_values,
                'p_values_adjusted': p_values_adj,
                'sample_size': len(numeric_data)
            }
            
            # Find significant correlations
            significant_pairs = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Only upper triangle to avoid duplicates
                        r = corr_matrix.loc[col1, col2]
                        p_adj = p_values_adj.loc[col1, col2]
                        if p_adj < alpha and not pd.isna(r):
                            significant_pairs.append({
                                'variable1': col1,
                                'variable2': col2,
                                'correlation': r,
                                'p_value': p_values.loc[col1, col2],
                                'p_adjusted': p_adj,
                                'significant': True
                            })
            
            correlation_results['significant_correlations'] = significant_pairs
            
            # Group-wise analysis if grouping column exists
            if group_col in self.regionprops.columns:
                print("  📊 Computing group-wise correlations...")
                
                group_results = {}
                for group in self.regionprops[group_col].unique():
                    group_data = self.regionprops[self.regionprops[group_col] == group]
                    group_numeric = group_data[available_columns].select_dtypes(include=[np.number])
                    
                    if len(group_numeric) >= 3 and len(group_numeric.columns) >= 2:  # Need minimum samples
                        try:
                            group_corr = group_numeric.corr(method=method)
                            group_results[group] = {
                                'correlation_matrix': group_corr,
                                'sample_size': len(group_numeric)
                            }
                        except Exception as e:
                            print(f"    ⚠️ Group {group} correlation failed: {e}")
                            group_results[group] = {
                                'error': str(e),
                                'sample_size': len(group_numeric)
                            }
                
                correlation_results['by_group'] = group_results
            
            # Add metadata
            correlation_results['metadata'] = {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'n_observations': len(self.regionprops),
                'n_groups': len(self.regionprops[group_col].unique()) if group_col in self.regionprops.columns else 1,
                'n_variables': len(available_columns),
                'correction_method': 'fdr_bh'
            }
            
            print(f"  ✅ Found {len(significant_pairs)} significant correlations (α = {alpha})")
            
        except Exception as e:
            correlation_results['error'] = f"Correlation analysis failed: {str(e)}"
            print(f"  ❌ Correlation analysis failed: {e}")
            import traceback
            traceback.print_exc()
        
        # ✅ Store results in instance
        self.correlation_results[cache_key] = correlation_results
        
        return correlation_results
        
    def posthoc_tests_comprehensive(self, column: str, group_col: str = 'group', 
                                   alpha: float = 0.05, method: str = 'auto') -> Dict[str, any]:
        """
        Comprehensive post-hoc analysis with caching
        """
        # Create cache key
        cache_key = f"{column}_{group_col}_{method}_{alpha}"
        
        # Check if results already cached
        if cache_key in self.posthoc_results:
            print(f"📋 Using cached post-hoc results for {column}")
            return self.posthoc_results[cache_key]
        
        print(f"🔬 Performing post-hoc analysis for {column}...")
        
        regionprops = self.regionprops
        
        results = {}
        
        try:
            if 'ANOVA' in method or 't-test' in method:
                # Tukey's HSD for parametric tests
                tukey = pairwise_tukeyhsd(
                    endog=regionprops[column].dropna(),
                    groups=regionprops.loc[regionprops[column].dropna(), group_col]
                )
                results['tukey'] = pd.DataFrame(data=tukey.summary().data[1:], 
                                              columns=tukey.summary().data[0])
            else:
                # Dunn's test for non-parametric tests
                try:
                    import scikit_posthocs as sp
                    dunn_results = sp.posthoc_dunn(regionprops, val_col=column, 
                                                 group_col=group_col, p_adjust='bonferroni')
                    results['dunn'] = dunn_results
                except ImportError:
                    # Fallback: Pairwise Mann-Whitney with Bonferroni correction
                    results['pairwise_mannwhitney'] = self.pairwise_mannwhitney(column, group_col)
        
        except Exception as e:
            print(f"Post-hoc test failed for {column}: {e}")
            results['error'] = f"Post-hoc test failed: {e}"
        
        # ✅ Store results in instance
        self.posthoc_results[cache_key] = results
        
        return results

    def get_all_cached_results(self) -> Dict[str, Dict]:
        """
        Get all cached analysis results
        
        Returns
        -------
        Dict[str, Dict]
            Dictionary containing all cached results by analysis type
        """
        return {
            'outliers': self.outlier_results,
            'normality': self.normality_results,
            'comparisons': self.comparison_results,
            'correlations': self.correlation_results,
            'posthoc': self.posthoc_results,
            'effect_sizes': self.effect_sizes
        }

    def clear_all_cache(self) -> None:
        """Clear all cached results"""
        self.outlier_results.clear()
        self.normality_results.clear()
        self.comparison_results.clear()
        self.correlation_results.clear()
        self.posthoc_results.clear()
        self.effect_sizes.clear()
        print("✅ All analysis caches cleared")

    def get_cache_summary(self) -> pd.DataFrame:
        """
        Get summary of all cached results
        
        Returns
        -------
        pd.DataFrame
            Summary of cached analyses
        """
        cache_info = []
        
        for analysis_type, cache_dict in self.get_all_cached_results().items():
            for cache_key, results in cache_dict.items():
                cache_info.append({
                    'Analysis Type': analysis_type.title(),
                    'Cache Key': cache_key,
                    'Cached At': results.get('metadata', {}).get('analysis_date', 'Unknown'),
                    'Results Available': 'Yes' if results else 'No',
                    'Size (approx)': f"{len(str(results))} chars"
                })
        
        return pd.DataFrame(cache_info)

    def has_cached_results(self, analysis_type: str = None) -> bool:
        """
        Check if cached results exist
        
        Parameters
        ----------
        analysis_type : str, optional
            Specific analysis type to check ('outliers', 'correlations', etc.)
            If None, checks if any cached results exist
            
        Returns
        -------
        bool
            True if cached results exist
        """
        all_results = self.get_all_cached_results()
        
        if analysis_type:
            return bool(all_results.get(analysis_type, {}))
        else:
            return any(cache_dict for cache_dict in all_results.values())

    def correlation_summary(self) -> pd.DataFrame:
        """
        Get summary of all correlation analyses performed
        
        Returns
        -------
        pd.DataFrame
            Summary of correlation results
        """
        if not self.correlation_results:
            return pd.DataFrame(columns=['Method', 'Columns', 'Date', 'Significant Pairs'])
        
        summary_data = []
        for cache_key, results in self.correlation_results.items():
            summary_data.append({
                'Cache Key': cache_key,
                'Method': results.get('method', 'Unknown'),
                'Columns': ', '.join(results.get('columns', [])),
                'Analysis Date': results.get('metadata', {}).get('analysis_date', 'Unknown'),
                'N Observations': results.get('metadata', {}).get('n_observations', 'Unknown'),
                'Significant Pairs': len(results.get('significant_correlations', [])),
                'Groups Analyzed': results.get('metadata', {}).get('n_groups', 'Unknown')
            })
        
        return pd.DataFrame(summary_data)


    # ===== SUMMARY AND REPORTING =====
    
    def generate_summary_report(self, columns: List[str] = None) -> Dict[str, any]:
        """
        Generate comprehensive statistical summary report.
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to include in report
            
        Returns
        -------
        Dict[str, any]
            Comprehensive summary report
        """
        if columns is None:
            columns = self.get_numeric_columns()[:10]  # Limit to first 10 for readability
        
        print("🔍 Generating comprehensive statistical report...")
        
        report = {
            'dataset_info': self.get_dataset_info(),
            'descriptive_stats': self.get_descriptive_statistics(columns),
            'outlier_analysis': {},
            'normality_tests': {},
            'group_comparisons': {},
            'correlations': {},
            'recommendations': []
        }
        
        # Outlier analysis
        print("  📊 Analyzing outliers...")
        report['outlier_analysis']['iqr'] = self.detect_outliers_iqr(columns)
        report['outlier_analysis']['zscore'] = self.detect_outliers_zscore(columns)
        
        # Normality tests
        print("  📈 Testing normality...")
        report['normality_tests'] = self.test_normality(columns)
        
        # Group comparisons
        print("  🔬 Comparing groups...")
        report['group_comparisons'] = self.compare_groups_comprehensive(columns)
        
        # Correlations
        print("  🔗 Analyzing correlations...")
        report['correlations'] = self.correlation_analysis(columns)
        
        # Generate recommendations
        report['recommendations'] = self.generate_recommendations(report)
        
        print("✅ Statistical analysis complete!")
        return report
    
    def get_dataset_info(self) -> Dict[str, any]:
        """Get basic dataset information"""
        regionprops = self.regionprops
        numeric_cols = self.get_numeric_columns()
        
        return {
            'total_observations': len(regionprops),
            'numeric_columns': len(numeric_cols),
            'groups': regionprops['group'].value_counts().to_dict() if 'group' in regionprops.columns else {},
            'missing_data': regionprops[numeric_cols].isnull().sum().to_dict(),
            'data_types': regionprops[numeric_cols].dtypes.to_dict()
        }
    
    def get_descriptive_statistics(self, columns: List[str]) -> pd.DataFrame:
        """Get descriptive statistics for specified columns"""
        regionprops = self.regionprops
        
        if 'group' in regionprops.columns:
            return regionprops.groupby('group')[columns].describe()
        else:
            return regionprops[columns].describe()
    
    def generate_recommendations(self, report: Dict) -> List[str]:
        """Generate statistical recommendations based on analysis"""
        recommendations = []
        
        # Check outliers
        outlier_counts = []
        for method, results in report['outlier_analysis'].items():
            if isinstance(results, dict):
                total_outliers = sum(r.outlier_count for r in results.values() if hasattr(r, 'outlier_count'))
                outlier_counts.append(total_outliers)
        
        if outlier_counts and max(outlier_counts) > len(self.regionprops) * 0.1:
            recommendations.append("⚠️ High number of outliers detected. Consider investigating data quality.")
        
        # Check normality
        non_normal_count = 0
        for col_results in report['normality_tests'].values():
            for group_result in col_results.values():
                if not group_result.assumptions_met:
                    non_normal_count += 1
        
        if non_normal_count > 0:
            recommendations.append("📊 Some groups show non-normal distributions. Non-parametric tests were used where appropriate.")
        
        # Check group comparisons
        significant_comparisons = []
        for col, results in report['group_comparisons'].items():
            if 'comparison' in results and results['comparison'].p_value < 0.05:
                significant_comparisons.append(col)
        
        if significant_comparisons:
            recommendations.append(f"🎯 Significant group differences found in: {', '.join(significant_comparisons[:5])}")
        
        # Check effect sizes
        large_effects = []
        for col, results in report['group_comparisons'].items():
            if 'comparison' in results and results['comparison'].effect_size:
                if abs(results['comparison'].effect_size) > 0.8:  # Large effect for Cohen's d
                    large_effects.append(col)
        
        if large_effects:
            recommendations.append(f"💪 Large effect sizes detected in: {', '.join(large_effects[:3])}")
        
        return recommendations

    def print_summary_report(self, columns: List[str] = None):
        """Print a formatted summary report"""
        report = self.generate_summary_report(columns)
        
        print("\n" + "="*80)
        print("📊 EXPERIMENT STATISTICAL ANALYSIS REPORT")
        print("="*80)
        
        # Dataset info
        info = report['dataset_info']
        print(f"\n📋 Dataset Information:")
        print(f"  Total observations: {info['total_observations']:,}")
        print(f"  Numeric columns: {info['numeric_columns']}")
        print(f"  Groups: {', '.join([f'{k}({v})' for k, v in info['groups'].items()])}")
        
        # Missing data
        missing = info['missing_data']
        if any(missing.values()):
            print(f"  Missing data: {sum(missing.values())} values across {sum(1 for v in missing.values() if v > 0)} columns")
        
        # Outliers summary
        print(f"\n🎯 Outlier Analysis:")
        for method, results in report['outlier_analysis'].items():
            if isinstance(results, dict):
                total_outliers = sum(r.outlier_count for r in results.values() if hasattr(r, 'outlier_count'))
                print(f"  {method.upper()}: {total_outliers} outliers detected")
        
        # Group comparisons summary
        print(f"\n🔬 Group Comparisons:")
        significant_tests = []
        for col, results in report['group_comparisons'].items():
            if 'comparison' in results:
                result = results['comparison']
                if result.p_value < 0.05:
                    significant_tests.append((col, result.test_name, result.p_value, result.effect_size))
        
        if significant_tests:
            print(f"  Significant differences found in {len(significant_tests)} variables:")
            for col, test, p_val, effect in significant_tests[:5]:  # Show first 5
                effect_str = f", effect={effect:.3f}" if effect else ""
                print(f"    {col}: {test}, p={p_val:.4f}{effect_str}")
        else:
            print("  No significant group differences detected")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        print("\n" + "="*80)
    # ===== DATA SAVING AND EXPORTING =====

    def save(self, filepath: Union[str, Path], method: str = 'json', 
             include_results: bool = True, compress: bool = True) -> None:
        """
        Save ExperimentStatistics object
        
        Parameters
        ----------
        filepath : str or Path
            Path to save file
        method : str
            Save method ('pickle', 'json', 'hdf5', 'separate')
        include_results : bool
            Whether to include cached analysis results
        compress : bool
            Whether to compress the saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if method == 'pickle':
            self._save_pickle(filepath, include_results, compress)
        elif method == 'json':
            self._save_json(filepath, include_results)
        elif method == 'hdf5':
            self._save_hdf5(filepath, include_results)
        elif method == 'separate':
            self._save_separate_files(filepath, include_results)
        else:
            raise ValueError(f"Unknown save method: {method}")
        
        print(f"✅ ExperimentStatistics saved to: {filepath}")

    def _save_pickle(self, filepath: Path, include_results: bool, compress: bool) -> None:
        """Save as pickle file"""
        import pickle
        
        # Prepare data
        save_data = {
            'experiment_data': self.regionprops,
            'metadata': {
                'created_at': pd.Timestamp.now().isoformat(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
                'pandas_version': pd.__version__,
                'include_results': include_results
            }
        }
        
        # Add cached results if requested
        if include_results and hasattr(self, '_cached_results'):
            save_data['cached_results'] = self._cached_results
        
        # Save with or without compression
        if compress:
            import gzip
            filepath = filepath.with_suffix('.pkl.gz')
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            filepath = filepath.with_suffix('.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_json(self, filepath: Path, include_results: bool) -> None:
        """Save as JSON file (metadata + results, data as CSV)"""
        import json
        
        # Save experiment data as CSV
        csv_path = filepath.with_suffix('.csv')
        self.regionprops.to_csv(csv_path, index=False)
        
        # Save metadata as JSON
        json_path = filepath.with_suffix('.json')
        
        # Get column information safely
        try:
            numeric_columns = self.get_numeric_columns()
        except:
            numeric_columns = list(self.regionprops.select_dtypes(include=[np.number]).columns)
        
        try:
            categorical_columns = self.get_categorical_columns()
        except:
            categorical_columns = list(self.regionprops.select_dtypes(include=['object', 'category']).columns)
        
        metadata = {
            'data_file': str(csv_path.name),
            'created_at': pd.Timestamp.now().isoformat(),
            'n_observations': len(self.regionprops),
            'columns': list(self.regionprops.columns),
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns,
            'data_shape': list(self.regionprops.shape),
            'column_dtypes': {col: str(dtype) for col, dtype in self.regionprops.dtypes.items()}
        }
        
        # Add summary statistics for numeric columns
        if numeric_columns:
            try:
                metadata['summary_stats'] = self.regionprops[numeric_columns].describe().to_dict()
            except Exception as e:
                metadata['summary_stats_error'] = str(e)
        
        # Add FULL cached results if requested
        if include_results:
            try:
                metadata['cached_analyses_keys'] = {
                    'outliers': list(self.outlier_results.keys()) if hasattr(self, 'outlier_results') else [],
                    'normality': list(self.normality_results.keys()) if hasattr(self, 'normality_results') else [],
                    'comparisons': list(self.comparison_results.keys()) if hasattr(self, 'comparison_results') else [],
                    'correlations': list(self.correlation_results.keys()) if hasattr(self, 'correlation_results') else [],
                    'posthoc': list(self.posthoc_results.keys()) if hasattr(self, 'posthoc_results') else []
                }
                
                # ✅ NOW SAVE THE ACTUAL RESULTS
                metadata['analysis_results'] = {}
                
                # Save outlier results
                if hasattr(self, 'outlier_results') and self.outlier_results:
                    metadata['analysis_results']['outliers'] = self._serialize_outlier_results()
                
                # Save normality results  
                if hasattr(self, 'normality_results') and self.normality_results:
                    metadata['analysis_results']['normality'] = self._serialize_normality_results()
                
                # Save comparison results
                if hasattr(self, 'comparison_results') and self.comparison_results:
                    metadata['analysis_results']['comparisons'] = self._serialize_comparison_results()
                
                # Save correlation results
                if hasattr(self, 'correlation_results') and self.correlation_results:
                    metadata['analysis_results']['correlations'] = self._serialize_correlation_results()
                
                # Save post-hoc results
                if hasattr(self, 'posthoc_results') and self.posthoc_results:
                    metadata['analysis_results']['posthoc'] = self._serialize_posthoc_results()
                    
            except Exception as e:
                metadata['cached_analyses_error'] = str(e)
                import traceback
                metadata['cached_analyses_traceback'] = traceback.format_exc()
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)  # default=str handles non-serializable objects
        
        print(f"✅ Data saved to: {csv_path}")
        print(f"✅ Metadata saved to: {json_path}")

    def _save_hdf5(self, filepath: Path, include_results: bool) -> None:
        """Save as HDF5 file"""
        try:
            import h5py
            
            filepath = filepath.with_suffix('.h5')
            
            with h5py.File(filepath, 'w') as f:
                # Save main data
                df = self.regionprops
                
                # Store each column appropriately
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Convert strings to bytes for HDF5
                        string_data = df[col].astype(str).values
                        f.create_dataset(f'data/{col}', data=string_data.astype('S'))
                    else:
                        f.create_dataset(f'data/{col}', data=df[col].values)
                
                # Store metadata
                f.attrs['created_at'] = pd.Timestamp.now().isoformat()
                f.attrs['n_observations'] = len(df)
                f.attrs['columns'] = [col.encode() for col in df.columns]
                
                # Store cached results if available
                if include_results and hasattr(self, '_cached_results'):
                    results_group = f.create_group('cached_results')
                    # Store results (would need custom serialization for complex objects)
                    
        except ImportError:
            raise ImportError("h5py required for HDF5 saving. Install with: pip install h5py")

    def _save_separate_files(self, filepath: Path, include_results: bool) -> None:
        """Save as separate files in a directory"""
        save_dir = filepath.with_suffix('')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main data
        data_path = save_dir / 'experiment_data.csv'
        self.regionprops.to_csv(data_path, index=False)
        
        # Save metadata
        metadata = {
            'created_at': pd.Timestamp.now().isoformat(),
            'data_file': 'experiment_data.csv',
            'n_observations': len(self.regionprops),
            'columns': list(self.regionprops.columns),
            'numeric_columns': self.get_numeric_columns(),
            'categorical_columns': self.get_categorical_columns()
        }
        
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save analysis results separately if requested
        if include_results:
            results_dir = save_dir / 'analysis_results'
            results_dir.mkdir(exist_ok=True)
            
            # Example: Save correlation matrices, test results, etc.
            # This would be implemented based on what results you want to cache
    def _serialize_outlier_results(self) -> Dict[str, any]:
        """Serialize outlier results for JSON storage"""
        serialized = {}
        
        for method, results in self.outlier_results.items():
            serialized[method] = {}
            
            if isinstance(results, dict):
                for col, outlier_analysis in results.items():
                    if hasattr(outlier_analysis, '__dict__'):  # OutlierAnalysis object
                        serialized[method][col] = {
                            'method': outlier_analysis.method,
                            'outlier_count': outlier_analysis.outlier_count,
                            'outlier_percentage': outlier_analysis.outlier_percentage,
                            'outlier_indices': outlier_analysis.outlier_indices,
                            'threshold_lower': outlier_analysis.threshold_lower,
                            'threshold_upper': outlier_analysis.threshold_upper,
                            'z_scores': outlier_analysis.z_scores.tolist() if outlier_analysis.z_scores is not None else None
                        }
                    else:
                        serialized[method][col] = str(outlier_analysis)
            elif hasattr(results, '__dict__'):  # Single OutlierAnalysis object
                serialized[method] = {
                    'method': results.method,
                    'outlier_count': results.outlier_count,
                    'outlier_percentage': results.outlier_percentage,
                    'outlier_indices': results.outlier_indices,
                    'threshold_lower': results.threshold_lower,
                    'threshold_upper': results.threshold_upper,
                    'z_scores': results.z_scores.tolist() if results.z_scores is not None else None
                }
        
        return serialized

    def _serialize_normality_results(self) -> Dict[str, any]:
        """Serialize normality test results for JSON storage"""
        serialized = {}
        
        for col, col_results in self.normality_results.items():
            serialized[col] = {}
            
            for group, test_result in col_results.items():
                if hasattr(test_result, '__dict__'):  # StatisticalTestResult object
                    serialized[col][group] = {
                        'test_name': test_result.test_name,
                        'statistic': test_result.statistic,
                        'p_value': test_result.p_value,
                        'effect_size': test_result.effect_size,
                        'confidence_interval': test_result.confidence_interval,
                        'interpretation': test_result.interpretation,
                        'assumptions_met': test_result.assumptions_met,
                        'sample_size': test_result.sample_size,
                        'power': test_result.power
                    }
                else:
                    serialized[col][group] = str(test_result)
        
        return serialized

    def _serialize_comparison_results(self) -> Dict[str, any]:
        """Serialize group comparison results for JSON storage"""
        serialized = {}
        
        for col, col_results in self.comparison_results.items():
            serialized[col] = {}
            
            for test_type, test_result in col_results.items():
                if hasattr(test_result, '__dict__'):  # StatisticalTestResult object
                    serialized[col][test_type] = {
                        'test_name': test_result.test_name,
                        'statistic': test_result.statistic,
                        'p_value': test_result.p_value,
                        'effect_size': test_result.effect_size,
                        'confidence_interval': test_result.confidence_interval,
                        'interpretation': test_result.interpretation,
                        'assumptions_met': test_result.assumptions_met,
                        'sample_size': test_result.sample_size,
                        'power': test_result.power
                    }
                elif isinstance(test_result, dict):
                    # Handle nested results (like post-hoc tests)
                    serialized[col][test_type] = {}
                    for key, value in test_result.items():
                        if isinstance(value, pd.DataFrame):
                            serialized[col][test_type][key] = value.to_dict('records')
                        else:
                            serialized[col][test_type][key] = value
                else:
                    serialized[col][test_type] = str(test_result)
        
        return serialized

    def _serialize_correlation_results(self) -> Dict[str, any]:
        """Serialize correlation analysis results for JSON storage"""
        serialized = {}
        
        for cache_key, results in self.correlation_results.items():
            serialized[cache_key] = {}
            
            # Basic information
            for key in ['method', 'columns', 'alpha']:
                if key in results:
                    serialized[cache_key][key] = results[key]
            
            # Overall results
            if 'overall' in results:
                overall = results['overall']
                serialized[cache_key]['overall'] = {}
                
                if 'correlation_matrix' in overall:
                    serialized[cache_key]['overall']['correlation_matrix'] = overall['correlation_matrix'].to_dict()
                
                if 'p_values' in overall:
                    serialized[cache_key]['overall']['p_values'] = overall['p_values'].to_dict()
                
                if 'p_values_adjusted' in overall:
                    serialized[cache_key]['overall']['p_values_adjusted'] = overall['p_values_adjusted'].to_dict()
                
                if 'sample_size' in overall:
                    serialized[cache_key]['overall']['sample_size'] = overall['sample_size']
            
            # Significant correlations
            if 'significant_correlations' in results:
                serialized[cache_key]['significant_correlations'] = results['significant_correlations']
            
            # Group-wise results
            if 'by_group' in results:
                serialized[cache_key]['by_group'] = {}
                for group, group_results in results['by_group'].items():
                    serialized[cache_key]['by_group'][group] = {}
                    
                    if 'correlation_matrix' in group_results:
                        serialized[cache_key]['by_group'][group]['correlation_matrix'] = group_results['correlation_matrix'].to_dict()
                    
                    if 'sample_size' in group_results:
                        serialized[cache_key]['by_group'][group]['sample_size'] = group_results['sample_size']
                    
                    if 'error' in group_results:
                        serialized[cache_key]['by_group'][group]['error'] = group_results['error']
            
            # Metadata
            if 'metadata' in results:
                serialized[cache_key]['metadata'] = results['metadata']
        
        return serialized

    def _serialize_posthoc_results(self) -> Dict[str, any]:
        """Serialize post-hoc test results for JSON storage"""
        serialized = {}
        
        for cache_key, results in self.posthoc_results.items():
            serialized[cache_key] = {}
            
            for test_type, test_results in results.items():
                if isinstance(test_results, pd.DataFrame):
                    serialized[cache_key][test_type] = test_results.to_dict('records')
                elif isinstance(test_results, dict):
                    serialized[cache_key][test_type] = {}
                    for key, value in test_results.items():
                        if isinstance(value, pd.DataFrame):
                            serialized[cache_key][test_type][key] = value.to_dict('records')
                        else:
                            serialized[cache_key][test_type][key] = value
                else:
                    serialized[cache_key][test_type] = str(test_results)
        
        return serialized
 

    def _deserialize_outlier_results(self, serialized_data: Dict) -> Dict:
        """Reconstruct outlier results from serialized data"""
        outlier_results = {}
        
        for method, method_results in serialized_data.items():
            if isinstance(method_results, dict):
                outlier_results[method] = {}
                for col, result_data in method_results.items():
                    if isinstance(result_data, dict) and 'method' in result_data:
                        # Reconstruct OutlierAnalysis object
                        outlier_results[method][col] = OutlierAnalysis(
                            method=result_data.get('method', ''),
                            outlier_indices=result_data.get('outlier_indices', []),
                            outlier_count=result_data.get('outlier_count', 0),
                            outlier_percentage=result_data.get('outlier_percentage', 0.0),
                            threshold_lower=result_data.get('threshold_lower'),
                            threshold_upper=result_data.get('threshold_upper'),
                            z_scores=np.array(result_data['z_scores']) if result_data.get('z_scores') else None
                        )
                    else:
                        outlier_results[method][col] = result_data
        
        return outlier_results

    def _deserialize_normality_results(self, serialized_data: Dict) -> Dict:
        """Reconstruct normality results from serialized data"""
        normality_results = {}
        
        for col, col_results in serialized_data.items():
            normality_results[col] = {}
            for group, result_data in col_results.items():
                if isinstance(result_data, dict) and 'test_name' in result_data:
                    # Reconstruct StatisticalTestResult object
                    normality_results[col][group] = StatisticalTestResult(
                        test_name=result_data.get('test_name', ''),
                        statistic=result_data.get('statistic', 0.0),
                        p_value=result_data.get('p_value', 1.0),
                        effect_size=result_data.get('effect_size'),
                        confidence_interval=tuple(result_data['confidence_interval']) if result_data.get('confidence_interval') else None,
                        interpretation=result_data.get('interpretation', ''),
                        assumptions_met=result_data.get('assumptions_met', False),
                        sample_size=result_data.get('sample_size', 0),
                        power=result_data.get('power')
                    )
                else:
                    normality_results[col][group] = result_data
        
        return normality_results

    def _deserialize_comparison_results(self, serialized_data: Dict) -> Dict:
        """Reconstruct comparison results from serialized data"""
        comparison_results = {}
        
        for col, col_results in serialized_data.items():
            comparison_results[col] = {}
            for test_type, result_data in col_results.items():
                if isinstance(result_data, dict) and 'test_name' in result_data:
                    # Reconstruct StatisticalTestResult object
                    comparison_results[col][test_type] = StatisticalTestResult(
                        test_name=result_data.get('test_name', ''),
                        statistic=result_data.get('statistic', 0.0),
                        p_value=result_data.get('p_value', 1.0),
                        effect_size=result_data.get('effect_size'),
                        confidence_interval=tuple(result_data['confidence_interval']) if result_data.get('confidence_interval') else None,
                        interpretation=result_data.get('interpretation', ''),
                        assumptions_met=result_data.get('assumptions_met', False),
                        sample_size=result_data.get('sample_size', 0),
                        power=result_data.get('power')
                    )
                elif isinstance(result_data, dict):
                    # Handle nested dictionary results
                    comparison_results[col][test_type] = {}
                    for key, value in result_data.items():
                        if isinstance(value, list) and value and isinstance(value[0], dict):
                            # Convert list of dicts back to DataFrame
                            comparison_results[col][test_type][key] = pd.DataFrame(value)
                        else:
                            comparison_results[col][test_type][key] = value
                else:
                    comparison_results[col][test_type] = result_data
        
        return comparison_results

    def _deserialize_correlation_results(self, serialized_data: Dict) -> Dict:
        """Reconstruct correlation results from serialized data"""
        correlation_results = {}
        
        for cache_key, results in serialized_data.items():
            correlation_results[cache_key] = {}
            
            # Basic information
            for key in ['method', 'columns', 'alpha', 'significant_correlations', 'metadata']:
                if key in results:
                    correlation_results[cache_key][key] = results[key]
            
            # Reconstruct overall results
            if 'overall' in results:
                correlation_results[cache_key]['overall'] = {}
                overall = results['overall']
                
                if 'correlation_matrix' in overall:
                    correlation_results[cache_key]['overall']['correlation_matrix'] = pd.DataFrame(overall['correlation_matrix'])
                
                if 'p_values' in overall:
                    correlation_results[cache_key]['overall']['p_values'] = pd.DataFrame(overall['p_values'])
                
                if 'p_values_adjusted' in overall:
                    correlation_results[cache_key]['overall']['p_values_adjusted'] = pd.DataFrame(overall['p_values_adjusted'])
                
                if 'sample_size' in overall:
                    correlation_results[cache_key]['overall']['sample_size'] = overall['sample_size']
            
            # Reconstruct group-wise results
            if 'by_group' in results:
                correlation_results[cache_key]['by_group'] = {}
                for group, group_results in results['by_group'].items():
                    correlation_results[cache_key]['by_group'][group] = {}
                    
                    if 'correlation_matrix' in group_results:
                        correlation_results[cache_key]['by_group'][group]['correlation_matrix'] = pd.DataFrame(group_results['correlation_matrix'])
                    
                    for key in ['sample_size', 'error']:
                        if key in group_results:
                            correlation_results[cache_key]['by_group'][group][key] = group_results[key]
        
        return correlation_results

    def _deserialize_posthoc_results(self, serialized_data: Dict) -> Dict:
        """Reconstruct post-hoc results from serialized data"""
        posthoc_results = {}
        
        for cache_key, results in serialized_data.items():
            posthoc_results[cache_key] = {}
            
            for test_type, test_results in results.items():
                if isinstance(test_results, list) and test_results and isinstance(test_results[0], dict):
                    # Convert list of dicts back to DataFrame
                    posthoc_results[cache_key][test_type] = pd.DataFrame(test_results)
                elif isinstance(test_results, dict):
                    posthoc_results[cache_key][test_type] = {}
                    for key, value in test_results.items():
                        if isinstance(value, list) and value and isinstance(value[0], dict):
                            # Convert list of dicts back to DataFrame
                            posthoc_results[cache_key][test_type][key] = pd.DataFrame(value)
                        else:
                            posthoc_results[cache_key][test_type][key] = value
                else:
                    posthoc_results[cache_key][test_type] = test_results
        
        return posthoc_results

    # Add a method to view loaded results:
    def print_loaded_results_summary(self) -> None:
        """Print summary of all loaded analysis results"""
        print("\n📋 LOADED ANALYSIS RESULTS SUMMARY")
        print("=" * 50)
        
        all_results = self.get_all_cached_results()
        
        for analysis_type, results in all_results.items():
            if results:
                print(f"\n🔬 {analysis_type.upper()}:")
                
                if analysis_type == 'outliers':
                    for method, method_results in results.items():
                        if isinstance(method_results, dict):
                            total_outliers = sum(
                                r.outlier_count for r in method_results.values() 
                                if hasattr(r, 'outlier_count')
                            )
                            print(f"  {method}: {total_outliers} total outliers across {len(method_results)} columns")
                
                elif analysis_type == 'correlations':
                    for cache_key, corr_results in results.items():
                        method = corr_results.get('method', 'unknown')
                        n_significant = len(corr_results.get('significant_correlations', []))
                        columns = corr_results.get('columns', [])
                        print(f"  {method}: {n_significant} significant correlations among {len(columns)} variables")
                
                elif analysis_type == 'comparisons':
                    for col, comp_results in results.items():
                        if 'comparison' in comp_results:
                            comp = comp_results['comparison']
                            test_name = comp.test_name if hasattr(comp, 'test_name') else str(comp)
                            p_val = comp.p_value if hasattr(comp, 'p_value') else 'unknown'
                            print(f"  {col}: {test_name}, p={p_val}")
                
                else:
                    print(f"  {len(results)} cached result(s)")
            else:
                print(f"\n🔬 {analysis_type.upper()}: No results")
    @classmethod
    def load(cls, filepath: Union[str, Path], experiment_class=None):
        """
        Load ExperimentStatistics object
        
        Parameters
        ----------
        filepath : str or Path
            Path to saved file
        experiment_class : class, optional
            Experiment class to use. If None, creates a mock experiment.
            
        Returns
        -------
        ExperimentStatistics
            Loaded statistics object
        """
        filepath = Path(filepath)
        
        # Detect file type
        if filepath.suffix == '.pkl' or filepath.suffix == '.gz':
            return cls._load_pickle(filepath, experiment_class)
        elif filepath.suffix == '.json':
            return cls._load_json(filepath, experiment_class)
        elif filepath.suffix == '.h5':
            return cls._load_hdf5(filepath, experiment_class)
        elif filepath.is_dir():
            return cls._load_separate_files(filepath, experiment_class)
        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")

    @classmethod
    def _load_pickle(cls, filepath: Path, experiment_class):
        """Load from pickle file"""
        import pickle
        import gzip
        
        # Handle compressed files
        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        
        # Reconstruct experiment
        if experiment_class:
            experiment = experiment_class()
            regionprops = data['experiment_data']
        else:
            # Create mock experiment
            class MockExperiment:
                def __init__(self, regionprops):
                    self.regionprops = regionprops
            experiment = MockExperiment(data['experiment_data'])
        
        # Create statistics object
        stats = cls(experiment)
        
        # Restore cached results if available
        if 'cached_results' in data:
            stats._cached_results = data['cached_results']
        
        print(f"✅ Loaded ExperimentStatistics from: {filepath}")
        print(f"   Data shape: {data['experiment_data'].shape}")
        print(f"   Created: {data['metadata'].get('created_at', 'Unknown')}")
        
        return stats

    @classmethod
    def _load_json(cls, filepath: Path):
        """Load from JSON + CSV files with full result reconstruction"""
        import json
        
        # Load metadata
        json_path = filepath.with_suffix('.json')
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        # Load data
        csv_path = filepath.parent / metadata['data_file']
        regionprops = pd.read_csv(csv_path)
        
        # Create statistics object
        stats = cls(regionprops)
        
        # ✅ RECONSTRUCT THE ACTUAL RESULTS
        if 'analysis_results' in metadata:
            analysis_results = metadata['analysis_results']
            
            # Reconstruct outlier results
            if 'outliers' in analysis_results:
                stats.outlier_results = stats._deserialize_outlier_results(analysis_results['outliers'])
            
            # Reconstruct normality results
            if 'normality' in analysis_results:
                stats.normality_results = stats._deserialize_normality_results(analysis_results['normality'])
            
            # Reconstruct comparison results
            if 'comparisons' in analysis_results:
                stats.comparison_results = stats._deserialize_comparison_results(analysis_results['comparisons'])
            
            # Reconstruct correlation results
            if 'correlations' in analysis_results:
                stats.correlation_results = stats._deserialize_correlation_results(analysis_results['correlations'])
            
            # Reconstruct post-hoc results
            if 'posthoc' in analysis_results:
                stats.posthoc_results = stats._deserialize_posthoc_results(analysis_results['posthoc'])
        
        print(f"✅ Loaded ExperimentStatistics from: {json_path}")
        print(f"   Data shape: {regionprops.shape}")
        print(f"   Created: {metadata.get('created_at', 'Unknown')}")
        
        # Show what was loaded
        if 'analysis_results' in metadata:
            loaded_analyses = []
            for analysis_type, results in metadata['analysis_results'].items():
                if results:
                    loaded_analyses.append(f"{analysis_type}({len(results)} items)")
            if loaded_analyses:
                print(f"   Loaded analyses: {', '.join(loaded_analyses)}")
        
        return stats

    @classmethod
    def _load_separate_files(cls, filepath: Path):
        """Load from separate files directory"""
        import json
        
        # Load metadata
        with open(filepath / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load data
        data_path = filepath / metadata['data_file']
        regionprops = pd.read_csv(data_path)
        
        stats = cls(regionprops)
        print(f"✅ Loaded ExperimentStatistics from: {filepath}")
        print(f"   Data shape: {regionprops.shape}")
        
        return stats

    # Add caching functionality to store analysis results
    def _get_cache_key(self, method_name: str, *args, **kwargs) -> str:
        """Generate cache key for analysis results"""
        import hashlib
        key_data = f"{method_name}_{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _cache_result(self, method_name: str, result, *args, **kwargs) -> None:
        """Cache analysis result"""
        if not hasattr(self, '_cached_results'):
            self._cached_results = {}
        
        cache_key = self._get_cache_key(method_name, *args, **kwargs)
        self._cached_results[cache_key] = {
            'result': result,
            'timestamp': pd.Timestamp.now().isoformat(),
            'method': method_name,
            'args': args,
            'kwargs': kwargs
        }

    def _get_cached_result(self, method_name: str, *args, **kwargs):
        """Get cached analysis result"""
        if not hasattr(self, '_cached_results'):
            return None
        
        cache_key = self._get_cache_key(method_name, *args, **kwargs)
        return self._cached_results.get(cache_key, {}).get('result')

    def clear_cache(self) -> None:
        """Clear all cached results"""
        if hasattr(self, '_cached_results'):
            self._cached_results.clear()
        print("✅ Cache cleared")

    def get_cache_info(self) -> pd.DataFrame:
        """Get information about cached results"""
        if not hasattr(self, '_cached_results') or not self._cached_results:
            return pd.DataFrame(columns=['Method', 'Timestamp', 'Args', 'Kwargs'])
        
        cache_info = []
        for cache_key, data in self._cached_results.items():
            cache_info.append({
                'Cache Key': cache_key[:10] + '...',
                'Method': data['method'],
                'Timestamp': data['timestamp'],
                'Args': str(data['args'])[:50] + '...' if len(str(data['args'])) > 50 else str(data['args']),
                'Kwargs': str(data['kwargs'])[:50] + '...' if len(str(data['kwargs'])) > 50 else str(data['kwargs'])
            })
        
        return pd.DataFrame(cache_info)

# ===== TEST FUNCTIONS =====
# Replace the test functions section with this improved version:

# ===== TEST FUNCTIONS =====

def create_test_experiment() -> "Experiment":
    """Create a test experiment with synthetic data"""
    from experiment import Experiment
    
    # Create mock experiment with synthetic data
    class TestExperiment:
        def __init__(self):
            self.regionprops = create_synthetic_test_data()
    
    return TestExperiment()

def create_synthetic_test_data() -> pd.DataFrame:
    """Create synthetic test dataset for statistical analysis"""
    np.random.seed(42)
    
    n_samples = 300
    
    # Create groups with different characteristics
    data = []
    
    # Control group - normal distribution
    for i in range(100):
        data.append({
            'group': 'Control',
            'sample': f'Control_S{i//10}',
            'area': np.random.normal(100, 15),
            'perimeter': np.random.normal(40, 8),
            'eccentricity': np.random.beta(2, 5),
            'solidity': np.random.beta(8, 2),
            'aspect_ratio': np.random.lognormal(0, 0.3),
            'frame': np.random.randint(0, 20)
        })
    
    # Treatment 1 - shifted mean, different variance
    for i in range(100):
        data.append({
            'group': 'Treatment_1',
            'sample': f'Treatment1_S{i//10}',
            'area': np.random.normal(120, 20),  # Higher mean, higher variance
            'perimeter': np.random.normal(45, 12),
            'eccentricity': np.random.beta(3, 4),
            'solidity': np.random.beta(6, 3),
            'aspect_ratio': np.random.lognormal(0.2, 0.4),
            'frame': np.random.randint(0, 20)
        })
    
    # Treatment 2 - non-normal distribution
    for i in range(100):
        data.append({
            'group': 'Treatment_2',
            'sample': f'Treatment2_S{i//10}',
            'area': np.random.exponential(80) + 50,  # Skewed distribution
            'perimeter': np.random.gamma(2, 20),
            'eccentricity': np.random.beta(1, 8),
            'solidity': np.random.uniform(0.5, 1.0),
            'aspect_ratio': np.random.exponential(1) + 0.5,
            'frame': np.random.randint(0, 20)
        })
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, 10, replace=False)
    df = pd.DataFrame(data)
    for idx in outlier_indices:
        df.loc[idx, 'area'] *= 3  # Make it an outlier
    
    return df

def load_real_test_experiment() -> "Experiment":
    """Load real experiment data for testing"""
    try:
        from experiment import Experiment
        experiment = Experiment()
        experiment.load_test_data()
        return experiment
    except Exception as e:
        print(f"⚠️ Could not load real experiment data: {e}")
        print("   Using synthetic data instead...")
        return create_test_experiment()

def test_outlier_detection(regionprops: pd.DataFrame, columns: List[str] = None):
    """
    Test outlier detection methods
    
    Parameters
    ----------
    regionprops : pd.DataFrame
        DataFrame containing region properties.
    columns : List[str], optional
        Columns to test. If None, uses default numeric columns.
    """
    print("\n" + "="*50)
    print("🧪 TESTING OUTLIER DETECTION")
    print("="*50)
    
    if regionprops is None:
        regionprops = create_test_experiment().regionprops
        print("📊 Using synthetic test data")
    else:
        print("📊 Using provided experiment data")

    stats_analyzer = ExperimentStatistics(regionprops)

    if columns is None:
        columns = stats_analyzer.get_numeric_columns()[:4]  # Test first 4 columns
    
    print(f"   Testing columns: {columns}")
    print(f"   Dataset size: {len(regionprops.index)} observations")
    
    # Test IQR method
    print("\n📊 Testing IQR outlier detection...")
    try:
        iqr_results = stats_analyzer.detect_outliers_iqr(columns)
        for col, result in iqr_results.items():
            print(f"  {col}: {result.outlier_count} outliers ({result.outlier_percentage:.1f}%)")
    except Exception as e:
        print(f"  ❌ IQR detection failed: {e}")
    
    # Test Z-score method
    print("\n📊 Testing Z-score outlier detection...")
    try:
        zscore_results = stats_analyzer.detect_outliers_zscore(columns)
        for col, result in zscore_results.items():
            print(f"  {col}: {result.outlier_count} outliers ({result.outlier_percentage:.1f}%)")
    except Exception as e:
        print(f"  ❌ Z-score detection failed: {e}")
    
    # Test Isolation Forest
    print("\n📊 Testing Isolation Forest outlier detection...")
    try:
        isolation_result = stats_analyzer.detect_outliers_isolation_forest(columns)
        print(f"  Multivariate outliers: {isolation_result.outlier_count} ({isolation_result.outlier_percentage:.1f}%)")
    except ImportError:
        print("  ⚠️ Isolation Forest requires scikit-learn (not available)")
    except Exception as e:
        print(f"  ❌ Isolation Forest failed: {e}")
    
    # Test outlier removal
    print("\n🧹 Testing outlier removal...")
    try:
        original_size = len(regionprops)
        clean_data = stats_analyzer.remove_outliers(method='iqr', columns=columns, inplace=False)
        removed_count = original_size - len(clean_data)
        print(f"  Removed {removed_count} outliers ({removed_count/original_size*100:.1f}%)")
    except Exception as e:
        print(f"  ❌ Outlier removal failed: {e}")
    
    print("✅ Outlier detection tests completed!")

def test_normality_testing(regionprops: pd.DataFrame = None, columns: List[str] = None):
    """
    Test normality testing methods
    
    Parameters
    ----------
    regionprops : pd.DataFrame, optional
        DataFrame containing region properties. If None, creates synthetic test data.
    columns : List[str], optional
        Columns to test. If None, uses default numeric columns.
    """
    print("\n" + "="*50)
    print("🧪 TESTING NORMALITY ASSESSMENT")
    print("="*50)
    
    if regionprops is None:
        regionprops = create_test_experiment().regionprops
        print("📊 Using synthetic test data")
    else:
        print("📊 Using provided experiment data")

    stats_analyzer = ExperimentStatistics(regionprops)
    
    if columns is None:
        columns = stats_analyzer.get_numeric_columns()[:3]  # Test first 3 columns
    
    print(f"   Testing columns: {columns}")
    print(f"   Groups: {list(regionprops['group'].unique()) if 'group' in regionprops.columns else 'No groups'}")
    
    print("\n📈 Testing normality for each group and variable...")
    try:
        normality_results = stats_analyzer.test_normality(columns)
        
        for col, col_results in normality_results.items():
            print(f"\n  {col}:")
            for group, result in col_results.items():
                status = "✓ Normal" if result.assumptions_met else "✗ Non-normal"
                print(f"    {group}: {result.test_name}, p={result.p_value:.4f} {status} (n={result.sample_size})")
    except Exception as e:
        print(f"  ❌ Normality testing failed: {e}")
    
    print("\n📊 Testing homoscedasticity...")
    try:
        homosced_results = stats_analyzer.test_homoscedasticity(columns)
        for col, result in homosced_results.items():
            status = "✓ Equal variances" if result.assumptions_met else "✗ Unequal variances"
            print(f"  {col}: {result.test_name}, p={result.p_value:.4f} {status} (n={result.sample_size})")
    except Exception as e:
        print(f"  ❌ Homoscedasticity testing failed: {e}")
    
    print("✅ Normality testing completed!")

def test_group_comparisons(regionprops: pd.DataFrame = None, columns: List[str] = None):
    """
    Test group comparison methods
    
    Parameters
    ----------
    regionprops : pd.DataFrame, optional
        DataFrame containing region properties. If None, creates synthetic test data.
    columns : List[str], optional
        Columns to test. If None, uses default numeric columns.
    """
    print("\n" + "="*50)
    print("🧪 TESTING GROUP COMPARISONS")
    print("="*50)
    
    if regionprops is None:
        regionprops = create_test_experiment().regionprops
        print("📊 Using synthetic test data")
    else:
        print("📊 Using provided experiment data")

    stats_analyzer = ExperimentStatistics(regionprops)
    
    if columns is None:
        columns = stats_analyzer.get_numeric_columns()[:3]  # Test first 3 columns
    
    # Check if we have groups
    if 'group' not in regionprops.columns:
        print("  ⚠️ No 'group' column found in data. Cannot perform group comparisons.")
        return

    groups = regionprops['group'].unique()
    print(f"   Testing columns: {columns}")
    print(f"   Groups: {list(groups)} ({len(groups)} groups)")
    
    print("\n🔬 Testing comprehensive group comparisons...")
    try:
        comparison_results = stats_analyzer.compare_groups_comprehensive(columns)
        
        for col, results in comparison_results.items():
            print(f"\n  {col}:")
            if 'comparison' in results:
                comp = results['comparison']
                print(f"    Test: {comp.test_name}")
                print(f"    Statistic: {comp.statistic:.4f}, p-value: {comp.p_value:.4f}")
                if comp.effect_size is not None:
                    print(f"    Effect size: {comp.effect_size:.4f}")
                print(f"    Interpretation: {comp.interpretation}")
                print(f"    Assumptions met: {comp.assumptions_met}")
                
                # Show post-hoc results if available
                if 'posthoc' in results:
                    posthoc = results['posthoc']
                    if 'tukey' in posthoc and isinstance(posthoc['tukey'], pd.DataFrame):
                        significant_pairs = len(posthoc['tukey'][posthoc['tukey']['reject'] == True]) if 'reject' in posthoc['tukey'].columns else 0
                        print(f"    Post-hoc (Tukey): {len(posthoc['tukey'])} comparisons, {significant_pairs} significant")
                    elif 'dunn' in posthoc:
                        print(f"    Post-hoc (Dunn): Available")
                        print(posthoc['dunn'].head())
                    elif 'pairwise_mannwhitney' in posthoc and isinstance(posthoc['pairwise_mannwhitney'], pd.DataFrame):
                        sig_pairs = posthoc['pairwise_mannwhitney']['significant'].sum()
                        total_pairs = len(posthoc['pairwise_mannwhitney'])
                        print(f"    Post-hoc (Mann-Whitney): {sig_pairs}/{total_pairs} significant pairs")
                    elif 'error' in posthoc:
                        print(f"    Post-hoc: {posthoc['error']}")
            else:
                print(f"    No comparison results available")
    except Exception as e:
        print(f"  ❌ Group comparison failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Group comparison testing completed!")

def test_correlation_analysis(regionprops: pd.DataFrame = None, columns: List[str] = None):
    """
    Test correlation analysis
    
    Parameters
    ----------
    regionprops : pd.DataFrame, optional
        DataFrame containing region properties. If None, creates synthetic test data.
    columns : List[str], optional
        Columns to test. If None, uses default numeric columns.
    """
    print("\n" + "="*50)
    print("🧪 TESTING CORRELATION ANALYSIS")
    print("="*50)

    if regionprops is None:
        regionprops = create_test_experiment().regionprops
        print("📊 Using synthetic test data")
    else:
        print("📊 Using provided experiment data")

    stats_analyzer = ExperimentStatistics(regionprops)
    
    if columns is None:
        columns = stats_analyzer.get_numeric_columns()[:4]  # Test first 4 columns for correlation matrix
    
    print(f"   Testing columns: {columns}")
    print(f"   Dataset size: {len(regionprops)} observations")
    
    print("\n🔗 Testing correlation analysis...")
    try:
        corr_results = stats_analyzer.correlation_analysis(columns)
        
        # Overall correlations
        if 'overall' in corr_results:
            overall_corr = corr_results['overall']['correlation_matrix']
            print(f"\n  Overall correlations (n={len(regionprops)}):")
            print(overall_corr)
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(overall_corr.columns)):
                for j in range(i+1, len(overall_corr.columns)):
                    col1, col2 = overall_corr.columns[i], overall_corr.columns[j]
                    corr_val = overall_corr.loc[col1, col2]
                    if not pd.isna(corr_val):  # Skip NaN correlations
                        corr_pairs.append((col1, col2, corr_val))
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Show top correlations
            n_show = min(3, len(corr_pairs))
            for col1, col2, corr_val in corr_pairs[:n_show]:
                strength = "Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.3 else "Weak"
                print(f"    {col1} - {col2}: r={corr_val:.3f} ({strength})")
        
        # Group-wise correlations
        if 'by_group' in corr_results and corr_results['by_group']:
            print(f"\n  Group-wise correlations:")
            for group, group_results in corr_results['by_group'].items():
                group_corr = group_results['correlation_matrix']
                n_samples = group_results['sample_size']
                
                # Find strongest correlation in this group
                max_corr = 0
                max_pair = None
                for i in range(len(group_corr.columns)):
                    for j in range(i+1, len(group_corr.columns)):
                        col1, col2 = group_corr.columns[i], group_corr.columns[j]
                        corr_val = group_corr.loc[col1, col2]
                        if not pd.isna(corr_val) and abs(corr_val) > max_corr:
                            max_corr = abs(corr_val)
                            max_pair = (col1, col2, corr_val)
                
                if max_pair:
                    print(f"    {group} (n={n_samples}): Strongest: {max_pair[0]}-{max_pair[1]} r={max_pair[2]:.3f}")
                else:
                    print(f"    {group} (n={n_samples}): No valid correlations")
        else:
            print("  No group-wise correlations available")
            
    except Exception as e:
        print(f"  ❌ Correlation analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Correlation analysis testing completed!")

def test_comprehensive_report(regionprops: pd.DataFrame = None, columns: List[str] = None):
    """
    Test comprehensive statistical report generation
    
    Parameters
    ----------
    regionprops : pd.DataFrame, optional
        DataFrame containing region properties. If None, creates synthetic test data.
    columns : List[str], optional
        Columns to include in report. If None, uses default numeric columns.
    """
    print("\n" + "="*50)
    print("🧪 TESTING COMPREHENSIVE REPORT")
    print("="*50)

    if regionprops is None:
        regionprops = create_test_experiment().regionprops
        print("📊 Using synthetic test data")
    else:
        print("📊 Using provided experiment data")

    stats_analyzer = ExperimentStatistics(regionprops)
    
    if columns is None:
        columns = stats_analyzer.get_numeric_columns()[:4]  # Limit for readability
    
    print(f"   Analyzing columns: {columns}")
    
    print("\n📋 Generating comprehensive statistical report...")
    try:
        # Generate and print report
        stats_analyzer.print_summary_report(columns)
    except Exception as e:
        print(f"  ❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Comprehensive report testing completed!")

def test_column_detection(regionprops: pd.DataFrame = None):
    """Test column type detection methods"""
    print("\n" + "="*50)
    print("🧪 TESTING COLUMN DETECTION")
    print("="*50)
    
    if regionprops is None:
        from experiment_statistics import create_test_experiment
        regionprops = create_test_experiment()
    
    stats = ExperimentStatistics(regionprops)
    
    # Test all column detection methods
    try:
        numeric_cols = stats.get_numeric_columns()
        print(f"✅ Numeric columns: {numeric_cols}")
    except Exception as e:
        print(f"❌ get_numeric_columns failed: {e}")
    
    try:
        categorical_cols = stats.get_categorical_columns()
        print(f"✅ Categorical columns: {categorical_cols}")
    except Exception as e:
        print(f"❌ get_categorical_columns failed: {e}")
    
    try:
        column_types = stats.get_column_types()
        print(f"✅ Column types summary: {len(column_types)} categories")
    except Exception as e:
        print(f"❌ get_column_types failed: {e}")
    
    try:
        stats.print_column_summary()
        print("✅ Column summary printed successfully")
    except Exception as e:
        print(f"❌ print_column_summary failed: {e}")
    
    print("✅ Column detection testing completed!")
    
def run_all_tests(regionprops: pd.DataFrame = None, columns: List[str] = None):
    """
    Run all statistical analysis tests
    
    Parameters
    ----------
    regionprops : pd.DataFrame, optional
        DataFrame containing region properties. If None, creates synthetic test data.
    columns : List[str], optional
        Columns to test. If None, uses appropriate defaults for each test.
    """
    print("🚀 RUNNING ALL STATISTICAL ANALYSIS TESTS")
    print("="*60)
    
    if regionprops is None:
        print("📊 No regionprops provided - will use synthetic test data")
        regionprops = create_test_experiment().regionprops
    else:
        print(f"📊 Using provided experiment with {len(regionprops)} observations")
    
    try:
        # Run all tests with the same experiment instance
        test_outlier_detection(regionprops, columns)
        test_normality_testing(regionprops, columns)
        test_group_comparisons(regionprops, columns)
        test_correlation_analysis(regionprops, columns)
        test_comprehensive_report(regionprops, columns)

        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
def define_filter(timeframe_threshold_low: int, timeframe_threshold_high: int = None):
    """Defines filter to apply to experiment"""
    from experiment import RegionpropsFilter
    return RegionpropsFilter(column='frame', filter_type='numeric', threshold_low=timeframe_threshold_low, threshold_high=timeframe_threshold_high)

def run_tests_with_real_data(columns=['mean_intensity-1', 'area']):
    """Run tests with real experiment data"""
    print("🔬 RUNNING TESTS WITH REAL EXPERIMENT DATA")
    print("="*60)
    
    try:
        experiment = load_real_test_experiment()
        # Check if column 'frame' is in regionprops to apply filter
        if 'experiment' not in locals() or experiment is None:
            raise ValueError("Failed to load real experiment data.")
        if 'frame' in experiment.regionprops.columns:
            #find maximum frame
            endpoint = experiment.regionprops['frame'].max()
            filter = define_filter(timeframe_threshold_low=endpoint)
            experiment.add_filter(filter)
            experiment.filter_data(apply_to_original=True)
        run_all_tests(experiment.regionprops, columns=columns)
    except Exception as e:
        print(f"❌ Failed to run tests with real data: {e}")
        print("   Falling back to synthetic data...")
        run_all_tests()

def run_tests_with_synthetic_data():
    """Run tests with synthetic data"""
    print("🧪 RUNNING TESTS WITH SYNTHETIC DATA")
    print("="*60)
    
    experiment = create_test_experiment()
    run_all_tests(experiment)

def run_performance_benchmark(regionprops, n_iterations: int = 3):
    """
    Run performance benchmark of statistical functions
    
    Parameters
    ----------
    regionprops : pd.DataFrame, optional
        DataFrame containing region properties. If None, creates synthetic test data.
    n_iterations : int
        Number of iterations for timing
    """
    import time
    
    print("⏱️ RUNNING PERFORMANCE BENCHMARK")
    print("="*50)
    
    if regionprops is None:
        regionprops = create_test_experiment().regionprops

    stats_analyzer = ExperimentStatistics(regionprops)
    columns = stats_analyzer.get_numeric_columns()[:3]
    
    print(f"   Dataset size: {len(regionprops)} observations")
    print(f"   Testing columns: {columns}")
    print(f"   Iterations: {n_iterations}")
    
    benchmarks = {}
    
    # Benchmark outlier detection
    print("\n📊 Benchmarking outlier detection...")
    start_time = time.time()
    for _ in range(n_iterations):
        stats_analyzer.detect_outliers_iqr(columns)
    benchmarks['outlier_detection_iqr'] = (time.time() - start_time) / n_iterations
    
    # Benchmark normality testing
    print("📈 Benchmarking normality testing...")
    start_time = time.time()
    for _ in range(n_iterations):
        stats_analyzer.test_normality(columns)
    benchmarks['normality_testing'] = (time.time() - start_time) / n_iterations
    
    # Benchmark group comparisons
    print("🔬 Benchmarking group comparisons...")
    start_time = time.time()
    for _ in range(n_iterations):
        stats_analyzer.compare_groups_comprehensive(columns)
    benchmarks['group_comparisons'] = (time.time() - start_time) / n_iterations
    
    # Benchmark correlation analysis
    print("🔗 Benchmarking correlation analysis...")
    start_time = time.time()
    for _ in range(n_iterations):
        stats_analyzer.correlation_analysis(columns)
    benchmarks['correlation_analysis'] = (time.time() - start_time) / n_iterations
    
    print("\n⏱️ BENCHMARK RESULTS:")
    print("-" * 40)
    for test_name, avg_time in benchmarks.items():
        print(f"  {test_name}: {avg_time:.3f}s average")
    
    total_time = sum(benchmarks.values())
    print(f"\n  Total analysis time: {total_time:.3f}s")
    print("="*50)

def run_analysis(timelapse: bool = False, timecolumn: str = 'frame'):
    """Run a full analysis session and save results
    Arguments:
    timelapse : bool
        Whether to run analysis over timepoints
    timecolumn : str
        Column name for timepoints
    """
    import os
    print("\n" + "="*50)
    print("🧪 RUNNING FULL ANALYSIS SESSION")
    print("="*50)
    # Load experiment
    experiment = load_real_test_experiment()
    if timelapse and timecolumn not in experiment.regionprops.columns:
        print(f"⚠️ Time column '{timecolumn}' not found in data. Running single timepoint analysis instead.")
        timelapse = False
    elif timelapse and timecolumn in experiment.regionprops.columns:
        print(f"📊 Running timelapse analysis over column '{timecolumn}'")
        endpoint = experiment.regionprops[timecolumn].max()
        timepoints = np.arange(0, endpoint+1)
        for timepoint in timepoints:
            filter = define_filter(timeframe_threshold_low=timepoint, timeframe_threshold_high=timepoint)
            experiment.add_filter(filter)
            regionprops = experiment.filter_data(apply_to_original=False)
            # Analysis session
            stats = ExperimentStatistics(regionprops)
            # Perform analyses
            stats.remove_outliers(method='iqr', columns=['area', 'perimeter', 'mean_intensity-1'], inplace=True)
            stats.compare_groups_comprehensive(['mean_intensity-1'])
            stats.correlation_analysis(['mean_intensity-1', 'eccentricity'])

            # Save with all results cached
            _today = pd.Timestamp.now().strftime("%Y%m%d")
            path = os.path.join(os.getcwd(),"Test_outputs","ExperimentStatistics",f"Stats_results_{_today}")
            if not os.path.exists(path):
                os.makedirs(path)
            stats.save(os.path.join(path,f"Stats_timepoint_{timepoint}.json"), method='json',include_results=True)
            print(f"✅ Saved analysis for timepoint {timepoint} to {path}")
            experiment.clear_filters()
    else:
        print("📊 Running single timepoint analysis")
        regionprops = experiment.regionprops
        stats = ExperimentStatistics(regionprops)
        # Perform analyses
        stats.remove_outliers(method='iqr', columns=['area', 'perimeter', 'mean_intensity-1'], inplace=True)
        stats.compare_groups_comprehensive(['mean_intensity-1'])
        stats.correlation_analysis(['mean_intensity-1', 'eccentricity'])

        # Save with all results cached
        _today = pd.Timestamp.now().strftime("%Y%m%d")
        path = os.path.join(os.getcwd(),"Test_outputs","ExperimentStatistics",f"Stats_results_{_today}")
        if not os.path.exists(path):
            os.makedirs(path)
        stats.save(os.path.join(path,f"Stats_single_timepoint.json"), method='json',include_results=True)
        print(f"✅ Saved analysis to {path}")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments for different test modes
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'real':
            run_tests_with_real_data(columns=['mean_intensity-1'])
        elif mode == 'synthetic':
            run_tests_with_synthetic_data()
        elif mode == 'benchmark':
            run_performance_benchmark()
        elif mode == 'all':
            print("🔄 RUNNING ALL TEST MODES")
            print("="*60)
            run_tests_with_synthetic_data()
            run_tests_with_real_data()
            run_performance_benchmark()
        elif mode == 'timelapse':
            run_analysis(timelapse=True)
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: real, synthetic, benchmark, all")
    else:
        # Default: run all tests
        run_all_tests()