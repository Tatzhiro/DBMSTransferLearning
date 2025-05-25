import pandas as pd
import statsmodels.api as sm
from .system_configuration import SystemConfiguration
from .utils import set_unimportant_columns_to_one_value, drop_unimportant_parameters
from abc import ABC, abstractmethod
from sklearn.linear_model import LassoCV, ElasticNetCV, LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from IPython import embed
import numpy as np
from copy import deepcopy
import itertools

class FeatureSelector(ABC):
    def __init__(self, system=None):
        self.system = system
        
    @abstractmethod
    def select_important_features(self, df: pd.DataFrame, system: SystemConfiguration) -> list:
        pass
    
    @abstractmethod
    def get_parameter_importance(self):
        pass


class UtilityFeatureSelector(FeatureSelector):
    def select_important_features(self, df: pd.DataFrame, system: SystemConfiguration) -> list:
        return ["clients", "checkpoint_interval"]
    
    def get_parameter_importance(self):
        raise NotImplementedError("UtilityFeatureSelector does not have parameter importance")


class LassoFeatureSelector(FeatureSelector):
    def select_important_features(self, df: pd.DataFrame, system: SystemConfiguration) -> list:
        self.parameters = system.get_param_names()
        self.perf_metric = system.get_perf_metric()
        
        model = make_pipeline(StandardScaler(), LassoCV())
        X = df[self.parameters]
        y = df[self.perf_metric]
        model.fit(X, y)
        self.coef = [(self.parameters[i], coef) for i, coef in enumerate(model["lassocv"].coef_) if coef != 0]
        return [param for param, _ in self.coef]
    
    def get_parameter_importance(self):
        return self.coef
    
    def select_parameter_to_sample(self):
        param = self.coef
        param_magnitude = np.abs([coef for _, coef in param])
        param_frq = param_magnitude / np.sum(param_magnitude)
        distribution = param_frq
        return np.random.choice([param for param, _ in param], p=distribution)
    
    def get_parameter_vector(self):
        vector = np.zeros(len(self.parameters))
        important_params = [p[0] for p in self.coef]
        coef = [p[1] for p in self.coef]
        param_magnitude = np.abs(coef)
        param_frq = param_magnitude / np.sum(param_magnitude)
        dic = {p: coef for p, coef in zip(important_params, param_frq)}
        for i, p in enumerate(self.parameters):
            if p in important_params:
                vector[i] = dic[p]
            else:
                vector[i] = 0
        return vector


class ElasticNetFeatureSelector(FeatureSelector):
    def select_important_features(self, df: pd.DataFrame, system: SystemConfiguration) -> list:
        model = make_pipeline(StandardScaler(), ElasticNetCV())
        X = df[system.get_param_names()]
        y = df[system.get_perf_metric()]
        model.fit(X, y)
        self.coef = [(system.get_param_names()[i], coef) for i, coef in enumerate(model["elasticnetcv"].coef_) if coef != 0]
        return [param for param, _ in self.coef]

    def get_parameter_importance(self):
        return self.coef
        
        
class HandPickedFeatureSelector(FeatureSelector):
    def select_important_features(self, df: pd.DataFrame, system: SystemConfiguration) -> list:
        return self.features
    
    def get_parameter_importance(self):
        raise NotImplementedError("HandPickedFeatureSelector does not have parameter importance")
    
    def __init__(self, features) -> None:
        self.features = features
    
    
class UserSelectLassoFeatureSelector(LassoFeatureSelector):
    def select_important_features(self, df: pd.DataFrame, system: SystemConfiguration) -> list:
        important_features = super().select_important_features(df, system)
        selected_features = [x for x in important_features if x not in self.parameters_to_remove]
        selected_features = list(set(selected_features + self.parameters_to_add))
        return selected_features
    
    def get_parameter_importance(self):
        return super().get_parameter_importance()
    
    def __init__(self, parameters_to_remove: list[str] = [], parameters_to_add: list[str] = []) -> None:
        self.parameters_to_remove = parameters_to_remove
        self.parameters_to_add = parameters_to_add


class L2SFeatureSelector(FeatureSelector):
    def select_important_features(self, df: pd.DataFrame, system: SystemConfiguration) -> list:
        self.parameters = system.get_param_names()
        self.perf_metric = system.get_perf_metric()

        important_clm_idxs = self.__stepwise_feature_selection(df, system)
    
        # concatenate parameters that are chosen by stepwise regression
        self.important_params = [self.parameters[i] for i in important_clm_idxs]
        
        X, y = self.__filter_data(df, important_clm_idxs, system)
        matrix = np.zeros((len(self.parameters), len(self.parameters)))
        model = Pipeline([("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)),
                                ("linear", LinearRegression(fit_intercept=True))])
        self.pmodel = model.fit(X, y)
        self.coefficients = pd.Series(model["linear"].coef_, index=model["poly"].get_feature_names_out())

        for name, coef in self.coefficients.items():
            # Ignore bias (intercept)
            if name == "1":
                continue

            terms = name.split(" ")
            
            if len(terms) == 1:
                term = terms[0]
                index = int(term.removeprefix("x"))
                param = self.important_params[index]
                # Linear term: put on diagonal
                i = self.parameters.index(param)
                matrix[i, i] = abs(coef)
            elif len(terms) == 2:
                index_i = int(terms[0].removeprefix("x"))
                index_j = int(terms[1].removeprefix("x"))
                # Interaction term: fill symmetric matrix
                param_i = self.important_params[index_i]
                param_j = self.important_params[index_j]
                i = self.parameters.index(param_i)
                j = self.parameters.index(param_j)
                matrix[i, j] = abs(coef)
                
        self.matrix = matrix

        return self.important_params


    def __stepwise_feature_selection(self, df: pd.DataFrame, system: SystemConfiguration,
                                initial_list=[],
                                threshold_in=0.01,
                                threshold_out=0.05,
                                verbose=True) -> list:
        ndim = len(self.parameters)
        features = [i for i in range(ndim)]
        included = list(initial_list)

        while True:
            changed = False
            # forward
            excluded = list(set(features) - set(included))
            new_pval = pd.Series(index=excluded)
            for new_feature in excluded:
                X, y = self.__filter_data(df, included + [new_feature], system)
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X))).fit()
                new_pval[new_feature] = model.pvalues.tail(1).iloc[0]
            best_pval = new_pval.min()

            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print('Add {:30} with p-value {:.6}'.format(self.parameters[best_feature], best_pval))

            # backward
            X, y = self.__filter_data(df, included, system)
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X))).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()

            if worst_pval > threshold_out:
                changed = True
                worst_feature = pvalues.idxmax()
                drop_param = included[worst_feature]
                included.remove(drop_param)
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(self.parameters[drop_param], worst_pval))

            if not changed:
                break
        
        X, y = self.__filter_data(df, included, system)
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X))).fit()
        self.coefficients = model.params
        return included
    
    def __filter_data(self, df: pd.DataFrame, feature_idxs: list, system: SystemConfiguration) -> pd.DataFrame:
        filter_df = deepcopy(df)
        features = [self.parameters[i] for i in feature_idxs]

        filter_df = drop_unimportant_parameters(filter_df, features, system)
        if len(filter_df) == 0: raise RuntimeError("filtered df has no rows, check if the default parameter values are set correctly")

        X = filter_df[features].to_numpy()
        y = filter_df[self.perf_metric].to_numpy()
        
        for df_clm, idx in zip(filter_df[features].columns, feature_idxs):
            assert df_clm == self.parameters[idx]

        return X, y
    
    def select_parameter_to_sample(self, interaction: bool = True):
        if interaction:
            return self.select_parameter_to_sample_2d()
        else:
            return self.select_parameter_to_sample_1d()
    
    def select_parameter_to_sample_1d(self):
        param = self.coefficients.drop("const")
        param_magnitude = np.abs(param)
        param_frq = param_magnitude / np.sum(param_magnitude)
        distribution = param_frq.to_list()
        return np.random.choice(self.important_params, p=distribution)
    
    def select_parameter_to_sample_2d(self):
        param_list = self.parameters
        param = self.matrix
        param_magnitude = np.abs(param)
        param_frq = param_magnitude / np.sum(param_magnitude)
        distribution = param_frq.flatten()
        idx = np.random.choice(len(distribution), p=distribution)
        i = idx // len(param_list)
        j = idx % len(param_list)
        return param_list[i], param_list[j]
    
    def get_parameter_importance(self):
        param = self.coefficients
        if "const" in self.coefficients.index:
            param = self.coefficients.drop("const")
        param_magnitude = np.abs(param)
        param_frq = param_magnitude / np.sum(param_magnitude)
        importance = [(self.important_params[i], coef) for i, coef in enumerate(param_frq)]
        return importance
    
    def get_parameter_vector(self):
        vector = np.zeros(len(self.parameters))
        param = self.coefficients
        if "const" in self.coefficients.index:
            param = self.coefficients.drop("const")
        param_magnitude = np.abs(param)
        param_frq = param_magnitude / np.sum(param_magnitude)
        for i, p in enumerate(self.parameters):
            if p in self.important_params:
                vector[i] = param_frq[self.important_params.index(p)]
            else:
                vector[i] = 0
        return vector
    

class ImportanceFeatureSelector(FeatureSelector):
    def __init__(self, df: pd.DataFrame=None, system: SystemConfiguration=None) -> None:
        self.df = df
        self.system = system
        
    
    def select_important_features(self, df: pd.DataFrame, system: SystemConfiguration) -> None:
        self.df = df
        self.system = system
        self.vector = self.get_parameter_vector()
        threshold = 1 / len(self.system.get_param_names())
        self.important_params = [p for p, v in zip(self.system.get_param_names(), self.vector) if v > threshold]
        return self.important_params
    
    def get_parameter_importance(self):
        raise NotImplementedError("FeatureImportance does not have parameter importance")
    
    
    def select_parameter_to_sample(self):
        importance_vector = deepcopy(self.vector)
        for i, p in enumerate(self.system.get_param_names()):
            if p not in self.important_params:
                importance_vector[i] = 0
        # renormalize
        importance_vector = importance_vector / np.sum(importance_vector)
        return np.random.choice(self.system.get_param_names(), p=importance_vector)

        
    def get_significant_range(self, df: pd.DataFrame):
        min = df[self.system.get_perf_metric()].min()
        max = df[self.system.get_perf_metric()].max()
        return max - min
        
    
    def get_parameter_vector(self):
        vector = np.zeros(len(self.system.get_param_names()))
        
        self.df = self.system.preprocess_param_values(self.df)
        
        for i, p in enumerate(self.system.get_param_names()):
            df = drop_unimportant_parameters(self.df, [p], self.system)
            range = self.get_significant_range(df)
            vector[i] = range
        
        vector = vector / np.sum(vector)
        return vector
    
    def get_parameter_matrix(self):
        param_names = self.system.get_param_names()
        matrix = np.zeros((len(param_names), len(param_names)))
        
        # Preprocess the dataframe once
        self.df = self.system.preprocess_param_values(self.df)
        
        for i, p in enumerate(param_names):
            for j, q in enumerate(param_names):
                if i > j:
                    continue
                df = drop_unimportant_parameters(self.df, [p, q], self.system)
                range = self.get_significant_range(df)
                matrix[i][j] = range

        # Normalize the vector so that the effects sum to 1
        matrix = matrix / np.sum(matrix)
        return matrix
        
    
    def select_parameter_pair_to_sample(self):
        if not hasattr(self, "matrix"):
            self.matrix = self.get_parameter_matrix()
        matrix = self.matrix
        flat_matrix = matrix.flatten()
        flat_matrix = flat_matrix / np.sum(flat_matrix)
        idx = np.random.choice(len(flat_matrix), p=flat_matrix)
        i = idx // len(matrix)
        j = idx % len(matrix)
        return self.system.get_param_names()[i], self.system.get_param_names()[j]
    
    
class MultiImportanceFeatureSelector(ImportanceFeatureSelector):
    def __init__(self, df: pd.DataFrame=None, system: SystemConfiguration=None) -> None:
        self.df = df
        self.system = system
        
    def get_parameter_matrix(self):
        param_names = self.system.get_param_names()
        matrix = np.zeros((len(param_names), len(param_names)))
        
        # Preprocess the dataframe once
        self.df = self.system.preprocess_param_values(self.df)
        full_matrix = np.zeros((len(param_names), len(param_names)))
        
        for i, p in enumerate(param_names):
            for j, q in enumerate(param_names):
                if i > j:
                    continue
                df = drop_unimportant_parameters(self.df, [p, q], self.system)
                min = df[self.system.get_perf_metric()].quantile(0.01)
                max = df[self.system.get_perf_metric()].quantile(0.99)
                full_matrix[i][j] = max - min
                range = self.get_significant_range(df)
                if p != q and (not self.q_is_important(df, p, q) or not self.q_is_important(df, q, p)):
                    range = 0
                matrix[i][j] = range

        # Normalize the vector so that the effects sum to 1
        print("full matrix", full_matrix)
        matrix = matrix / np.sum(matrix)
        return matrix


    def get_significant_range(self, df: pd.DataFrame, threshold_factor: float = 1.05):
        min = df[self.system.get_perf_metric()].quantile(0.01)
        max = df[self.system.get_perf_metric()].quantile(0.99)
        if max / min < threshold_factor:
            return 0
        return max - min
    
    
    def q_is_important(self, df, p, q):
        # Check if q is important given p
        # get unique values of p
        unique_p = df[p].unique()
        # for each unique value of p, get the range of q
        for val in unique_p:
            df_p = df[df[p] == val]
            # 99th percentile of q
            min = df_p[self.system.get_perf_metric()].quantile(0.01)
            max = df_p[self.system.get_perf_metric()].quantile(0.99)
            if max > 2 * min:
                return True
        return False
        
    
    def select_important_features(self) -> None:
        if not hasattr(self, "matrix"):
            self.matrix = self.get_parameter_matrix()
        
        important_params = []
        for i, p in enumerate(self.system.get_param_names()):
            # if all the elements in ith row are 0, then skip
            if np.sum(self.matrix[i]) == 0:
                continue
            if np.sum(self.matrix[:, i]) == 0:
                continue
            
            important_params.append(p)
        return important_params
    
    