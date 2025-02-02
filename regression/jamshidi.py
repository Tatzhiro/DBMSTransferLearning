import pandas as pd
import statsmodels.api as sm
from .system_configuration import SystemConfiguration
from .utils import set_unimportant_columns_to_one_value, drop_unimportant_parameters
from abc import ABC, abstractmethod
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from IPython import embed
import numpy as np
from copy import deepcopy

class FeatureSelector(ABC):
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
        model = make_pipeline(StandardScaler(), LassoCV())
        X = df[system.get_param_names()]
        y = df[system.get_perf_metric()]
        model.fit(X, y)
        self.coef = [(system.get_param_names()[i], coef) for i, coef in enumerate(model["lassocv"].coef_) if coef != 0]
        return [param for param, _ in self.coef]
    
    def get_parameter_importance(self):
        return self.coef


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
                included.remove(included[worst_feature])
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(self.parameters[worst_feature], worst_pval))

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
    
    def select_parameter_to_sample(self):
        param = self.coefficients.drop("const")
        param_magnitude = np.abs(param)
        param_frq = param_magnitude / np.sum(param_magnitude)
        distribution = param_frq.to_list()
        return np.random.choice(self.important_params, p=distribution)
    
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