import pandas as pd
from .system_configuration import SystemConfiguration
import numpy as np
from copy import deepcopy
import warnings
from IPython import embed
import re


def set_unimportant_columns_to_one_value(df: pd.DataFrame, important_params: list, system: SystemConfiguration) -> pd.DataFrame:
    default = system.get_default_param_values()

    for param in system.get_param_names():
        if param not in important_params:
            if param in default.keys() and param in df.columns:
                if len(df[df[param] == default[param]]) > 0:
                    df = df[df[param] == default[param]]
                else:
                    warnings.warn(f"Default parameter value for {param} is not in the data frame", UserWarning)
            else:
                warnings.warn(f"Default parameter value for {param} does not exist", UserWarning)
    return df


def drop_unimportant_parameters(df: pd.DataFrame, important_params: list, system: SystemConfiguration) -> pd.DataFrame:
    default = system.get_default_param_values()

    for param in system.get_param_names():
        if param not in important_params:
            if param in default.keys() and param in df.columns:
                if len(df[df[param] == default[param]]) > 0:
                    df = df[df[param] == default[param]]
                    df = df.drop(columns=param)
                else:
                    raise UserWarning(f"Default parameter value for {param} is not in the data frame")
            else:
                raise UserWarning(f"Default parameter value for {param} does not exist")
    return df


def group_features(df, important_features, system: SystemConfiguration):
    conf_df = deepcopy(df)
    conf_df = set_unimportant_columns_to_one_value(conf_df, important_features, system)
    conf_df["config"] = conf_df[important_features].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    series = conf_df.groupby("config")[system.get_perf_metric()].mean()
    return pd.DataFrame({"config": series.keys(), system.get_perf_metric(): series.values})


def read_data_csv(filename: str, system: SystemConfiguration, workload: str):
    df = pd.read_csv(filename)
    df = system.preprocess_param_values(df)
    if workload == "None":
        df = df[system.get_param_names() + [system.get_perf_metric()]]
    else:
        df = df[df.get("workload", df["workload_label"]) == workload][system.get_param_names() + [system.get_perf_metric()]]
    return df


def filter_default_config(df: pd.DataFrame, system: SystemConfiguration) -> pd.DataFrame:
    """Filter DataFrame to rows where all DBMS parameters match default configuration values."""
    defaults = system.get_default_param_values()
    param_cols = [p for p in defaults if p in df.columns]
    if not param_cols:
        return df
    processed = system.preprocess_param_values(df.copy())
    mask = pd.Series(True, index=df.index)
    for param in param_cols:
        mask &= (processed[param] == defaults[param])
    return df[mask]


def epsilon_greedy(epsilon) -> bool:
    return np.random.uniform() < epsilon
