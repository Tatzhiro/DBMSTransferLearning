from pandas import DataFrame
import pandas as pd
from abc import ABC, abstractmethod
from IPython import embed
import re


class SystemConfiguration(ABC):
    def get_param_names(self) -> list:
        return self.param_names

    def get_machine_specs(self) -> list:
        return self.machine_specs

    def get_perf_metric(self) -> str:
        return self.perf_metric

    def get_features(self) -> list:
        return self.param_names + self.machine_specs

    def get_default_param_values(self) -> dict:
        return self.default_param_values

    def set_machine_specs(self, df: DataFrame, filename: str) -> DataFrame:
        if "num_core" in self.machine_specs:
            num_core = getNumCore(filename)
            df["num_core"] = num_core
        if "mem_size" in self.machine_specs:
            mem_size = getMemSize(filename)
            df["mem_size"] = mem_size
        return df

    @abstractmethod
    def get_machine_independent_parameters(self) -> list:
        pass

    @abstractmethod
    def get_machine_dependent_parameters(self) -> list:
        pass

    @abstractmethod
    def normalize_columns(self, df: DataFrame, filename: str) -> DataFrame:
        pass

    @abstractmethod
    def preprocess_param_values(self, df: DataFrame) -> DataFrame:
        pass

    def __init__(self, perf_metric, param_names, machine_specs, machine_dependent_params, default_param_values) -> None:
        self.perf_metric: str = perf_metric
        self.param_names: list = param_names
        self.machine_specs: list = machine_specs
        self.machine_dependent_params: list = machine_dependent_params
        self.default_param_values: dict = default_param_values


class LineairDBConfiguration(SystemConfiguration):
    def normalize_columns(self, df: DataFrame, filename: str) -> DataFrame:
        df["clients"] = df["clients"] / getNumCore(filename)
        return df

    def get_machine_independent_parameters(self) -> list:
        return [param for param in self.param_names if param not in self.machine_dependent_params]

    def get_machine_dependent_parameters(self) -> list:
        return self.machine_dependent_params

    def preprocess_param_values(self, df: DataFrame) -> DataFrame:
        return df

    def __init__(self) -> None:
        perf_metric = "tps"
        param_names = ["clients", "epoch_duration", "checkpoint_interval", "rehash_threshold", "prefetch_locality"]
        default_param_values = {
            "clients": 1,
            "epoch_duration": 40,
            "checkpoint_interval": 30,
            "rehash_threshold": 0.8,
            "prefetch_locality": 3
        }
        machine_dependent_params = ["clients"]
        machine_specs = ["num_core"]
        super().__init__(perf_metric, param_names, machine_specs, machine_dependent_params, default_param_values)


class MySQLConfiguration(SystemConfiguration):
    def normalize_columns(self, df: DataFrame, filename: str) -> DataFrame:
        df["innodb_buffer_pool_size"] = df["innodb_buffer_pool_size"] / getMemSize(filename)
        df["innodb_read_io_threads"] = df["innodb_read_io_threads"] / getNumCore(filename)
        df["innodb_write_io_threads"] = df["innodb_write_io_threads"] / getNumCore(filename)
        return df

    def get_machine_independent_parameters(self) -> list:
        return [param for param in self.param_names if param not in self.machine_dependent_params]

    def get_machine_dependent_parameters(self) -> list:
        return self.machine_dependent_params

    def preprocess_param_values(self, df: DataFrame) -> DataFrame:
        if df[self.param_names].apply(pd.to_numeric, errors='coerce').notnull().all().all():
            return df

        df["innodb_buffer_pool_size"] = pd.to_numeric(df["innodb_buffer_pool_size"].str.removesuffix("GB"), errors='coerce')
        df["innodb_adaptive_hash_index"] = pd.to_numeric(df["innodb_adaptive_hash_index"] == "ON").astype(int)

        numeric_file_size = []
        for val in df["innodb_log_file_size"].to_list():
            if "MB" in val:
                num_val = int(val.removesuffix("MB")) / 1000
                numeric_file_size.append(num_val)
            elif "GB" in val:
                num_val = int(val.removesuffix("GB"))
                numeric_file_size.append(num_val)
        df["innodb_log_file_size"] = numeric_file_size

        return df

    def __init__(self) -> None:
        perf_metric = "tps"
        param_names = [
            "innodb_buffer_pool_size", "innodb_read_io_threads", "innodb_write_io_threads",
            "innodb_flush_log_at_trx_commit", "innodb_adaptive_hash_index", "sync_binlog",
            "innodb_lru_scan_depth", "innodb_buffer_pool_instances", "innodb_change_buffer_max_size",
            "innodb_io_capacity", "innodb_log_file_size", "table_open_cache"
        ]
        default_param_values = {
            "innodb_buffer_pool_size": 1, "innodb_read_io_threads": 2, "innodb_write_io_threads": 2,
            "innodb_flush_log_at_trx_commit": 1, "innodb_adaptive_hash_index": True, "sync_binlog": 1,
            "innodb_lru_scan_depth": 1024, "innodb_buffer_pool_instances": 1,
            "innodb_change_buffer_max_size": 25, "innodb_io_capacity": 100,
            "innodb_log_file_size": 0.048, "table_open_cache": 4000
        }
        machine_dependent_params = [
            "innodb_buffer_pool_size", "innodb_read_io_threads", "innodb_write_io_threads"
        ]
        machine_specs = ["num_core", "mem_size"]
        super().__init__(perf_metric, param_names, machine_specs, machine_dependent_params, default_param_values)


def getNumCore(spec_string):
  return int(re.search(r'\d+', spec_string).group())

def getMemSize(spec_string):
  return int(re.search(r'\d+c(\d+)g', spec_string).group(1))
