import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

from .graph import getNumCore
from .jamshidi import L2SFeatureSelector
from .system_configuration import SystemConfiguration, LineairDBConfiguration
from .model import ValovModel

class MultiDimGraph(ABC):
  def plot_gauss(self, plot_save_path, prediction_file: str, true_file: str, parameters: list = []):
    if len(parameters) == 0: parameters = self.system.get_param_names()
    df = pd.read_csv(prediction_file)
    df = self.df_normalizer.drop_useless_columns(df, ["low", "high"])
    df = self.system.normalize_columns(df, prediction_file)

    conf_interval = {"tps": None, "low": None, "high": None}
    for perf_metric in ["low", "high", "tps"]:
      # not recommended to change system.perf_metric
      # make sure to set to the original perf_metric
      self.df_normalizer.system.perf_metric = perf_metric
      series = self.df_normalizer.extract_columns(df, parameters)
      conf_interval[perf_metric] = series

    x_values = conf_interval['tps'].index
    mean_prediction = conf_interval['tps'].values
    low_prediction = conf_interval['low'].values
    high_prediction = conf_interval['high'].values
    plt.plot(x_values, mean_prediction, marker='x', color="green", linestyle="dashed", linewidth=1, markersize=2, label=prediction_file.replace("result/", ""))
    plt.fill_between(
        x_values.ravel(),
        low_prediction,
        high_prediction,
        color="tab:orange",
        alpha=0.5,
        label=r"95% confidence interval",
    )
    
    self.plot_performance_model(plot_save_path, parameters, [true_file])

  def plot_performance_model(self, plot_save_path, parameters: list = [], file_names: list = []):
    if len(parameters) == 0: parameters = self.system.get_param_names()
    if len(file_names) == 0: file_names = self.file_names

    
    line_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    line_styles = ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed']

    for filename, color, style in zip(file_names, line_colors, line_styles):
      df = self.df_normalizer.make_data(filename, parameters)
      x_values = df.index
      y_values = df.values
      plt.plot(x_values, y_values, marker='x', color=color, linestyle=style, linewidth=1, markersize=2, label=filename.replace("result/", ""))

    plt.legend(fontsize=16)
    plt.xticks([])
    self.line_graph_labels(x_label='Combined Features', y_label=self.system.perf_metric, title='Line Graph for Multiple Features vs. Target Variable', fig_save_path=plot_save_path)
    plt.close()

  def plot_regression(self, file_name1, file_name2, plot_save_path, want_all_params: bool = True):
    self.df_normalizer.reduce_features = not want_all_params
    source_df, target_df, important_params = self.df_normalizer.make_domain_target_data(file_name1, file_name2, True)
    merge_df = pd.merge(source_df, target_df, on='config', how='outer')

    model = LinearRegression()
    x_perf = merge_df.loc[:, [f"{self.system.get_perf_metric()}_x"]].values
    y_perf = merge_df.loc[:, f"{self.system.get_perf_metric()}_y"].values
    model.fit(x_perf, y_perf)

    plt.figure()
    plt.scatter(x_perf, y_perf)
    plt.plot(x_perf, model.predict(x_perf))
    plt.xlabel(f"{file_name1.replace('result/', '')} Throughput", fontsize=12)
    plt.ylabel(f"{file_name2.replace('result/', '')} Throughput", fontsize=12)
    r2 = self.calculateR2(model, merge_df)
    if want_all_params: annotation = f"r-squared = {format(r2, '.3f')}\nn = {len(merge_df)}"
    else: annotation = f"r-squared = {format(r2, '.3f')}\nn = {len(merge_df)}\nc = {important_params}"
    plt.annotate(annotation, xy=(0.05, 0.85), xycoords='axes fraction', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    plt.savefig(plot_save_path, bbox_inches='tight')
    plt.close()
    return model

  def line_graph_labels(self, x_label, y_label, title, fig_save_path):
    fontsize = 18
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')

  def calculateR2(self, model, df):
    x_perf = df.loc[:, [f"{self.system.get_perf_metric()}_x"]].values
    y_perf = df.loc[:, f"{self.system.get_perf_metric()}_y"].values
    y_pred = model.predict(x_perf)
    return r2_score(y_perf, y_pred)

  def __init__(self, file_names: list, workload: str, system: SystemConfiguration) -> None:
    self.file_names: list = file_names
    self.workload: str = workload
    self.system: SystemConfiguration = system
    self.df_normalizer: ValovModel = ValovModel(workload, system)


class LineairDBGraph(MultiDimGraph):
  def plot_clients_vs_core(self):
    client_df = pd.DataFrame(columns=["tps"])
    for filename in self.file_names:
      df = self.df_normalizer.preprocess(filename)
      df = self.df_normalizer.l2s.set_unimportant_columns_to_one_value(df, ["clients"])
      num_core = getNumCore(filename)
      client_df.loc[num_core] = [df[df["clients"] == 1]["tps"].mean()]
    x_values = client_df.index
    y_values = client_df.values
    plt.plot(x_values, y_values, marker='x',linewidth=1, markersize=2, linestyle="solid")
    plt.xticks(np.arange(min(x_values), max(x_values)+1, 2.0))
    self.line_graph_labels(x_label='num_core', y_label='Throughput', title='#core vs throughput (clients = nproc)', fig_save_path="clients_vs_core.pdf")
    plt.close()

  def plot_core_vs_throughput(self):
    client_df = pd.DataFrame(columns=["tps"])
    for filename in self.file_names:
      df = self.df_normalizer.drop_useless_columns(pd.read_csv(filename))
      df = self.df_normalizer.l2s.set_unimportant_columns_to_one_value(df, ["clients"])
      num_core = getNumCore(filename)
      if df[df["clients"] == 4]["clients"].count():
        client_df.loc[num_core] = [df[df["clients"] == 4]["tps"].mean()]
    x_values = client_df.index
    y_values = client_df.values
    plt.plot(x_values, y_values, marker='x',linewidth=1, markersize=2, linestyle="solid")
    plt.xticks(np.arange(min(x_values), max(x_values)+1, 2.0))
    self.line_graph_labels(x_label='num_core', y_label='Throughput', title='#core vs throughput (clients = 4)', fig_save_path="core_vs_tps.pdf")
    plt.close()
    
  def __init__(self, workload="a"):
    file_names = [
                      "result/4c6g-result.csv",
                      "result/8c12g-result.csv",
                      "result/12c16g-result.csv",
                      "result/16c24g-result.csv",
                      "result/24c32g-result.csv",
                     ]
    system = LineairDBConfiguration()
    super().__init__(file_names, workload, system)
    
