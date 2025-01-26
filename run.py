from regression.multid_graph import LineairDBGraph
from regression.model import MLModel, SMLinearModel, ValovModel
from regression.system_configuration import LineairDBConfiguration
from regression.feature_engineer import FeatureEngineer, LogCoreDistanceEngineer, NoCoreEngineer
from regression.pipeline import MultivariateRegressionPipeline, MultivariateDataLoader, SimpleRegressionDataLoader, SimpleRegressionPipeline
from regression.jamshidi import LassoFeatureSelector
from regression.proposed import Proposed
from regression.utils import set_unimportant_columns_to_one_value, drop_unimportant_parameters, read_data_csv, group_features
from regression.distribution_distance import BhattacharyyaDistance
from regression.plot import PlotDesign
import regression.plot as plot
import os
from IPython import embed
from scipy.stats import kstest
# from dictances import bhattacharyya
import pandas as pd
import numpy as np
import pylab
import re
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from regression.graph import getNumCore, getMemSize
import matplotlib.pyplot as plt
from regression.plot import plot_bargraph, PlotDesign
from sklearn.gaussian_process.kernels import WhiteKernel,Exponentiation, DotProduct, RationalQuadratic, RBF, ConstantKernel, Matern, DotProduct

def main():

  # legends()
  plot_inv()
  # performance_function()
  # simulation()
  # distribution_distance()
  # split_file_by_datasize()
  # merge()
  # gaussian_process()
  # plot_all_data()
  # feature_engineering()
  # cross_validation()
  # regression_analysis(LinearRegression())

  # graph = LineairDBGraph("a")
  # graph.plot_clients_vs_core()
  # graph.plot_core_vs_throughput()
  # plot_single_param()
  # plot_all()
  # graph.plot_regression(graph.file_names[0], graph.file_names[4], "fig.pdf", False)
  
# def legends():
#   x = np.linspace(1, 100, 1000)
#   y1 = np.log(x)
#   y2 = np.sin(x)
#   y3 = np.sin(x)
#   y4 = np.sin(x)
#   y5 = np.sin(x)
#   fig = plt.figure("Line plot")
#   legendFig = plt.figure("Legend plot")
#   ax = fig.add_subplot(111)
#   line1, = ax.plot(x, y1, c="red", lw=4, linestyle="dashdot")
#   line2, = ax.plot(x, y1, c="green", lw=1, linestyle="--")
#   legendFig.legend([line1, line2], ["y=log(x)", "y=sin(x)"], loc='center')
#   legendFig.savefig('legend.png')
  
#   figlegend.legend(lines, ('Proposed Method', 'ModelShift', 'L2S', 'DataReuse', 'L2S+DataReuse'), 'center')
  
def plot_inv():
  df = pd.read_csv("outputs/transfer_learning/vanilla/vanilla.csv", index_col=0)
  # df = pd.read_csv("outputs/transfer_learning/below_threshold/below_75_all.csv", index_col=0)

  # df.drop(columns=["Baseline"], inplace=True)
  plot_bargraph(df, PlotDesign("a", "b"), "fig.pdf")
  
  

def performance_function():
  # data = {
  #   "4c6g": "result/4c6g-result.csv",
  #   "8c12g": "result/8c12g-result.csv",
  #   "16c24g": "result/16c24g-result.csv"
  # }
  # parameter = "checkpoint_interval"
  # plot_df = pd.DataFrame()
  # for name in data:
  #   df = pd.read_csv(data[name])
  #   df = group_features(df, [parameter], LineairDBConfiguration())
  #   datum = {df["config"][i] : df["tps"][i] for i in range(len(df))}
  #   plot_df[name] = datum
  # plot_linegraph_from_df(plot_df)
  
  data = {
    "4c6g": "old/20230809_ycsb_a/lineairdb4v.csv",
    "8c12g": "old/20230809_ycsb_a/lineairdb8v.csv",
    "16c24g": "old/20230809_ycsb_a/lineairdb16v.csv",
  }
  parameter = "clients"
  plot_df = pd.DataFrame()
  for name in data:
    df = pd.read_csv(data[name])
    datum = {df["clients"][i] : df["tps"][i] for i in range(len(df))}
    plot_df[name] = datum
  plot_linegraph_from_df(plot_df)
    
  

def simulation():
  pipeline = Proposed("a", LineairDBConfiguration())
  pipeline.initialize("result/8c12g-result.csv", "result/4c6g-result.csv", "result/24c32g-result.csv")
  pipeline.simulate(10)
  df = pipeline.read_data_csv("result/24c32g-result.csv")
  tps = pipeline.predict(df)
  embed()
  


def distribution_distance():
  train = [
            "result/4c6g-result.csv",
            "result/8c12g-result.csv",
            "result/12c16g-result.csv",
            "result/16c24g-result.csv",
            "result/24c32g-result.csv"
          ]
  loader = MultivariateDataLoader("a", LineairDBConfiguration(), LassoFeatureSelector(), False)
  parameters = ["clients", "epoch_duration", "checkpoint_interval", "rehash_threshold", "prefetch_locality"]
  x1 = loader.load_training_df([train[0]])
  x2 = loader.load_training_df([train[4]])
  print(BhattacharyyaDistance().compute_distance(x1, x2, parameters, LineairDBConfiguration()))
    # x1 = drop_unimportant_parameters(x1, [parameter], LineairDBConfiguration())
    # x2 = drop_unimportant_parameters(x2, [parameter], LineairDBConfiguration())
    # x1v = x1.groupby(parameter)["tps"].mean()
    # x2v = x2.groupby(parameter)["tps"].mean()
    # df = pd.merge(x1v, x2v, on=parameter, how="outer").fillna(0)
    # x = df["tps_x"].values / df["tps_x"].values.sum()
    # y = df["tps_y"].values / df["tps_y"].values.sum()
    # print(kstest(x, y))
    # x_dic = {df.index[i] : x[i] for i in range(len(df))}
    # y_dic = {df.index[i] : y[i] for i in range(len(df))}
    # print(bhattacharyya(x_dic, y_dic))


def split_file_by_datasize():
  mac = ["4c6g", "8c12g", "12c16g", "16c24g", "24c32g"]
  for m in mac:
    df_old = pd.read_csv(f"result/mysql/{m}-result.csv")
    data_sizes = df_old["table_size"].drop_duplicates()
    for data_size in data_sizes:
      df = df_old[df_old["table_size"] == data_size]
      data_size_in_m = data_size / 1000000
      df.to_csv(f"result/mysql/{m}-{data_size_in_m}M.csv", index=False)

def merge():
  mac = ["4c6g", "8c12g", "12c16g", "16c24g", "24c32g"]
  for m in mac:
    df_old = pd.read_csv(f"result/{m}-result.csv")
    df_new = pd.read_csv(f"docker/lineairdb/env/{m}-result.csv")
    pd.concat([df_old, df_new]).to_csv(f"result/{m}-result.csv", index=False)

def gaussian_process():
  workload = "a"
  system = LineairDBConfiguration()
  train = [
            "result/4c6g-result.csv",
            "result/8c12g-result.csv",
            "result/12c16g-result.csv",
            # "result/16c24g-result.csv"
            # "result/24c32g-result.csv"
          ]
  target = "result/24c32g-result.csv"

  kernels = {"RBF kernel": 1.0 * RBF() + WhiteKernel(),
             "Rational quadratic Kernel": 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1) + WhiteKernel(),
             "Matern kernel": 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5) + WhiteKernel(), 
             "Exponential quadratic kernel": 1.0 * Exponentiation(RationalQuadratic(), exponent=2)}
  sizes = [1, 5, 10, 15]
  df = pd.DataFrame(index=sizes)
  for name in kernels:
    kernel = kernels[name]
    scores = []
    for size in sizes:
      m = MLModel(workload, system, GaussianProcessRegressor(kernel=kernel, random_state=0), True)
      scores.append(m.cross_validation(train, target, train_size=size))
    df[f"{name}"] = scores

  m = MLModel("a", LineairDBConfiguration(), LinearRegression(), True, NoCoreEngineer())
  scores = []
  for size in sizes:
    scores.append(m.cross_validation(train, target, train_size=size))
  df["Linear regression"] = scores
  plot_linegraph_from_df(df)


def series_to_df(series):
  df = pd.DataFrame(\
    {'tps': series.values,\
    'key1': [float(i.split('_')[0]) for i in series.index],\
    'key2': [float(i.split('_')[1]) for i in series.index],\
    }, index = series.index)
  df.sort_values(['key2'], ascending = [True], inplace = True)
  df.sort_values(['key1'], ascending = [True], inplace = True)
  return df

def plot_all_data():
  workload = "a"
  system = LineairDBConfiguration()
  train = [
            "result/4c6g-result.csv",
            "result/8c12g-result.csv",
            # "result/12c16g-result.csv",
            # "result/16c24g-result.csv"
            # "result/24c32g-result.csv"
          ]
  target = "result/24c32g-result.csv"
  normalize = False
  regressor = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), random_state=0)
  make_data = MLModel(workload, system, LinearRegression(), False, normalize=normalize)
  model = MLModel(workload, system, regressor, False, normalize=normalize)
  log_model = MLModel(workload, system, LinearRegression(), False, LogCoreDistanceEngineer(), normalize=normalize)
  new = MultivariateRegressionPipeline()
  model.make_model(train, target, 15)
  log_model.make_model(train, target, 1)
  new.fit_with_target(train, target, 1)

  val_data = make_data.make_domain_target_data(train + [target])
  predict_feature = val_data.drop(columns="tps")
  prediction = model.predict(predict_feature)
  
  val_data = make_data.make_domain_target_data(train + [target])
  predict_feature = val_data.drop(columns="tps")
  log_prediction = new.model.predict(predict_feature)


  real_df = ValovModel(workload, system).extract_columns(val_data, ["num_core", "clients"])

  val_data["tps"] = prediction
  prediction = ValovModel(workload, system).extract_columns(val_data, ["num_core", "clients"])

  val_data["tps"] = log_prediction
  log_prediction = ValovModel(workload, system).extract_columns(val_data, ["num_core", "clients"])
  predict_df = series_to_df(prediction)
  log_predict_df = series_to_df(log_prediction)
  real_df = series_to_df(real_df)


  for y_values, name, mark in zip([predict_df["tps"], log_predict_df["tps"], real_df["tps"]], ["gp prediction", "log_prediction", "real"], ["x", "+", "o"]):
      x_values = [f"{core}_{thread}" for core, thread in zip(real_df["key1"], real_df["key2"])]
      plt.plot(x_values, y_values, marker=mark, linewidth=1, markersize=2, label=name)

  plt.legend(fontsize=16)
  plt.xticks(rotation=90)
  plt.tight_layout()
  plt.savefig("fig.pdf", bbox_inches='tight')
  plt.close()
  

def feature_engineering():
  train = [
            "result/4c6g-result.csv",
            "result/8c12g-result.csv",
            "result/12c16g-result.csv",
            # "result/16c24g-result.csv"
            # "result/24c32g-result.csv"
          ]
  target = "result/24c32g-result.csv"

  models = {"LinearRegression": LinearRegression()}
  sizes = [1, 3, 5, 10, 15]
  df = pd.DataFrame(index=sizes)

  name = "LR"
  # for name in models:
  #   m = MLModel("a", LineairDBConfiguration(), models[name], True, LogCoreDistanceEngineer())
  #   scores = []
  #   # scores.append(m.validation(train, target))
  #   for size in sizes:
  #     if size == 0: continue
  #     scores.append(m.cross_validation(train, target, train_size=size))
  #   df[f"{name} + L2S"] = scores

  # for name in models:
  #   m = MLModel("a", LineairDBConfiguration(), models[name], False, LogCoreDistanceEngineer())
  #   scores = []
  #   # scores.append(m.validation(train, target))
  #   for size in sizes:
  #     if size == 0: continue
  #     scores.append(m.cross_validation(train, target, train_size=size))
  #   df[f"{name} + L2S + Log"] = scores

  loader = MultivariateDataLoader("a", LineairDBConfiguration())
  m = MultivariateRegressionPipeline(loader)
  scores = []
  for size in sizes:
    scores.append(m.cross_validation(train, target, train_size=size))
  df[f"{name} + refactor"] = scores
  
  loader = MultivariateDataLoader("a", LineairDBConfiguration())
  m = MultivariateRegressionPipeline(loader, LogCoreDistanceEngineer())
  scores = []
  for size in sizes:
    scores.append(m.cross_validation(train, target, train_size=size))
  df[f"{name} + refactor + Log"] = scores



  # loader = SimpleRegressionDataLoader("a", LineairDBConfiguration(), None)
  # m = SimpleRegressionPipeline(loader)
  # scores = []
  # for size in sizes:
  #   scores.append(m.cross_validation(train[1], target, train_size=size))
  # df[f"{name} valov"] = scores

  plot_linegraph_from_df(df)

def cross_validation():
  train = [
            "result/4c6g-result.csv",
            "result/8c12g-result.csv",
            "result/12c16g-result.csv"
            # "result/24c32g-result.csv"
          ]
  target = "result/24c32g-result.csv"
  loader = SimpleRegressionDataLoader("a")
  ldb = ValovModel("a", LineairDBConfiguration(), True)
  # rt = ValovModel("a", LineairDBConfiguration(), model=DecisionTreeRegressor())

  models = {"RegressionTree": DecisionTreeRegressor(), "Lasso": LinearRegression(), "GaussianProcess": GaussianProcessRegressor()}
  sizes = [1, 5, 10, 15]
  df = pd.DataFrame(index=sizes)
  linear_scores = []
  # rt_scores = []
  for size in sizes:
    linear_scores.append(ldb.cross_validation(train[1], target, train_size=size))
    # rt_scores.append(rt.cross_validation(train[0], target, train_size=size))
  df["ValovLinear + L2S"] = linear_scores
  # df["ValovTree + L2S"] = rt_scores
  for name in models:
    m = MLModel("a", LineairDBConfiguration(), models[name], False, LogCoreDistanceEngineer())
    if name == "GaussianProcess": continue
    scores = []
    for size in sizes:
      scores.append(m.cross_validation(train, target, train_size=size))
    df[f"{name} + L2S"] = scores
  embed()
  plot_linegraph_from_df(df)

def plot_linegraph_from_df(df):
    fontsize = 22
    plt.figure(figsize=(9, 6))  # Adjust the figure size if needed
    
    for column in df.columns:
        plt.plot(df.index, df[column], marker='o', label=column)

    plt.xlabel('clients', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel('Throughput', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)  # Add legend to the plot
    plt.grid(True)  # Add grid lines for better readability
    plt.tight_layout()
    plt.savefig("model_comparison.pdf", bbox_inches='tight')
    plt.close()

def plot_performance_model():
  workload = "a"
  parameters = ["clients", "checkpoint_interval"]

  target = "result/24c32g-result.csv"
  predict_result = re.sub(r"(result/.*?)-result", r"\1-predict", target, count=1)
  graph = LineairDBGraph(workload)
  graph.plot_performance_model("fig.pdf", parameters, [predict_result, target])
  # graph.plot_gauss("fig.pdf", predict_result, target, parameters)

def regression_analysis(model_type):
  workload = "a"
  system = LineairDBConfiguration()
  train = [
            # "result/4c6g-result.csv",
            # "result/8c12g-result.csv",
            "result/12c16g-result.csv",
            "result/16c24g-result.csv"
          ]
  target = "result/24c32g-result.csv"

  num_core = getNumCore(target)
  predict_result = re.sub(r"(result/.*?)-result", r"\1-predict", target, count=1)

  normalize = False

  regression = MLModel(workload, system, model_type, True, normalize=normalize)
  # regression = SMLinearModel(workload, LineairDBConfiguration())
  regression.make_model(train, target)

  df_new = ValovModel(workload, system).drop_useless_columns(pd.read_csv(target)).drop(columns="tps")
  df_new["num_core"] = num_core
  if normalize:
    df_new["clients"] = df_new["clients"] / num_core
  tps = regression.predict(df_new)
  # tps, std = model.predict(df_new, return_std=True)
  # low = tps - 1.96 * std
  # high = tps + 1.96 * std
  # tps = model.predict(df_new)

  df_new["tps"] = tps
  # df_new["low"] = low
  # df_new["high"] = high
  df_new["workload"] = workload
  if normalize:
    df_new["clients"] = df_new["clients"] * num_core
  df_new.to_csv(predict_result)
  plot_performance_model()

def plot_single_param():
  workloads = ["a", "c", "wo"]
  parameters = ["clients", "epoch_duration", "checkpoint_interval", "rehash_threshold", "prefetch_locality"]
  for workload in workloads:
    graph = LineairDBGraph(workload)
    for param in parameters:
      os.makedirs(f"figure/perf_model/workload_{workload}", exist_ok=True)
      graph.plot_performance_model(f"figure/perf_model/workload_{workload}/{param}.pdf", [param])


def plot_all():
  workloads = ["a", "c", "wo"]
  machine_name = ["4c6g", "8c12g", "12c16g", "16c24g", "24c32g"]
  for workload in workloads:
    graph = LineairDBGraph(workload)
    
    os.makedirs(f"figure/perf_model", exist_ok=True)
    graph.plot_performance_model(f"figure/perf_model/workload_{workload}.pdf")

    for l2s in [True, False]:
      dir = f"figure/regression/workload_{workload}"
      os.makedirs(dir, exist_ok=True)
      for i in [0, 1, 2, 3, 4]:
        for j in [1, 2, 3, 4]:
          if i >= j: continue
          plot_path = f"{dir}/{machine_name[i]}_vs_{machine_name[j]}_l2s_{l2s}.pdf"
          graph.plot_regression(graph.file_names[i], graph.file_names[j], plot_path, not l2s)

if __name__ == "__main__":
  main()