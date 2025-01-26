import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed
import os
import re
from sklearn import linear_model
from sklearn.metrics import r2_score

class Graph:
  def getPerformanceFunction(self, machine):
    df = df.groupby(self.configuration)[self.performance].mean().reset_index()
    return df
  
  def readMachineCSV(self, machine, data_directory):
    path_to_csv = getParentDirectoryOfThisFile()+f"/{data_directory}"
    return pd.read_csv(f'{path_to_csv}/{machine}.csv')
  
  def setDataDirectory(self, data_directory): 
    self.output_directory = self.output_directory.replace(self.data_directory, data_directory)
    self.data_directory = data_directory

  def setOutputDirectory(self, directory): self.output_directory = directory
  
  def __init__(self):
    self.machines = [
                    "lineairdb4v",
                    "lineairdb8v",
                    "lineairdb16v"
                    ]
    self.configuration = "checkpoint_interval" # X
    self.performance="tps" # Y
    self.data_directory = "20230802_ycsb_c_silo_handler"
    self.output_directory = f'{getParentDirectoryOfThisFile()}/{self.data_directory}'

    self.LINE_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    self.LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed']


class Linegraph(Graph):
  def outputLinegraph(self):
    for machine, color, style in zip(self.machines, self.LINE_COLORS, self.LINE_STYLES):
      performance_function_df = self.getPerformanceFunction(machine)
      plt.plot(performance_function_df[self.configuration], performance_function_df[self.performance], label=machine, color=color, linestyle=style, marker="x")
    self.labelNumThreadsVsThroughput()
    self.saveGraph(self.configuration)
    plt.close()

  def labelNumThreadsVsThroughput(self):
    plt.legend()
    plt.xlabel("Number of Threads", fontsize=12)
    plt.ylabel("Throughput (transactions/sec)", fontsize=12)
    plt.show()

  def saveGraph(self, filename):
    os.makedirs(f"{self.output_directory}/linegraph", exist_ok=True)
    plt.savefig(f"{self.output_directory}/linegraph/{filename}.pdf")

class Regression(Graph):
  def outputScatterPlot(self):
    for x in self.machines:
      for y in self.machines:
        if getNumCore(x) >= getNumCore(y): continue
        df_x = self.getPerformanceFunction(x)
        df_y = self.getPerformanceFunction(y)
        CONFIGURATION = df_x[self.configuration]
        X_PERFORMANCE = df_x.loc[:, [self.performance]].values
        Y_PERFORMANCE = df_y.loc[:, self.performance].values
        linear_regression_model = linear_model.LinearRegression()
        linear_regression_model.fit(X_PERFORMANCE, Y_PERFORMANCE)

        plt.figure()
        plt.scatter(X_PERFORMANCE, Y_PERFORMANCE)
        plt.plot(X_PERFORMANCE, linear_regression_model.predict(X_PERFORMANCE))

        self.label(x, y, CONFIGURATION, X_PERFORMANCE, Y_PERFORMANCE, linear_regression_model)
        self.saveGraph(f"{getNumCore(x)}_vs_{getNumCore(y)}")
        plt.close()

  def label(self, x_name, y_name, configuration, x_performance, y_performance, model): 
    self.labelXYaxis(x_name, y_name)
    self.labelConfigurations(configuration, x_performance, y_performance)
    r2 = self.calculateR2(model, x_performance, y_performance)
    self.annotate(r2, len(x_performance))

  def labelXYaxis(self, x, y):
    plt.xlabel(f"{x} Throughput", fontsize=12)
    plt.ylabel(f"{y} Throughput", fontsize=12)
  
  def labelConfigurations(self, configuration, X, Y):
    for i, label in enumerate(configuration): 
      plt.text(X[i], Y[i], label, fontsize=8)

  def calculateR2(self, model, X, Y):
    Y_PRED = model.predict(X)
    return r2_score(Y, Y_PRED)

  def annotate(self, r2, n):
    plt.annotate(f"r-squared = {format(r2, '.3f')}\n \
                  n = {n}\n \
                  label: {self.configuration}", \
                  xy=(0.05, 0.85), xycoords='axes fraction', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    
  def saveGraph(self, x_vs_y):
    os.makedirs(f'{self.output_directory}/regression/{self.configuration}', exist_ok=True)
    plt.savefig(f"{self.output_directory}/regression/{self.configuration}/{x_vs_y}.pdf")

def getParentDirectoryOfThisFile():
  return os.path.dirname(os.path.realpath(__file__))

def getNumCore(spec_string):
  return int(re.search(r'\d+', spec_string).group())

def getMemSize(spec_string):
  return int(re.search(r'\d+c(\d+)g', spec_string).group(1))