import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from IPython import embed
import numpy as np

class PlotDesign:
    def __init__(self, x_label: str, y_label: str, figsize: (int, int) = (6.4, 4.8),
                 draw_x_ticks: list = True, x_tick_rotation: int = 0, x_tick_fontsize: int = 22,
                 xlim: list = None, ylim: list = None,
                 fontsize: int = 22, line_width: int = 2, marker_size: int = 10,
                 line_colors: list[str] = ['red', 'blue', 'green', 'orange', 'purple', 'brown'], 
                 line_styles: list[str] = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 1)), (0, (3, 10, 1, 10))]) -> None:
      self.x_label: str = x_label
      self.y_label: str = y_label
      self.figsize: (int, int) = figsize
      self.x_ticks: bool = draw_x_ticks
      self.x_tick_rotation: int = x_tick_rotation
      self.x_tick_fontsize: int = x_tick_fontsize
      self.xlim: list = xlim
      self.ylim: list = ylim
      self.fontsize: int = fontsize
      self.line_width: int = line_width
      self.marker_size: int = marker_size
      self.line_colors: list = line_colors
      self.line_styles: list = line_styles

def plot_linegraph_from_df(df: DataFrame, design: PlotDesign, output_name: str):
    if ".pdf" not in output_name:
        output_name = output_name + ".pdf"
        
    plt.figure(figsize=design.figsize)  # Adjust the figure size if needed
    
    for column, color, style in zip(df.columns, design.line_colors, design.line_styles):
        plt.plot(df.index, df[column], marker='x', label=column, color=color, linestyle=style, linewidth=design.line_width, markersize=design.marker_size)

    if design.xlim != None: plt.xlim(design.xlim)
    if design.ylim != None: plt.ylim(design.ylim)
    if "%" in design.y_label:
        import matplotlib.ticker as mtick
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, symbol=''))
    
    plt.xlabel(f"{design.x_label}", fontsize=design.fontsize)
    if design.x_ticks == True: plt.xticks(fontsize=design.x_tick_fontsize, rotation=design.x_tick_rotation) 
    else: plt.xticks([])
    plt.ylabel(f"{design.y_label}", fontsize=design.fontsize)
    plt.yticks(fontsize=design.fontsize)
    plt.legend(fontsize=design.fontsize)  # Add legend to the plot
    plt.grid(True)  # Add grid lines for better readability
    plt.tight_layout()
    plt.savefig(f"{output_name}", bbox_inches='tight')
    plt.close()
    
    
def plot_bargraph(df: DataFrame, design: PlotDesign, output_name: str, thresholds: list = [70, 60, 50, 40, 30, 20, 10, 5]):
    if ".pdf" not in output_name:
        output_name = output_name + ".pdf"
        
    table = {}
    for threshold in thresholds:
        row = []
        for column in df.columns:
            for size in df.index:
                error = df[column][size] * 100
                if error <= threshold:
                    row.append(size)
                    break
                if size == df.index[-1]:
                    row.append(np.nan)
        table[f"{threshold}"] = row

    bar_df = DataFrame(table).T
    bar_df.columns = df.columns

    plots = bar_df.plot(kind="bar", figsize=design.figsize, rot=0, fontsize=design.fontsize, legend=False, color=design.line_colors, width=0.8)

    plt.grid(True)  # Add grid lines for better readability
    plt.ylabel(f"{design.y_label}", fontsize=design.fontsize)
    plt.xlabel(f"{design.x_label}", fontsize=design.fontsize)
    plt.tight_layout()
    output_name = output_name.replace(".pdf", "_barplot.pdf")
    plt.legend(bbox_to_anchor =(1.05, 1.2), ncol = 3, fontsize=design.fontsize-4)
    plt.savefig(f"{output_name}", bbox_inches='tight')
    plt.close()
    

def save_legends(ax, output_name: str):
    if ".pdf" not in output_name:
        output_name = output_name + ".pdf"
    legendFig = plt.figure("Legend plot")
    legendFig.legend(*ax.get_legend_handles_labels(), loc='center', fontsize=22)
    legendFig.savefig(f"{output_name}", bbox_inches='tight')
    plt.close()

def scatterplot_from_df(df: DataFrame, column_name: str, design: PlotDesign, output_name: str):
    if ".pdf" not in output_name:
        output_name = output_name + ".pdf"

    x = df.loc[:, [f"{column_name}_x"]].values
    y = df[f"{column_name}_y"]
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)
    parameters = df['features'].drop_duplicates().iloc[0]

    plt.figure(figsize=design.figsize)
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.xlabel(f"{design.x_label}", fontsize=design.fontsize)
    plt.ylabel(f"{design.y_label}", fontsize=design.fontsize)
    
    # only 1 parameter
    if "_" not in df["config"].iloc[0]:
        for i, label in enumerate(df["config"]): 
            plt.text(x[i], y[i], label, fontsize=design.fontsize - 2)
    plt.annotate(f"r-squared = {format(r2, '.3f')}\nn = {len(x)}\nparameters: {parameters}", \
                  xy=(0.05, 0.85), xycoords='axes fraction', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=design.fontsize - 2)
    plt.tight_layout()
    plt.savefig(f"{output_name}", bbox_inches='tight')
    plt.close()


def machine_datasize_axis_label(filename: str):
    name = filename.removesuffix(".csv")
    strings = name.split("-")
    label = " ".join(strings)
    return label
