"""
Phase6MultiRunAggregator.py
============================

Phase 6 - Multi-run aggregator: mean +/- std across independent repetitions (v5).

PURPOSE
-------
Walks the per-run outputs (now uniquely named via DRO_RUN_SUFFIX=_run{i})
under data/ and data/figures/, computes mean / std across runs, writes the
aggregated summary CSV, the forest plot, and the multi-run interpretation
files at the top level of data/multirun/.

v5 BUGFIX
---------
In v5, RunMulti.py sets DRO_RUN_SUFFIX=_run{i} per run, so each Phase 1-5
subprocess writes its outputs as Divvy_202608_run{i}_*.* files DIRECTLY
to data/ and data/figures/.  This eliminates the v4 bug where MoveRunOutputs
ToPerRunFolder ran after each phase batch and re-cleared the Run{i} folders.

Phase6 simply reads the per-run CSVs (which already have the _run{i}_ suffix)
from data/ and produces the aggregated summary at data/multirun/.
"""

# Import the argparse module for command-line argument parsing.
import argparse
# Import the pandas library for tabular data manipulation.
import pandas as pd
# Import the numpy library for numerical array operations.
import numpy as np
# Import the matplotlib pyplot module for static chart generation.
import matplotlib.pyplot as plt
# Import the seaborn library for enhanced visualizations.
import seaborn as sns
# Import the os library for file path construction.
import os
# Import the shared configuration module for global knobs and paths.
import Config
# Import the interpretation utility for auto-generated report writing.
import Interpretation

# Configure matplotlib global settings.
plt.rcParams["axes.unicode_minus"] = False
# Enable constrained layout.
plt.rcParams["figure.constrained_layout.use"] = True
# Set the savefig DPI.
plt.rcParams["savefig.dpi"] = 300
# Enable the grid.
plt.rcParams["axes.grid"] = True
# Set the grid linestyle.
plt.rcParams["grid.linestyle"] = "--"
# Set the grid alpha.
plt.rcParams["grid.alpha"] = 0.45

# Reuse the method edge colors from Phase5Visualization.
METHOD_EDGE_COLORS = {"EV": "#5B8AA6", "SAA": "#4F7F77", "DRO": "#1F3F6E"}


def BuildPerRunTag(cityKey, yearMonthTag, runIndex):
  """Build the per-run tag used to locate per-run artifacts."""
  # Return the joined per-run tag.
  return "%s_%s_run%d" % (cityKey, yearMonthTag, runIndex)


def AggregateMultiRunSummaries(cityKey, yearMonthTag, nRuns):
  """Walk each Run{i} subfolder and aggregate per-method statistics across runs.

  In v5, per-run outputs are tagged with the _run{i} suffix and live in data/.
  For backwards compatibility we also check the data/multirun/Run{i}/ subfolders.
  """
  # Initialize the per-run rows list.
  perRunRows = []
  # Iterate over each run index.
  for runIndex in range(1, nRuns + 1):
    # Build the per-run tag.
    runTag = BuildPerRunTag(cityKey, yearMonthTag, runIndex)
    # Build the per-run summary path (v5: directly in data/).
    summaryPathV5 = os.path.join(Config.DATA_DIRECTORY,
                                 "%s_statistical_summary.csv" % runTag)
    # Build the per-run summary path (v4: in data/multirun/Run{i}/).
    summaryPathV4 = os.path.join(Config.MultirunRunDirectory(runIndex),
                                 "%s_statistical_summary.csv" % runTag)
    # Choose the path that exists (v5 first, then v4).
    summaryPath = summaryPathV5 if (os.path.exists(summaryPathV5)) else summaryPathV4
    # Build the per-run comparative path (v5).
    comparativePathV5 = os.path.join(Config.DATA_DIRECTORY,
                                     "%s_comparative_validation.csv" % runTag)
    # Build the per-run comparative path (v4).
    comparativePathV4 = os.path.join(Config.MultirunRunDirectory(runIndex),
                                     "%s_comparative_validation.csv" % runTag)
    # Choose the path that exists.
    comparativePath = comparativePathV5 if (os.path.exists(comparativePathV5)) else comparativePathV4
    # Skip when the summary file is missing.
    if (not os.path.exists(summaryPath)):
      # Print a warning.
      print("  [warn] missing summary for %s; skipping this run." % runTag)
      # Continue to the next run.
      continue
    # Read the summary CSV.
    summaryFrame = pd.read_csv(summaryPath)
    # Add the run index column.
    summaryFrame["RunIndex"] = runIndex
    # Add the run tag column.
    summaryFrame["RunTag"] = runTag
    # Append to the per-run rows.
    perRunRows.append(summaryFrame)
  # If no per-run rows were collected, raise an error.
  if (not perRunRows):
    # Raise a runtime error.
    raise RuntimeError("No per-run statistical summaries found for city=%s, yearMonthTag=%s, nRuns=%d." % (
      cityKey, yearMonthTag, nRuns))
  # Concatenate all per-run dataframes.
  perRunDataFrame = pd.concat(perRunRows, ignore_index=True)
  # Return the per-run dataframe.
  return perRunDataFrame


def ComputeMeanStdAcrossRuns(perRunDataFrame):
  """Group by Method and compute mean / std across runs."""
  # Initialize the aggregated rows list.
  aggregatedRows = []
  # Iterate over each method group.
  for methodName, groupFrame in perRunDataFrame.groupby("Method"):
    # Append the aggregated row.
    aggregatedRows.append({
      "Method"                   : methodName,
      "RunsAggregated"           : int(len(groupFrame)),
      "MeanCost_MeanAcrossRuns"  : float(groupFrame["MeanCost"].mean()),
      "MeanCost_StdAcrossRuns"   : float(groupFrame["MeanCost"].std(ddof=1)),
      "MedianCost_MeanAcrossRuns": float(groupFrame["MedianCost"].mean()),
      "MedianCost_StdAcrossRuns" : float(groupFrame["MedianCost"].std(ddof=1)),
      "StdCost_MeanAcrossRuns"   : float(groupFrame["StdCost"].mean()),
      "StdCost_StdAcrossRuns"    : float(groupFrame["StdCost"].std(ddof=1)),
      "Ci95Lower_MeanAcrossRuns" : float(groupFrame["Ci95Lower"].mean()),
      "Ci95Upper_MeanAcrossRuns" : float(groupFrame["Ci95Upper"].mean()),
    })
  # Convert to a dataframe.
  return pd.DataFrame(aggregatedRows).sort_values("Method").reset_index(drop=True)


def GenerateMultiRunForestPlot(perRunDataFrame, aggregatedDataFrame, cityKey, yearMonthTag, nRuns):
  """Forest plot showing per-run means and the across-run mean +/- std."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(9.2, 5.6))
  # Define the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Iterate over each method.
  for methodIndex, methodName in enumerate(methodOrder):
    # Subset the per-run dataframe.
    methodSubset = perRunDataFrame[perRunDataFrame["Method"] == methodName].sort_values("RunIndex")
    # Iterate over each row.
    for _, row in methodSubset.iterrows():
      # Compute the y position.
      yPos = methodIndex + 0.1 * (row["RunIndex"] - (nRuns + 1) / 2.0)
      # Get the color.
      color = METHOD_EDGE_COLORS[methodName]
      # Plot the per-run point.
      axisObject.errorbar([row["MeanCost"]], [yPos],
                          xerr=[[row["MeanCost"] - row["Ci95Lower"]], [row["Ci95Upper"] - row["MeanCost"]]],
                          fmt="o", color=color, ecolor=color, alpha=0.55,
                          markersize=6, elinewidth=1.0, capsize=4)
    # Get the aggregated row for the method.
    aggregatedRow = aggregatedDataFrame[aggregatedDataFrame["Method"] == methodName].iloc[0]
    # Get the across-run mean.
    acrossMean = aggregatedRow["MeanCost_MeanAcrossRuns"]
    # Get the across-run std.
    acrossStd = aggregatedRow["MeanCost_StdAcrossRuns"]
    # Get the color.
    color = METHOD_EDGE_COLORS[methodName]
    # Plot the aggregated mean +/- std as a thicker square.
    axisObject.errorbar([acrossMean], [methodIndex - 0.45], xerr=[[acrossStd], [acrossStd]],
                        fmt="s", color=color, ecolor=color, markersize=14,
                        elinewidth=2.2, capsize=8, capthick=2.2,
                        label="%s (mean +/- std across %d runs)" % (methodName, nRuns))
    # Annotate the mean +/- std.
    axisObject.annotate("%.2f +/- %.2f" % (acrossMean, acrossStd),
                        xy=(acrossMean, methodIndex - 0.45), xytext=(8, 6),
                        textcoords="offset points", fontsize=9, color=color, fontweight="bold")
  # Set the y-axis ticks.
  axisObject.set_yticks(range(len(methodOrder)))
  # Set the y-axis tick labels.
  axisObject.set_yticklabels(methodOrder)
  # Set the x-axis label.
  axisObject.set_xlabel("Out-of-Sample Mean Operational Cost (USD)")
  # Set the y-axis label.
  axisObject.set_ylabel("Decision Model")
  # Set the title.
  axisObject.set_title("Multi-Run Stability: Per-Run Means (faint dots) and Across-Run Mean +/- Std (squares)\n"
                       "Smaller across-run std = more stable conclusion", fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(loc="lower right", frameon=True, fancybox=True)
  # Build the source note text.
  noteText = ("Source: %s %s trip data  |  %d independent runs with different random seeds  |  "
              "Forest plot of mean cost per run") % (cityKey, yearMonthTag, nRuns)
  # Annotate the source note.
  axisObject.annotate(noteText, xy=(0.5, -0.16), xycoords="axes fraction",
                      ha="center", va="top", fontsize=7, style="italic", color="#555555")
  # Build the base name.
  baseName = "multirun%d_forest_plot" % nRuns
  # Build the PDF path.
  pdfPath = Config.MultirunAggregatedFigureName(cityKey, yearMonthTag, nRuns, baseName + ".pdf")
  # Build the PNG path.
  pngPath = Config.MultirunAggregatedFigureName(cityKey, yearMonthTag, nRuns, baseName + ".png")
  # Save the figure as PDF.
  figureObject.savefig(pdfPath, dpi=300)
  # Save the figure as PNG.
  figureObject.savefig(pngPath, dpi=150)
  # Close the figure.
  plt.close(figureObject)
  # Return the paths.
  return pdfPath, pngPath


def main():
  """Main entry point: aggregate per-run summaries and write the forest plot."""
  # Build the argument parser.
  parser = argparse.ArgumentParser(description="Multi-run aggregator for the DRO pipeline.")
  # Add the city argument.
  parser.add_argument("--city", default=Config.ACTIVE_CITY_KEY, help="City key (default: %(default)s)")
  # Add the start year-month argument.
  parser.add_argument("--start-year-month", default=Config.ACTIVE_START_YEAR_MONTH,
                      help="Start year-month YYYYMM (default: %(default)s)")
  # Add the end year-month argument.
  parser.add_argument("--end-year-month", default=Config.ACTIVE_END_YEAR_MONTH,
                      help="End year-month YYYYMM (default: %(default)s)")
  # Add the n-runs argument.
  parser.add_argument("--n-runs", type=int, default=5, help="Number of runs to aggregate (default: %(default)s)")
  # Parse the arguments.
  args = parser.parse_args()
  # Get the city key.
  cityKey = args.city
  # Build the year-month tag.
  if (args.start_year_month == args.end_year_month):
    # Use the single-month tag.
    yearMonthTag = args.start_year_month
  else:
    # Use the range tag.
    yearMonthTag = "%s-%s" % (args.start_year_month, args.end_year_month)
  # Get the number of runs.
  nRuns = args.n_runs
  # Print the header.
  print("Phase 6 multi-run aggregator: city=%s year-month-tag=%s n-runs=%d" % (
    cityKey, yearMonthTag, nRuns))
  # Aggregate the per-run summaries.
  perRunDataFrame = AggregateMultiRunSummaries(cityKey, yearMonthTag, nRuns)
  # Compute the across-run mean/std.
  aggregatedDataFrame = ComputeMeanStdAcrossRuns(perRunDataFrame)
  # Build the aggregated summary path.
  summaryPath = Config.MultirunAggregatedName(cityKey, yearMonthTag, nRuns, "summary.csv")
  # Save the aggregated summary.
  aggregatedDataFrame.to_csv(summaryPath, index=False)
  # Print the summary path.
  print("  -> aggregated summary written to %s" % summaryPath)
  # Print the aggregated dataframe.
  print(aggregatedDataFrame.to_string(index=False))
  # Build the per-run path.
  perRunPath = Config.MultirunAggregatedName(cityKey, yearMonthTag, nRuns, "perrun.csv")
  # Save the per-run dataframe.
  perRunDataFrame.to_csv(perRunPath, index=False)
  # Print the per-run path.
  print("  -> per-run raw summary written to %s" % perRunPath)
  # Generate the forest plot.
  pdfPath, pngPath = GenerateMultiRunForestPlot(perRunDataFrame, aggregatedDataFrame,
                                                cityKey, yearMonthTag, nRuns)
  # Print the forest plot path.
  print("  -> forest plot written to %s and %s" % (pdfPath, pngPath))
  # Get the configuration.
  configuration = Config.EXPERIMENT_CONFIGURATION
  # Invoke the interpretation utility for the multi-run case.
  Interpretation.GenerateMultiRunInterpretation(
    cityKey, yearMonthTag, nRuns, aggregatedDataFrame, configuration)
  # Print the completion message.
  print("Phase 6 complete: aggregated %d runs across %d methods." % (
    nRuns, aggregatedDataFrame.shape[0]))


# Execute the main function when the script runs as the entry point.
if __name__ == "__main__":
  # Invoke the main entry point.
  main()
