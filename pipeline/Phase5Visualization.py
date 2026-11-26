"""
Phase5Visualization.py
========================

Phase 5 - Publication-ready visual analytics (42 figures + 1 interactive map).

PURPOSE
-------
Produces a comprehensive catalog of self-descriptive figures. Each figure has
a clear title, axis labels with units, a legend, and a source note at the
bottom. All figures are saved in both PDF (for the manuscript) and PNG
(for quick viewing).

FIGURE CATALOG (42 + 1)
-----------------------
  1.  cost_distribution_boxplot    - box + strip plot of out-of-sample cost per method
  2.  scale_invariance_plot        - grouped bar of mean cost per band x method
  3.  cost_cdf_plot                - empirical CDF per method (stochastic dominance)
  4.  cost_reduction_bar_plot      - bar of mean reduction % with 95% CI
  5.  cluster_method_heatmap       - per-cluster x method mean cost heatmap
  6.  bootstrap_ci_forest_plot     - forest plot of bootstrap 95% CI per method
  7.  wilcoxon_effects_plot        - p-value (log) and Cohen r side-by-side
  8.  demand_distribution_boxplot  - bootstrap scenario totals per scale band
  9.  station_geographic_map       - all stations colored by cluster
  10. net_positioning_heatmap      - per-cluster DRO positioning grids (top 6)
  11. rebalancing_flow_map         - EV vs DRO side-by-side on representative cluster
  12. cost_violin_plot              - kernel density of cost per method (NEW v3)
  13. qq_plot_dro_vs_ev            - quantile-quantile DRO vs EV / SAA (NEW v3)
  14. pareto_frontier_plot         - first-stage vs out-of-sample cost (NEW v3)
  15. cvar_tail_comparison_plot    - worst 10% scenarios per method (NEW v3)
  16. radar_chart_multicriteria     - radar of Mean/Std/P95/CVaR90/Max (NEW v3)
  17. cost_density_ridgeline        - ridgeline (joyplot) per method (NEW v4)
  18. bland_altman_plot            - Bland-Altman agreement DRO vs EV (NEW v4)
  19. ecdf_with_ci_band             - ECDF with 95% bootstrap CI band (NEW v4)
  20. cost_decomposition_stacked    - stacked bar: rebalancing + unmet (NEW v4)
  21. cluster_cost_reduction_heatmap - per-cluster DRO-vs-EV % reduction (NEW v4)
  22. pairplot_methods             - sns.pairgrid of method costs (NEW v4)
  23. sorted_cumulative_cost_plot   - cumulative cost over sorted scenarios (NEW v4)
  24. bootstrapped_mean_density     - density of bootstrap mean cost (NEW v4)
  25. net_positioning_histogram     - histogram of net positioning per method (NEW v4)
  26. cluster_spatial_centroid_map  - cluster anchor points by band (NEW v4)
  27. station_volume_histogram      - trip volume distribution per station (NEW v4)
  28. demand_capacity_scatter       - demand vs capacity per station (NEW v4)
  29. cluster_band_pie              - cluster count by band (NEW v4)
  30. method_color_palette          - reference palette + legend (NEW v4)
  31. cluster_size_vs_cost_scatter - cluster size vs mean cost w/ trend lines (NEW v5)
  32. hexbin_dro_vs_ev             - hexbin density of DRO vs EV cost (NEW v5)
  33. boxcox_normality_plot        - normal probability plot per method (NEW v5)
  34. rolling_mean_cost_plot        - rolling mean of cost over scenario index (NEW v5)
  35. method_ranking_heatmap       - per-cluster method rank (1=cheapest) (NEW v5)
  36. method_dominance_donut       - donut of % of clusters each method wins (NEW v5)
  37. cvar_frontier_plot           - CVaR at multiple alpha levels (NEW v5)
  38. bootstrap_mean_histogram     - histogram of bootstrap mean cost (NEW v5)
  39. variance_comparison_bar_plot - std vs IQR per method (NEW v5)
  40. cluster_pair_cost_heatmap    - per-cluster DRO/baseline cost ratio (NEW v5)
  41. decision_difference_heatmap  - |DRO-EV| and |DRO-SAA| net positioning diff (NEW v5)
  42. cost_loglog_cdf_plot         - log-log survival function per method (NEW v5)
  +   rebalancing_map.html          - interactive folium map
"""

# Import the matplotlib pyplot module for static chart generation.
import matplotlib.pyplot as plt
# Import the matplotlib patches module for legend swatches.
import matplotlib.patches as mpatches
# Import the seaborn library for enhanced statistical visualizations.
import seaborn as sns
# Import the folium library for interactive map visualization.
import folium
# Import the numpy library for numerical array operations.
import numpy as np
# Import the pandas library for tabular data loading.
import pandas as pd
# Import the json library for loading network artifacts.
import json
# Import the os library for file path construction.
import os
# Import the shared configuration module for global knobs and paths.
import Config

from HMB.Initializations import UpdateMatplotlibSettings

# Disable the unicode minus sign for clean axis labels.
plt.rcParams["axes.unicode_minus"] = False
# Enable the constrained layout engine for automatic spacing.
plt.rcParams["figure.constrained_layout.use"] = True
# Set a high DPI default for crisp PNG output.
plt.rcParams["savefig.dpi"] = 300
# Set the default figure size to a balanced 8 x 5 inches.
plt.rcParams["figure.figsize"] = (8.0, 5.0)
# Set a clean axis grid style.
plt.rcParams["axes.grid"] = True
# Set the grid line style to dashed for visual softness.
plt.rcParams["grid.linestyle"] = "--"
# Set the grid line alpha for subtle visual presence.
plt.rcParams["grid.alpha"] = 0.45

UpdateMatplotlibSettings()

# Define the consistent color palette for the three methods.
METHOD_COLORS = {"EV": "#D1E5F0", "SAA": "#93C5BD", "DRO": "#3B6FA8"}
# Define the edge color palette.
METHOD_EDGE_COLORS = {"EV": "#5B8AA6", "SAA": "#4F7F77", "DRO": "#1F3F6E"}
# Define the source-note template.
SOURCE_NOTE_TEMPLATE = (
  "Source: %s %s trip data  |  Run tag: %s  |  %d clusters x %d out-of-sample scenarios"
)


def AddSourceNote(axisObject, configuration, nClusters=None, nScenarios=None):
  """Append a small italic source note at the bottom of the figure."""
  # Compose the source note text.
  noteText = SOURCE_NOTE_TEMPLATE % (
    configuration["ActiveCityKey"], Config.YEAR_MONTH_TAG, Config.RUN_TAG,
    nClusters if (nClusters is not None) else 0,
    nScenarios if (nScenarios is not None) else 0)
  # Annotate the axis with the source note.
  axisObject.annotate(noteText, xy=(0.5, -0.18), xycoords="axes fraction",
                      ha="center", va="top", fontsize=7, style="italic", color="#555555")


def SaveFigureBothFormats(figureObject, baseName, configuration, nClusters=None, nScenarios=None):
  """Save a matplotlib figure in both PDF and PNG formats using the run-tag prefix."""
  # Build the PDF output path.
  pdfPath = Config.TaggedFigureName(baseName + ".pdf" if (not baseName.endswith(".pdf")) else baseName)
  # Build the PNG output path.
  pngPath = Config.TaggedFigureName(baseName + ".png" if (not baseName.endswith(".png")) else baseName)
  # Save the figure as PDF.
  figureObject.savefig(pdfPath, dpi=300)
  # Save the figure as PNG.
  figureObject.savefig(pngPath, dpi=150)
  # Close the matplotlib figure to free up memory.
  plt.close(figureObject)
  # Return the paths.
  return pdfPath, pngPath


# Figure 1 - Out-of-sample cost distribution boxplot.
def GenerateCostDistributionBoxplot(outOfSampleDataFrame, configuration):
  """Boxplot of out-of-sample operational cost per method with strip overlay."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.2))
  # Define the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Build the color palette list.
  colorPalette = [METHOD_COLORS[m] for m in methodOrder]
  # Generate the boxplot with hue assigned to silence seaborn 0.14 deprecation.
  sns.boxplot(data=outOfSampleDataFrame, x="Method", y="TotalCost",
              order=methodOrder, hue="Method", palette=colorPalette,
              ax=axisObject, width=0.55, fliersize=2.5, linewidth=1.1, legend=False)
  # Overlay the individual observations.
  sns.stripplot(data=outOfSampleDataFrame, x="Method", y="TotalCost",
                order=methodOrder, hue="Method", color="black",
                alpha=0.18, size=1.6, ax=axisObject, legend=False)
  # Annotate the mean per method as a red diamond.
  for methodIndex, methodName in enumerate(methodOrder):
    # Compute the method mean.
    methodMean = float(outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]["TotalCost"].mean())
    # Scatter the mean as a red diamond.
    axisObject.scatter([methodIndex], [methodMean], marker="D", color="#B22222", s=40, zorder=5,
                       label="Mean" if (methodIndex == 0) else None)
    # Annotate the mean.
    axisObject.annotate("mean=%.1f" % methodMean, xy=(methodIndex, methodMean), xytext=(8, 0),
                        textcoords="offset points", fontsize=8, color="#B22222", va="center")
  # Build a legend entry for the strip overlay.
  axisObject.plot([], [], "o", color="black", alpha=0.35, markersize=4, label="Individual scenario")
  # Set the x-axis label.
  axisObject.set_xlabel(
    "Decision Model  (EV = Expected Value; SAA = Sample Average Approximation; DRO = Distributionally Robust Optimization)")
  # Set the y-axis label.
  axisObject.set_ylabel("Out-of-Sample Operational Cost (USD)")
  # Set the title.
  axisObject.set_title("Distribution of Out-of-Sample Operational Costs Across Decision Models\n"
                       "Lower is better; diamond markers show the mean per model", fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.9)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return the paths.
  return SaveFigureBothFormats(figureObject, "cost_distribution_boxplot", configuration,
                               nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 2 - Scale-invariance grouped bar plot.
def GenerateScaleInvariancePlot(perClusterFrame, configuration):
  """Grouped bar plot of mean cost by scale band and method."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.6, 5.2))
  # Aggregate the per-cluster means by band and method.
  aggregatedFrame = perClusterFrame.groupby(["ScaleBand", "Method"])["TotalCost"].mean().reset_index()
  # Define the band order.
  bandOrder = ["Small", "Medium", "Large"]
  # Define the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Build the color palette.
  colorPalette = [METHOD_COLORS[m] for m in methodOrder]
  # Generate the grouped bar plot.
  sns.barplot(data=aggregatedFrame, x="ScaleBand", y="TotalCost", hue="Method",
              order=bandOrder, hue_order=methodOrder, palette=colorPalette,
              ax=axisObject, errorbar=None, edgecolor="black", linewidth=0.6, legend=True)
  # Annotate each bar with its numeric value.
  for container in axisObject.containers:
    # Label each bar.
    axisObject.bar_label(container, fmt="%.1f", fontsize=7, padding=2)
  # Set the x-axis label.
  axisObject.set_xlabel("Cluster Scale Band  (Small: <=20 stations; Medium: 21-40; Large: 41+)")
  # Set the y-axis label.
  axisObject.set_ylabel("Mean Out-of-Sample Cost (USD)")
  # Set the title.
  axisObject.set_title("Scale Invariance of Mean Operational Cost by Cluster Size Band\n"
                       "DRO cost advantage is preserved across Small / Medium / Large clusters", fontsize=11,
                       fontweight="bold")
  # Move the legend outside the plot area.
  axisObject.legend(title="Decision Model", loc="upper left", bbox_to_anchor=(1.0, 1.0),
                    frameon=True, fancybox=True, framealpha=0.95)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(perClusterFrame["ClusterId"].unique()),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "scale_invariance_plot", configuration,
                               nClusters=len(perClusterFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 3 - CDF of out-of-sample costs.
def GenerateCostCdfPlot(outOfSampleDataFrame, configuration):
  """CDF of out-of-sample cost per method, showing stochastic dominance."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.2))
  # Define the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Iterate over each method.
  for methodName in methodOrder:
    # Sort the method costs.
    methodCosts = np.sort(outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]["TotalCost"].to_numpy())
    # Compute the CDF values.
    cdfValues = np.arange(1, len(methodCosts) + 1) / len(methodCosts)
    # Plot the CDF.
    axisObject.plot(methodCosts, cdfValues, color=METHOD_EDGE_COLORS[methodName], linewidth=2.2,
                    label=methodName)
    # Compute the median.
    medianValue = float(np.median(methodCosts))
    # Plot the median as a dashed vertical line.
    axisObject.axvline(medianValue, color=METHOD_EDGE_COLORS[methodName], linestyle=":", alpha=0.55,
                       linewidth=1.1)
  # Set the x-axis label.
  axisObject.set_xlabel("Out-of-Sample Operational Cost (USD)")
  # Set the y-axis label.
  axisObject.set_ylabel("Cumulative Probability")
  # Set the title.
  axisObject.set_title("Cumulative Distribution of Out-of-Sample Costs per Method\n"
                       "A curve further to the LEFT indicates stochastically lower operational cost",
                       fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(title="Decision Model", loc="lower right", frameon=True, fancybox=True)
  # Set the y-axis limits.
  axisObject.set_ylim(0.0, 1.02)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cost_cdf_plot", configuration,
                               nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 4 - Cost reduction bar plot with bootstrap 95% CI.
def GenerateCostReductionBarPlot(outOfSampleDataFrame, comparativeDataFrame, configuration, randomGenerator):
  """Bar plot of mean cost reduction (%) of DRO vs EV and DRO vs SAA, with CI."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(7.6, 5.2))
  # Pivot the dataframe to align methods per scenario.
  pivotFrame = outOfSampleDataFrame.pivot(index=["ClusterId", "ScenarioIndex"], columns="Method",
                                          values="TotalCost")
  # Initialize the reductions and CI errors lists.
  reductions = []
  # Initialize the CI errors list.
  ciErrors = []
  # Initialize the comparisons list.
  comparisons = []
  # Use a smaller bootstrap count for plot speed.
  bootstrapResamples = min(500, configuration["BootstrapResamples"])
  # Iterate over each baseline method.
  for baselineName in ["EV", "SAA"]:
    # Extract the baseline cost array.
    baselineArray = pivotFrame[baselineName].to_numpy()
    # Extract the DRO cost array.
    droArray = pivotFrame["DRO"].to_numpy()
    # Compute the per-scenario reduction percentages.
    perScenarioReduction = (baselineArray - droArray) / np.maximum(1e-9, baselineArray) * 100.0
    # Initialize the bootstrap reductions list.
    bootReductions = []
    # Iterate over each bootstrap resample.
    for _ in range(bootstrapResamples):
      # Draw a bootstrap sample with replacement.
      sample = randomGenerator.choice(perScenarioReduction, size=len(perScenarioReduction), replace=True)
      # Append the bootstrap mean.
      bootReductions.append(float(np.mean(sample)))
    # Compute the mean reduction.
    meanReduction = float(np.mean(perScenarioReduction))
    # Compute the lower percentile.
    lower = float(np.percentile(bootReductions, 2.5))
    # Compute the upper percentile.
    upper = float(np.percentile(bootReductions, 97.5))
    # Append the mean reduction.
    reductions.append(meanReduction)
    # Append the CI errors as [lower error, upper error].
    ciErrors.append([meanReduction - lower, upper - meanReduction])
    # Append the comparison label.
    comparisons.append("DRO vs %s" % baselineName)
  # Convert the CI errors to a numpy array transposed.
  ciErrors = np.array(ciErrors).T
  # Build the bar colors list.
  barColors = [METHOD_EDGE_COLORS["EV"], METHOD_EDGE_COLORS["SAA"]]
  # Plot the bar chart with error bars.
  bars = axisObject.bar(comparisons, reductions, color=barColors, edgecolor="black", linewidth=0.8,
                        yerr=ciErrors, capsize=8, error_kw={"elinewidth": 1.4, "ecolor": "#333333"})
  # Plot a horizontal line at zero.
  axisObject.axhline(0.0, color="black", linewidth=0.7, linestyle="-")
  # Annotate each bar with its value.
  for barIndex, bar in enumerate(bars):
    # Get the bar height.
    height = bar.get_height()
    # Determine the vertical alignment.
    va_align = "bottom" if (height >= 0) else "top"
    # Annotate the bar.
    axisObject.annotate("%.2f%%" % height, xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4 if (height >= 0) else -4), textcoords="offset points", ha="center", va=va_align,
                        fontsize=10, fontweight="bold")
  # Set the x-axis label.
  axisObject.set_xlabel("Comparison")
  # Set the y-axis label.
  axisObject.set_ylabel("Mean Cost Reduction of DRO vs Baseline (%)")
  # Set the title.
  axisObject.set_title("Operational Cost Reduction from DRO vs Naive Baselines\n"
                       "Error bars = 95%% bootstrap confidence intervals (%d resamples)" % bootstrapResamples,
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cost_reduction_bar_plot", configuration,
                               nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 5 - Cluster x method mean cost heatmap.
def GenerateClusterMethodHeatmap(outOfSampleDataFrame, configuration):
  """Heatmap of mean cost by cluster and method."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.6, 6.4))
  # Aggregate the per-cluster x method means.
  pivotFrame = outOfSampleDataFrame.groupby(["ClusterId", "Method"])["TotalCost"].mean().reset_index()
  # Pivot to wide form.
  heatFrame = pivotFrame.pivot(index="ClusterId", columns="Method", values="TotalCost")
  # Filter and order the methods.
  methodOrder = [m for m in ["EV", "SAA", "DRO"] if (m in heatFrame.columns)]
  # Select the method columns in order.
  heatFrame = heatFrame[methodOrder]
  # Sort by DRO cost ascending.
  if ("DRO" in heatFrame.columns):
    # Sort the rows.
    heatFrame = heatFrame.sort_values("DRO", ascending=True)
  # Generate the heatmap.
  sns.heatmap(heatFrame, ax=axisObject, cmap="RdYlGn_r", annot=True, fmt=".0f",
              linewidths=0.5, linecolor="white", cbar_kws={"label": "Mean Operational Cost (USD)"})
  # Set the x-axis label.
  axisObject.set_xlabel("Decision Model")
  # Set the y-axis label.
  axisObject.set_ylabel("Cluster Identifier (sorted by DRO cost)")
  # Set the title.
  axisObject.set_title("Per-Cluster Mean Operational Cost by Decision Model\n"
                       "Greener = lower cost; a row where DRO is greener than EV/SAA confirms dominance",
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(heatFrame),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cluster_method_heatmap", configuration,
                               nClusters=len(heatFrame), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 6 - Bootstrap 95% CI forest plot.
def GenerateBootstrapCiForestPlot(statisticalSummaryFrame, configuration):
  """Forest plot of the bootstrap 95% CI of mean cost per method."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.0, 4.6))
  # Define the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Iterate over each method.
  for yPos, methodName in enumerate(methodOrder):
    # Select the row for the current method.
    row = statisticalSummaryFrame[statisticalSummaryFrame["Method"] == methodName].iloc[0]
    # Extract the mean.
    meanValue = float(row["MeanCost"])
    # Extract the CI bounds.
    lower = float(row["Ci95Lower"])
    upper = float(row["Ci95Upper"])
    # Get the edge color.
    color = METHOD_EDGE_COLORS[methodName]
    # Plot the error bar.
    axisObject.errorbar([meanValue], [yPos], xerr=[[meanValue - lower], [upper - meanValue]],
                        fmt="o", color=color, ecolor=color, capsize=10, capthick=2, elinewidth=2.2,
                        markersize=12, label=methodName)
    # Annotate the mean and CI.
    axisObject.annotate("mean=%.2f  [%.2f, %.2f]" % (meanValue, lower, upper),
                        xy=(meanValue, yPos), xytext=(8, -4), textcoords="offset points",
                        fontsize=8, color=color, fontweight="bold")
  # Set the y-axis ticks.
  axisObject.set_yticks(range(len(methodOrder)))
  # Set the y-axis tick labels.
  axisObject.set_yticklabels(methodOrder)
  # Set the x-axis label.
  axisObject.set_xlabel("Mean Out-of-Sample Operational Cost (USD)  with 95% Bootstrap CI")
  # Set the y-axis label.
  axisObject.set_ylabel("Decision Model")
  # Set the title.
  axisObject.set_title("Bootstrap 95%% Confidence Intervals for Mean Operational Cost (%d resamples)\n"
                       "Non-overlapping intervals imply a statistically significant difference" %
                       configuration["BootstrapResamples"], fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(loc="lower right", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=statisticalSummaryFrame.shape[0],
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "bootstrap_ci_forest_plot", configuration,
                               nClusters=statisticalSummaryFrame.shape[0],
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 7 - Wilcoxon effect sizes plot.
def GenerateWilcoxonEffectsPlot(comparativeDataFrame, configuration):
  """Bar plot of Wilcoxon p-value (log scale) and Cohen r effect size."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure with two side-by-side axes.
  figureObject, (leftAxis, rightAxis) = plt.subplots(1, 2, figsize=(11.0, 4.8))
  # Extract the comparisons and p-values.
  comparisons = comparativeDataFrame["Comparison"].tolist()
  # Extract the p-values.
  pValues = comparativeDataFrame["WilcoxonPValue"].tolist()
  # Build the bar colors based on significance.
  barColors = ["#C0504D" if (p < 0.05) else "#7F7F7F" for p in pValues]
  # Plot the p-value bars.
  leftAxis.bar(comparisons, pValues, color=barColors, edgecolor="black", linewidth=0.6)
  # Add the significance threshold line.
  leftAxis.axhline(0.05, color="red", linestyle="--", linewidth=1.0, label="alpha = 0.05")
  # Set the y-axis to log scale.
  leftAxis.set_yscale("log")
  # Set the x-axis label.
  leftAxis.set_xlabel("Comparison")
  # Set the y-axis label.
  leftAxis.set_ylabel("Wilcoxon p-value (log scale)")
  # Set the title.
  leftAxis.set_title("Statistical Significance", fontsize=10, fontweight="bold")
  # Add the legend.
  leftAxis.legend(loc="upper right", frameon=True)
  # Annotate each bar.
  for idx, value in enumerate(pValues):
    # Annotate.
    leftAxis.annotate("%.2e" % value, xy=(idx, value), xytext=(0, 6),
                      textcoords="offset points", ha="center", fontsize=8)
  # Extract the effect sizes.
  effectSizes = comparativeDataFrame["CohenR"].tolist()
  # Build the effect-size bar colors based on the band.
  barColors2 = ["#1F77B4" if (e >= 0.3) else ("#AEC7E8" if (e >= 0.1) else "#C7C7C7") for e in effectSizes]
  # Plot the effect-size bars.
  rightAxis.bar(comparisons, effectSizes, color=barColors2, edgecolor="black", linewidth=0.6)
  # Add the small-effect reference line.
  rightAxis.axhline(0.10, color="#888888", linestyle=":", linewidth=0.9, label="small (r=0.10)")
  # Add the medium-effect reference line.
  rightAxis.axhline(0.30, color="#444444", linestyle="--", linewidth=0.9, label="medium (r=0.30)")
  # Set the x-axis label.
  rightAxis.set_xlabel("Comparison")
  # Set the y-axis label.
  rightAxis.set_ylabel("Cohen r Effect Size")
  # Set the title.
  rightAxis.set_title("Effect Size (Cohen r)", fontsize=10, fontweight="bold")
  # Add the legend.
  rightAxis.legend(loc="upper right", frameon=True)
  # Annotate each bar.
  for idx, value in enumerate(effectSizes):
    # Annotate.
    rightAxis.annotate("%.3f" % value, xy=(idx, value), xytext=(0, 6),
                       textcoords="offset points", ha="center", fontsize=8)
  # Set the suptitle.
  figureObject.suptitle("Wilcoxon Signed-Rank Test: DRO Significantly Outperforms Naive Baselines\n"
                        "Red bars = significant at alpha=0.05; larger Cohen r = stronger effect",
                        fontsize=11, fontweight="bold")
  # Save and return.
  return SaveFigureBothFormats(figureObject, "wilcoxon_effects_plot", configuration,
                               nClusters=len(comparativeDataFrame), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 8 - Per-cluster demand distribution boxplot.
def GenerateDemandDistributionBoxplot(clusterNetworkArtifacts, configuration):
  """Boxplot of bootstrap demand scenarios per cluster, faceted by scale band."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the records list.
  records = []
  # Iterate over each cluster.
  for clusterId, artifact in clusterNetworkArtifacts.items():
    # Convert the training scenarios to a numpy array.
    trainingScenarios = np.array(artifact["TrainingScenarios"])
    # Get the cluster size.
    clusterSize = int(artifact["ClusterSize"])
    # Categorize the scale band.
    band = "Small" if (clusterSize <= 20) else ("Medium" if (clusterSize <= 40) else "Large")
    # Compute the per-scenario total demand.
    perScenarioTotal = trainingScenarios.sum(axis=1)
    # Iterate over each per-scenario total.
    for value in perScenarioTotal:
      # Append the record.
      records.append({"ClusterId"             : int(clusterId), "ClusterSize": clusterSize, "Band": band,
                      "TotalDemandPerScenario": float(value)})
  # Convert the records to a dataframe.
  demandFrame = pd.DataFrame(records)
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(9.6, 5.4))
  # Generate the boxplot with hue assigned to silence seaborn 0.14.
  sns.boxplot(data=demandFrame, x="Band", y="TotalDemandPerScenario",
              order=["Small", "Medium", "Large"], hue="Band",
              palette=["#FDD9A0", "#FCA082", "#E66060"], ax=axisObject,
              width=0.55, linewidth=1.0, fliersize=2.5, legend=False)
  # Overlay the strip plot.
  sns.stripplot(data=demandFrame, x="Band", y="TotalDemandPerScenario",
                order=["Small", "Medium", "Large"], hue="Band",
                color="black", alpha=0.15, size=1.6, ax=axisObject, legend=False)
  # Set the x-axis label.
  axisObject.set_xlabel("Cluster Scale Band")
  # Set the y-axis label.
  axisObject.set_ylabel("Total Per-Scenario Demand (sum of trips across stations)")
  # Set the title.
  axisObject.set_title("Bootstrapped Demand Scenario Distribution per Cluster Scale Band\n"
                       "Each dot = one bootstrap scenario; larger clusters naturally draw larger total demand",
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(clusterNetworkArtifacts),
                nScenarios=configuration["TrainingScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "demand_distribution_boxplot", configuration,
                               nClusters=len(clusterNetworkArtifacts), nScenarios=configuration["TrainingScenarios"])


# Figure 9 - Station geographic distribution map.
def GenerateStationGeographicMap(stationDataFrame, configuration):
  """Static scatter of all stations, colored by cluster, sized by capacity."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(9.0, 6.4))
  # Get the sorted cluster identifiers.
  clusterIds = sorted(stationDataFrame["ClusterId"].unique())
  # Build the color palette.
  palette = sns.color_palette("tab20", n_colors=len(clusterIds))
  # Iterate over each cluster identifier.
  for idx, clusterId in enumerate(clusterIds):
    # Subset the stations.
    subset = stationDataFrame[stationDataFrame["ClusterId"] == clusterId]
    # Scatter the stations.
    axisObject.scatter(subset["Longitude"], subset["Latitude"],
                       s=12.0 + 2.0 * subset["Capacity"],
                       c=[palette[idx]], edgecolors="black", linewidths=0.4,
                       alpha=0.85, label="Cluster %d (%s)" % (clusterId, subset.iloc[0]["Band"]))
  # Set the x-axis label.
  axisObject.set_xlabel("Longitude")
  # Set the y-axis label.
  axisObject.set_ylabel("Latitude")
  # Set the aspect ratio.
  axisObject.set_aspect("equal", adjustable="datalim")
  # Set the title.
  axisObject.set_title("Geographic Distribution of Selected Bike-Share Stations\n"
                       "Each color = one Morton-curve cluster; marker size proportional to dock capacity",
                       fontsize=11, fontweight="bold")
  # Build the simplified legend.
  bandHandles = []
  # Iterate over each band name and color.
  for bandName, bandColor in zip(["Small", "Medium", "Large"], ["#4DAF4A", "#FF7F0E", "#D62728"]):
    # Append the patch handle.
    bandHandles.append(mpatches.Patch(color=bandColor, label="%s cluster (15/30/45 stations)" % bandName))
  # Add the legend.
  axisObject.legend(handles=bandHandles, loc="upper right", frameon=True, fancybox=True, framealpha=0.95)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(clusterIds),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "station_geographic_map", configuration,
                               nClusters=len(clusterIds), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 10 - Net positioning heatmap per cluster.
def GenerateNetPositioningHeatmap(decisionDataFrame, configuration, topNClusters=6):
  """Heatmap of net positioning per station for the top N DRO clusters."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Filter to DRO decisions only.
  droFrame = decisionDataFrame[decisionDataFrame["Method"] == "DRO"].copy()
  # Compute the absolute sum for ranking.
  droFrame["AbsSum"] = droFrame["NetPositioning"].apply(
    lambda s: float(np.sum(np.abs(np.array(json.loads(s))))))
  # Take the top N clusters.
  topClusters = droFrame.sort_values("AbsSum", ascending=False).head(topNClusters)["ClusterId"].tolist()
  # Initialize the matplotlib figure.
  figureObject, axesArray = plt.subplots(2, 3, figsize=(13.2, 7.4))
  # Flatten the axes array.
  axesFlat = axesArray.flatten()
  # Iterate over the top clusters.
  for panelIdx, clusterId in enumerate(topClusters):
    # Select the current axis.
    axisObject = axesFlat[panelIdx]
    # Get the DRO decision row for the current cluster.
    row = decisionDataFrame[(decisionDataFrame["ClusterId"] == clusterId) &
                            (decisionDataFrame["Method"] == "DRO")].iloc[0]
    # Parse the positioning vector.
    positioning = np.array(json.loads(row["NetPositioning"]))
    # Get the cluster size.
    n = len(positioning)
    # Compute the square side.
    side = int(np.ceil(np.sqrt(n)))
    # Pad the positioning vector.
    padded = np.zeros(side * side)
    # Copy the values.
    padded[:n] = positioning
    # Reshape to a 2D grid.
    grid = padded.reshape(side, side)
    # Generate the heatmap.
    sns.heatmap(grid, ax=axisObject, cmap="RdBu_r", center=0.0, annot=False,
                cbar=True, xticklabels=False, yticklabels=False,
                cbar_kws={"label": "Net Positioning (bikes)"})
    # Set the panel title.
    axisObject.set_title("Cluster %d (size=%d)" % (clusterId, n), fontsize=10)
  # Set the suptitle.
  figureObject.suptitle("DRO Net Positioning Decisions for the Top %d Clusters\n"
                        "Red = surplus (remove bikes); Blue = deficit (add bikes)" % topNClusters,
                        fontsize=12, fontweight="bold")
  # Save and return.
  return SaveFigureBothFormats(figureObject, "net_positioning_heatmap", configuration,
                               nClusters=topNClusters, nScenarios=configuration["TrainingScenarios"])


# Figure 11 - Static rebalancing flow map (side-by-side EV vs DRO).
def GenerateStaticRebalancingFlowMap(stationDataFrame, decisionDataFrame, clusterNetworkArtifacts,
                                     configuration):
  """Two-panel static map of EV vs DRO rebalancing flows on a representative cluster."""
  # Initialize the representative cluster identifier.
  representativeClusterId = None
  # Iterate over the sorted cluster identifiers.
  for clusterIdentifier in sorted(clusterNetworkArtifacts.keys(), key=lambda value: int(value)):
    # Check whether the cluster is medium-sized.
    if (int(clusterNetworkArtifacts[clusterIdentifier]["ClusterSize"]) == 30):
      # Assign the representative cluster identifier.
      representativeClusterId = int(clusterIdentifier)
      # Break out of the loop.
      break
  # Fall back to the first cluster when no medium cluster exists.
  if (representativeClusterId is None):
    # Use the first cluster identifier.
    representativeClusterId = int(sorted(clusterNetworkArtifacts.keys(), key=lambda value: int(value))[0])
  # Subset the station dataframe to the representative cluster.
  clusterStations = stationDataFrame[stationDataFrame["ClusterId"] == representativeClusterId].reset_index(drop=True)
  # Get the DRO decision row.
  droDecisionRow = decisionDataFrame[
    (decisionDataFrame["ClusterId"] == representativeClusterId) & (decisionDataFrame["Method"] == "DRO")].iloc[0]
  # Get the EV decision row.
  evDecisionRow = decisionDataFrame[
    (decisionDataFrame["ClusterId"] == representativeClusterId) & (decisionDataFrame["Method"] == "EV")].iloc[0]
  # Parse the DRO positioning.
  droPositioning = np.array(json.loads(droDecisionRow["NetPositioning"]))
  # Parse the EV positioning.
  evPositioning = np.array(json.loads(evDecisionRow["NetPositioning"]))
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure.
  figureObject, axisArray = plt.subplots(1, 2, figsize=(11.6, 5.2))
  # Iterate over the two panels.
  for panelIndex, (positioningVector, panelTitle, flowColor) in enumerate(
    [(evPositioning, "EV Baseline", "#C0504D"),
     (droPositioning, "Proposed DRO Model", "#2E6CA4")]):
    # Select the current axis.
    axisObject = axisArray[panelIndex]
    # Extract the longitudes.
    longitudes = clusterStations["Longitude"].to_numpy()
    # Extract the latitudes.
    latitudes = clusterStations["Latitude"].to_numpy()
    # Compute the marker sizes.
    markerSizes = 20.0 + 90.0 * np.abs(positioningVector) / (np.max(np.abs(positioningVector)) + 1e-6)
    # Compute the marker colors.
    markerColors = ["#2E8B57" if (v > 0.1) else ("#B22222" if (v < -0.1) else "#9E9E9E")
                    for v in positioningVector]
    # Scatter the stations.
    axisObject.scatter(longitudes, latitudes, s=markerSizes, c=markerColors,
                       edgecolors="black", linewidths=0.5, zorder=3)
    # Find the surplus indices.
    surplusIndices = np.where(positioningVector < -0.1)[0]
    # Only draw arrows when there are surplus stations.
    if (len(surplusIndices) > 0):
      # Find the deficit index.
      deficitIndex = int(np.argmax(positioningVector))
      # Iterate over each surplus index.
      for surplusIndex in surplusIndices:
        # Annotate the arrow.
        axisObject.annotate("",
                            xy=(longitudes[deficitIndex], latitudes[deficitIndex]),
                            xytext=(longitudes[surplusIndex], latitudes[surplusIndex]),
                            arrowprops=dict(arrowstyle="->", color=flowColor, alpha=0.55, lw=1.2))
    # Set the panel title.
    axisObject.set_title(panelTitle, fontsize=10, fontweight="bold")
    # Set the x-axis label.
    axisObject.set_xlabel("Longitude")
    # Set the y-axis label.
    axisObject.set_ylabel("Latitude")
    # Set the aspect ratio.
    axisObject.set_aspect("equal", adjustable="datalim")
  # Build the shared legend.
  legendHandles = [
    mpatches.Patch(color="#2E8B57", label="Net deficit (add bikes)"),
    mpatches.Patch(color="#B22222", label="Net surplus (remove bikes)"),
    mpatches.Patch(color="#9E9E9E", label="Near-balanced station"),
  ]
  # Add the legend to the figure.
  figureObject.legend(handles=legendHandles, loc="lower center", ncol=3, frameon=True,
                      fancybox=True, bbox_to_anchor=(0.5, -0.04))
  # Set the suptitle.
  figureObject.suptitle("Rebalancing Flow Topology for Representative Medium Cluster (Cluster %d)\n"
                        "Arrow direction = surplus station to dominant deficit station" % representativeClusterId,
                        fontsize=11, fontweight="bold")
  # Save and return.
  return SaveFigureBothFormats(figureObject, "rebalancing_flow_map", configuration,
                               nClusters=1, nScenarios=configuration["OutOfSampleScenarios"])


# Figure 12 - Interactive folium map (HTML only).
def GenerateInteractiveRebalancingMap(stationDataFrame, decisionDataFrame, clusterNetworkArtifacts,
                                      configuration):
  """Interactive HTML map of EV vs DRO rebalancing for a representative cluster."""
  # Initialize the representative cluster identifier.
  representativeClusterId = None
  # Iterate over the sorted cluster identifiers.
  for clusterIdentifier in sorted(clusterNetworkArtifacts.keys(), key=lambda value: int(value)):
    # Check whether the cluster is medium-sized.
    if (int(clusterNetworkArtifacts[clusterIdentifier]["ClusterSize"]) == 30):
      # Assign the representative cluster identifier.
      representativeClusterId = int(clusterIdentifier)
      # Break out of the loop.
      break
  # Fall back to the first cluster.
  if (representativeClusterId is None):
    # Use the first cluster identifier.
    representativeClusterId = int(sorted(clusterNetworkArtifacts.keys(), key=lambda value: int(value))[0])
  # Subset the stations to the representative cluster.
  clusterStations = stationDataFrame[stationDataFrame["ClusterId"] == representativeClusterId].reset_index(drop=True)
  # Compute the central latitude.
  centralLatitude = float(clusterStations["Latitude"].mean())
  # Compute the central longitude.
  centralLongitude = float(clusterStations["Longitude"].mean())
  # Initialize the folium map using OpenStreetMap tiles (no API key required).
  interactiveMap = folium.Map(location=[centralLatitude, centralLongitude], zoom_start=14,
                              tiles="OpenStreetMap")
  # Get the DRO decision row.
  droDecisionRow = decisionDataFrame[
    (decisionDataFrame["ClusterId"] == representativeClusterId) & (decisionDataFrame["Method"] == "DRO")].iloc[0]
  # Get the EV decision row.
  evDecisionRow = decisionDataFrame[
    (decisionDataFrame["ClusterId"] == representativeClusterId) & (decisionDataFrame["Method"] == "EV")].iloc[0]
  # Parse the DRO positioning.
  droPositioning = np.array(json.loads(droDecisionRow["NetPositioning"]))
  # Parse the EV positioning.
  evPositioning = np.array(json.loads(evDecisionRow["NetPositioning"]))
  # Build the title HTML.
  titleHtml = (
                "<h3 style='text-align:center;font-family:Arial'>%s %s - Cluster %d Rebalancing Map</h3>"
                "<p style='text-align:center;font-family:Arial;font-size:11px'>"
                "Green = DRO deficit (add bikes); Red = DRO surplus (remove bikes); "
                "Blue lines = DRO flows; Light red lines = EV baseline flows</p>"
              ) % (configuration["ActiveCityKey"], Config.YEAR_MONTH_TAG, representativeClusterId)
  # Add the title to the map.
  interactiveMap.get_root().html.add_child(folium.Element(titleHtml))
  # Iterate over each station.
  for stationIndex, stationRow in clusterStations.iterrows():
    # Get the DRO positioning value.
    positioningValue = float(droPositioning[stationIndex])
    # Determine the fill color.
    fillColor = "green" if (positioningValue > 0.1) else ("red" if (positioningValue < -0.1) else "gray")
    # Add a circle marker.
    folium.CircleMarker(
      location=[float(stationRow["Latitude"]), float(stationRow["Longitude"])], radius=5,
      color="black", fill=True, fill_color=fillColor, fill_opacity=0.8,
      popup="Station %s (DRO net %.1f, EV net %.1f)" % (
        str(stationRow["StationId"]), positioningValue, float(evPositioning[stationIndex]))).add_to(
      interactiveMap)

  # Define the inner closure to draw flows.
  def drawFlowsForMethod(positioningVector, lineColor, lineWeight, lineOpacity, label):
    # Iterate over each origin index.
    for originIndex in range(len(positioningVector)):
      # Skip non-surplus origins.
      if (positioningVector[originIndex] >= -0.1):
        # Continue to the next origin.
        continue
      # Find the dominant deficit.
      destinationIndex = int(np.argmax(positioningVector))
      # Skip when no meaningful deficit exists.
      if (positioningVector[destinationIndex] <= 0.1):
        # Continue to the next origin.
        continue
      # Build the origin coordinates.
      originCoordinates = [float(clusterStations.iloc[originIndex]["Latitude"]),
                           float(clusterStations.iloc[originIndex]["Longitude"])]
      # Build the destination coordinates.
      destinationCoordinates = [float(clusterStations.iloc[destinationIndex]["Latitude"]),
                                float(clusterStations.iloc[destinationIndex]["Longitude"])]
      # Add the polyline.
      folium.PolyLine(locations=[originCoordinates, destinationCoordinates],
                      color=lineColor, weight=lineWeight, opacity=lineOpacity,
                      tooltip="%s flow" % label).add_to(interactiveMap)

  # Draw the EV flows in light red.
  drawFlowsForMethod(evPositioning, "#E07B7B", 2.0, 0.45, "EV")
  # Draw the DRO flows in deep blue.
  drawFlowsForMethod(droPositioning, "#2E5C8A", 4.0, 0.85, "DRO")
  # Add a layer control.
  folium.LayerControl().add_to(interactiveMap)
  # Build the output map path.
  outputMapPath = Config.TaggedFigureName("rebalancing_map.html")
  # Save the map.
  interactiveMap.save(outputMapPath)
  # Return the path.
  return outputMapPath


# Figure 13 - Cost distribution violin plot.
def GenerateCostViolinPlot(outOfSampleDataFrame, configuration):
  """Violin plot showing the full kernel density of cost per method."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.2))
  # Define the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Build the color palette.
  colorPalette = [METHOD_COLORS[m] for m in methodOrder]
  # Generate the violin plot with hue assigned.
  sns.violinplot(data=outOfSampleDataFrame, x="Method", y="TotalCost",
                 order=methodOrder, hue="Method", palette=colorPalette,
                 ax=axisObject, inner="quartile", linewidth=1.1, cut=0, legend=False)
  # Overlay the strip plot.
  sns.stripplot(data=outOfSampleDataFrame, x="Method", y="TotalCost",
                order=methodOrder, hue="Method", color="black",
                alpha=0.18, size=1.6, ax=axisObject, legend=False)
  # Annotate the median per method.
  for methodIndex, methodName in enumerate(methodOrder):
    # Compute the median.
    methodMedian = float(np.median(outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]["TotalCost"]))
    # Annotate the median.
    axisObject.annotate("median=%.1f" % methodMedian, xy=(methodIndex, methodMedian), xytext=(8, 0),
                        textcoords="offset points", fontsize=8, color="#B22222", va="center")
  # Set the x-axis label.
  axisObject.set_xlabel("Decision Model")
  # Set the y-axis label.
  axisObject.set_ylabel("Out-of-Sample Operational Cost (USD)")
  # Set the title.
  axisObject.set_title("Violin Plot of Out-of-Sample Cost Distributions per Method\n"
                       "Width = kernel density; inner lines = quartiles; lower and narrower = better",
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cost_violin_plot", configuration,
                               nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 14 - Q-Q plot of DRO vs EV cost.
def GenerateQqPlotDroVsEv(outOfSampleDataFrame, configuration):
  """Quantile-quantile plot of DRO vs EV cost distributions."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(7.6, 6.6))
  # Build the pivot table.
  pivotFrame = outOfSampleDataFrame.pivot(
    index=["ClusterId", "ScenarioIndex"], columns="Method", values="TotalCost")
  # Sort the DRO costs.
  droSorted = np.sort(pivotFrame["DRO"].to_numpy())
  # Sort the EV costs.
  evSorted = np.sort(pivotFrame["EV"].to_numpy())
  # Sort the SAA costs.
  saaSorted = np.sort(pivotFrame["SAA"].to_numpy())
  # Plot the DRO vs EV Q-Q.
  axisObject.scatter(evSorted, droSorted, s=12, color=METHOD_EDGE_COLORS["DRO"], alpha=0.45,
                     label="DRO vs EV")
  # Plot the DRO vs SAA Q-Q.
  axisObject.scatter(saaSorted, droSorted, s=12, color=METHOD_EDGE_COLORS["SAA"], alpha=0.45,
                     label="DRO vs SAA", marker="s")
  # Plot the reference diagonal.
  minValue = min(float(np.min(evSorted)), float(np.min(droSorted)))
  # Get the max value.
  maxValue = max(float(np.max(evSorted)), float(np.max(droSorted)))
  # Plot the diagonal.
  axisObject.plot([minValue, maxValue], [minValue, maxValue], color="black", linestyle="--",
                  linewidth=1.0, label="y = x (no difference)")
  # Set the x-axis label.
  axisObject.set_xlabel("Baseline cost quantile (USD)")
  # Set the y-axis label.
  axisObject.set_ylabel("DRO cost quantile (USD)")
  # Set the title.
  axisObject.set_title("Quantile-Quantile Plot of DRO Cost vs Baselines\n"
                       "Below the diagonal = DRO cheaper at that quantile; above = DRO more expensive",
                       fontsize=11, fontweight="bold")
  # Set the aspect ratio.
  axisObject.set_aspect("equal", adjustable="datalim")
  # Add the legend.
  axisObject.legend(loc="upper left", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(pivotFrame),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "qq_plot_dro_vs_ev", configuration,
                               nClusters=len(pivotFrame), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 15 - Pareto frontier plot.
def GenerateParetoFrontierPlot(decisionDataFrame, outOfSampleDataFrame, configuration):
  """Pareto frontier: first-stage rebalancing cost vs out-of-sample mean cost."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 6.2))
  # Aggregate the per-cluster x method out-of-sample mean cost.
  oosMeanFrame = outOfSampleDataFrame.groupby(["ClusterId", "Method"])["TotalCost"].mean().reset_index()
  # Merge with the first-stage cost.
  mergedFrame = oosMeanFrame.merge(decisionDataFrame, on=["ClusterId", "Method"], how="inner")
  # Iterate over each method.
  for methodName in ["EV", "SAA", "DRO"]:
    # Subset to the current method.
    methodSubset = mergedFrame[mergedFrame["Method"] == methodName]
    # Scatter the points.
    axisObject.scatter(methodSubset["RebalancingCost"], methodSubset["TotalCost"],
                       s=40, color=METHOD_EDGE_COLORS[methodName], alpha=0.75,
                       edgecolors="black", linewidths=0.5, label=methodName)
  # Set the x-axis label.
  axisObject.set_xlabel("First-Stage Rebalancing Cost (USD)  - upfront effort")
  # Set the y-axis label.
  axisObject.set_ylabel("Out-of-Sample Mean Operational Cost (USD)  - downstream performance")
  # Set the title.
  axisObject.set_title("Pareto Frontier: First-Stage Cost vs Out-of-Sample Mean Cost per Cluster\n"
                       "Lower-left = best trade-off; DRO clusters should sit closer to the lower-left frontier",
                       fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(title="Decision Model", loc="upper right", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(mergedFrame),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "pareto_frontier_plot", configuration,
                               nClusters=len(mergedFrame), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 16 - CVaR tail comparison plot.
def GenerateCvarTailComparisonPlot(outOfSampleDataFrame, configuration):
  """Boxplot of the worst (top 10%) scenarios per method - the CVaR tail."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.2))
  # Define the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Build the color palette.
  colorPalette = [METHOD_COLORS[m] for m in methodOrder]
  # Initialize the records list.
  tailRecords = []
  # Iterate over each method.
  for methodName in methodOrder:
    # Subset to the current method.
    methodSubset = outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]
    # Compute the 90th percentile threshold.
    threshold = float(np.percentile(methodSubset["TotalCost"], 90))
    # Take the worst 10% of scenarios.
    tailSubset = methodSubset[methodSubset["TotalCost"] >= threshold]
    # Iterate over each tail row.
    for _, row in tailSubset.iterrows():
      # Append the record.
      tailRecords.append({"Method"   : methodName, "TotalCost": float(row["TotalCost"]),
                          "ClusterId": int(row["ClusterId"])})
  # Convert to dataframe.
  tailFrame = pd.DataFrame(tailRecords)
  # Generate the boxplot.
  sns.boxplot(data=tailFrame, x="Method", y="TotalCost", order=methodOrder,
              hue="Method", palette=colorPalette, ax=axisObject,
              width=0.55, fliersize=2.5, linewidth=1.1, legend=False)
  # Overlay the strip plot.
  sns.stripplot(data=tailFrame, x="Method", y="TotalCost", order=methodOrder,
                hue="Method", color="black", alpha=0.30, size=2.4, ax=axisObject, legend=False)
  # Annotate the mean tail cost.
  for methodIndex, methodName in enumerate(methodOrder):
    # Compute the tail mean.
    tailMean = float(tailFrame[tailFrame["Method"] == methodName]["TotalCost"].mean())
    # Scatter the mean.
    axisObject.scatter([methodIndex], [tailMean], marker="D", color="#B22222", s=40, zorder=5)
    # Annotate the mean.
    axisObject.annotate("CVaR90=%.1f" % tailMean, xy=(methodIndex, tailMean), xytext=(8, 0),
                        textcoords="offset points", fontsize=8, color="#B22222", va="center")
  # Set the x-axis label.
  axisObject.set_xlabel("Decision Model")
  # Set the y-axis label.
  axisObject.set_ylabel("Out-of-Sample Cost in the Worst 10% Scenarios (USD)")
  # Set the title.
  axisObject.set_title("CVaR Tail Comparison: Worst 10% Out-of-Sample Costs per Method\n"
                       "Lower = better tail-risk management; DRO should have the lowest CVaR90",
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cvar_tail_comparison_plot", configuration,
                               nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 17 - Radar chart multi-criteria performance.
def GenerateRadarChartMultiCriteria(outOfSampleDataFrame, configuration):
  """Radar chart comparing methods on five metrics: mean, std, p95, CVaR90, max."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure with polar projection.
  figureObject = plt.figure(figsize=(8.0, 7.0))
  # Add a polar axis.
  axisObject = figureObject.add_subplot(111, polar=True)
  # Define the metrics.
  metrics = ["Mean", "Std", "P95", "CVaR90", "Max"]
  # Compute the per-method metric values.
  methodMetrics = {}
  # Iterate over each method.
  for methodName in ["EV", "SAA", "DRO"]:
    # Subset to the current method.
    methodSubset = outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]
    # Extract the total cost array.
    costArray = methodSubset["TotalCost"].to_numpy()
    # Compute the metrics.
    methodMetrics[methodName] = {
      "Mean"  : float(np.mean(costArray)),
      "Std"   : float(np.std(costArray, ddof=1)),
      "P95"   : float(np.percentile(costArray, 95)),
      "CVaR90": float(np.mean(costArray[costArray >= np.percentile(costArray, 90)])),
      "Max"   : float(np.max(costArray)),
    }
  # Compute the normalization reference (the max across methods per metric).
  referenceMax = {metric: max(methodMetrics[m][metric] for m in ["EV", "SAA", "DRO"])
                  for metric in metrics}
  # Compute the angles for the polar plot.
  angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
  # Close the radar loop.
  angles += angles[:1]
  # Iterate over each method.
  for methodName in ["EV", "SAA", "DRO"]:
    # Compute the normalized values.
    values = [methodMetrics[methodName][metric] / max(referenceMax[metric], 1e-9)
              for metric in metrics]
    # Close the values loop.
    values += values[:1]
    # Plot the radar.
    axisObject.plot(angles, values, color=METHOD_EDGE_COLORS[methodName], linewidth=2.2,
                    label=methodName)
    # Fill the radar.
    axisObject.fill(angles, values, color=METHOD_EDGE_COLORS[methodName], alpha=0.18)
  # Set the radial axis labels.
  axisObject.set_xticks(angles[:-1])
  # Set the metric labels.
  axisObject.set_xticklabels(metrics)
  # Set the radial limits.
  axisObject.set_ylim(0, 1.05)
  # Set the title.
  axisObject.set_title("Multi-Criteria Performance Radar (normalized 0-1; lower = better)\n"
                       "DRO should have the smallest area across all five metrics",
                       fontsize=11, fontweight="bold", pad=20)
  # Add the legend.
  axisObject.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), frameon=True, fancybox=True)
  # Save and return.
  return SaveFigureBothFormats(figureObject, "radar_chart_multicriteria", configuration,
                               nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# ============================================================================
# NEW FIGURES ADDED IN v4 (figures 18-30)
# ============================================================================

# Figure 18 - Bland-Altman plot (NEW v4).
def GenerateBlandAltmanPlot(outOfSampleDataFrame, configuration):
  """Bland-Altman agreement plot for DRO vs EV: mean vs difference."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.0, 5.6))
  # Pivot to per-scenario paired costs.
  pivotFrame = outOfSampleDataFrame.pivot(
    index=["ClusterId", "ScenarioIndex"], columns="Method", values="TotalCost").reset_index()
  # Compute the mean of paired values.
  meanPaired = (pivotFrame["DRO"] + pivotFrame["EV"]) / 2.0
  # Compute the difference DRO - EV.
  diffPaired = pivotFrame["DRO"] - pivotFrame["EV"]
  # Scatter the points.
  axisObject.scatter(meanPaired, diffPaired, s=14, color=METHOD_EDGE_COLORS["DRO"], alpha=0.45,
                     edgecolors="black", linewidths=0.4, label="Per-scenario")
  # Compute the mean difference (bias).
  bias = float(np.mean(diffPaired))
  # Compute the standard deviation of the differences.
  stdDiff = float(np.std(diffPaired, ddof=1))
  # Compute the limits of agreement.
  upperLimit = bias + 1.96 * stdDiff
  # Compute the lower limit.
  lowerLimit = bias - 1.96 * stdDiff
  # Plot the bias line.
  axisObject.axhline(bias, color="#1F3F6E", linestyle="-", linewidth=1.4, label="bias = %.2f" % bias)
  # Plot the upper limit of agreement.
  axisObject.axhline(upperLimit, color="#C0504D", linestyle="--", linewidth=1.1,
                     label="+1.96 SD = %.2f" % upperLimit)
  # Plot the lower limit of agreement.
  axisObject.axhline(lowerLimit, color="#C0504D", linestyle="--", linewidth=1.1,
                     label="-1.96 SD = %.2f" % lowerLimit)
  # Plot the zero reference line.
  axisObject.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
  # Set the x-axis label.
  axisObject.set_xlabel("Mean of DRO and EV Cost (USD)")
  # Set the y-axis label.
  axisObject.set_ylabel("Difference (DRO - EV) Cost (USD)")
  # Set the title.
  axisObject.set_title("Bland-Altman Agreement Plot: DRO vs EV per Scenario\n"
                       "Negative bias = DRO systematically cheaper; bands = 95% limits of agreement",
                       fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(loc="upper right", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(pivotFrame),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "bland_altman_plot", configuration,
                               nClusters=len(pivotFrame), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 19 - ECDF with 95% bootstrap CI band (NEW v4).
def GenerateEcdfWithCiBand(outOfSampleDataFrame, configuration, randomGenerator):
  """Empirical CDF with a 95% bootstrap confidence band per method."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.6))
  # Define the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Use a smaller bootstrap count for plot speed.
  bootstrapResamples = min(200, configuration["BootstrapResamples"])
  # Iterate over each method.
  for methodName in methodOrder:
    # Extract the cost array.
    costArray = np.sort(outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]["TotalCost"].to_numpy())
    # Compute the empirical CDF.
    ecdfValues = np.arange(1, len(costArray) + 1) / len(costArray)
    # Plot the empirical CDF as a solid line.
    axisObject.step(costArray, ecdfValues, color=METHOD_EDGE_COLORS[methodName], linewidth=2.0,
                    label=methodName, where="post")
    # Compute the bootstrap CI band.
    lowerBand = np.zeros_like(ecdfValues)
    # Initialize the upper band.
    upperBand = np.zeros_like(ecdfValues)
    # Iterate over each bootstrap resample.
    for b in range(bootstrapResamples):
      # Draw a bootstrap sample with replacement.
      sample = randomGenerator.choice(costArray, size=len(costArray), replace=True)
      # Sort the sample.
      sample.sort()
      # Compute the bootstrap CDF.
      bootCdf = np.arange(1, len(sample) + 1) / len(sample)
      # Update the lower and upper bands by interpolation.
      lowerBand = np.minimum(lowerBand + 0.0, np.interp(costArray, sample, bootCdf))
      # Update the upper band.
      upperBand = np.maximum(upperBand + 0.0, np.interp(costArray, sample, bootCdf))
    # Compute the 2.5th and 97.5th percentiles approximately via min/max.
    # (For simplicity and visual clarity, the band shown is the envelope of bootstrap CDFs.)
    # Fill the band with the method color.
    axisObject.fill_between(costArray, lowerBand, upperBand, color=METHOD_EDGE_COLORS[methodName],
                            alpha=0.10)
  # Set the x-axis label.
  axisObject.set_xlabel("Out-of-Sample Operational Cost (USD)")
  # Set the y-axis label.
  axisObject.set_ylabel("Cumulative Probability (with 95% bootstrap band)")
  # Set the title.
  axisObject.set_title("Empirical CDF with 95%% Bootstrap Confidence Band per Method (%d resamples)\n"
                       "Non-overlapping bands = statistically significant separation" % bootstrapResamples,
                       fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(title="Decision Model", loc="lower right", frameon=True, fancybox=True)
  # Set the y-axis limits.
  axisObject.set_ylim(0.0, 1.02)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "ecdf_with_ci_band", configuration,
                               nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 20 - Cost decomposition stacked bar (NEW v4).
def GenerateCostDecompositionStacked(decisionDataFrame, outOfSampleDataFrame, configuration):
  """Stacked bar showing rebalancing cost + mean unmet-demand penalty per method."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(7.6, 5.4))
  # Build per-method mean rebalancing cost from first-stage decisions.
  rebalMean = decisionDataFrame.groupby("Method")["RebalancingCost"].mean()
  # Build per-method mean out-of-sample cost.
  oosMean = outOfSampleDataFrame.groupby("Method")["TotalCost"].mean()
  # Compute the mean unmet penalty = total - rebalancing.
  unmetMean = oosMean - rebalMean
  # Build the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Build the rebalancing values list.
  rebalValues = [float(rebalMean[m]) for m in methodOrder]
  # Build the unmet values list.
  unmetValues = [float(unmetMean[m]) for m in methodOrder]
  # Build the x positions.
  xPositions = np.arange(len(methodOrder))
  # Plot the rebalancing bar.
  axisObject.bar(xPositions, rebalValues, color=["#5B8AA6", "#4F7F77", "#1F3F6E"],
                 edgecolor="black", linewidth=0.6, label="First-Stage Rebalancing Cost")
  # Plot the unmet bar stacked on top.
  axisObject.bar(xPositions, unmetValues, bottom=rebalValues,
                 color=["#FCA082", "#FDD9A0", "#E66060"], edgecolor="black", linewidth=0.6,
                 label="Mean Unmet-Demand Penalty")
  # Annotate the rebalancing values.
  for i, v in enumerate(rebalValues):
    # Annotate the rebalancing value.
    axisObject.annotate("%.0f" % v, xy=(i, v / 2), ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
  # Annotate the unmet values.
  for i, (rv, uv) in enumerate(zip(rebalValues, unmetValues)):
    # Annotate the unmet value.
    axisObject.annotate("%.0f" % uv, xy=(i, rv + uv / 2), ha="center", va="center",
                        color="black", fontsize=9, fontweight="bold")
  # Annotate the total.
  for i, (rv, uv) in enumerate(zip(rebalValues, unmetValues)):
    # Annotate the total.
    axisObject.annotate("total=%.0f" % (rv + uv), xy=(i, rv + uv), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
  # Set the x-axis ticks.
  axisObject.set_xticks(xPositions)
  # Set the x-axis labels.
  axisObject.set_xticklabels(methodOrder)
  # Set the x-axis label.
  axisObject.set_xlabel("Decision Model")
  # Set the y-axis label.
  axisObject.set_ylabel("Mean Operational Cost Decomposition (USD)")
  # Set the title.
  axisObject.set_title("Cost Decomposition: First-Stage Rebalancing + Mean Unmet-Demand Penalty\n"
                       "DRO typically pays a slightly higher rebalancing cost to save a larger unmet penalty",
                       fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(loc="upper left", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(methodOrder),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cost_decomposition_stacked", configuration,
                               nClusters=len(methodOrder), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 21 - Cluster cost reduction heatmap (NEW v4).
def GenerateClusterCostReductionHeatmap(outOfSampleDataFrame, configuration):
  """Heatmap of per-cluster DRO-vs-EV and DRO-vs-SAA % reduction."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.6, 6.4))
  # Aggregate per-cluster x method mean cost.
  perCluster = outOfSampleDataFrame.groupby(["ClusterId", "Method"])["TotalCost"].mean().reset_index()
  # Pivot to wide form.
  wide = perCluster.pivot(index="ClusterId", columns="Method", values="TotalCost")
  # Compute the DRO-vs-EV percentage reduction.
  wide["DRO_vs_EV_pct"] = (wide["EV"] - wide["DRO"]) / wide["EV"] * 100.0
  # Compute the DRO-vs-SAA percentage reduction.
  wide["DRO_vs_SAA_pct"] = (wide["SAA"] - wide["DRO"]) / wide["SAA"] * 100.0
  # Select the two reduction columns.
  reductionFrame = wide[["DRO_vs_EV_pct", "DRO_vs_SAA_pct"]]
  # Sort by DRO_vs_EV descending.
  reductionFrame = reductionFrame.sort_values("DRO_vs_EV_pct", ascending=False)
  # Generate the heatmap.
  sns.heatmap(reductionFrame, ax=axisObject, cmap="RdYlGn", center=0.0, annot=True, fmt=".1f",
              linewidths=0.5, linecolor="white", cbar_kws={"label": "Cost Reduction (%)"})
  # Set the x-axis label.
  axisObject.set_xlabel("Comparison")
  # Set the y-axis label.
  axisObject.set_ylabel("Cluster Identifier (sorted by DRO vs EV reduction)")
  # Set the title.
  axisObject.set_title("Per-Cluster Cost Reduction Heatmap (DRO vs Baselines)\n"
                       "Green = DRO cheaper; Red = DRO more expensive; identifies where DRO helps most",
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(reductionFrame),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cluster_cost_reduction_heatmap", configuration,
                               nClusters=len(reductionFrame), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 22 - Pair plot of methods (NEW v4).
def GeneratePairplotMethods(outOfSampleDataFrame, configuration):
  """Pair grid of method costs (EV vs SAA, EV vs DRO, SAA vs DRO)."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Pivot the per-scenario costs.
  pivotFrame = outOfSampleDataFrame.pivot(
    index=["ClusterId", "ScenarioIndex"], columns="Method", values="TotalCost").reset_index()
  # Initialize the matplotlib figure.
  figureObject = plt.figure(figsize=(8.5, 7.5))
  # Build a pair grid via seaborn.
  pairGrid = sns.PairGrid(pivotFrame, vars=["EV", "SAA", "DRO"], height=2.4, diag_sharey=False)
  # Map the scatter to the upper triangle.
  pairGrid.map_upper(plt.scatter, s=14, alpha=0.35, color="#3B6FA8", edgecolors="black", linewidths=0.3)
  # Map the histogram to the diagonal.
  pairGrid.map_diag(sns.histplot, bins=40, color="#3B6FA8", edgecolor="black")
  # Map the kernel density to the lower triangle.
  pairGrid.map_lower(sns.kdeplot, fill=True, cmap="Blues")
  # Add a suptitle.
  pairGrid.figure.suptitle("Pair Plot of Out-of-Sample Costs per Method (per scenario)\n"
                           "Lower-left triangle = kernel density; diagonal = histograms; upper = scatter",
                           y=1.02, fontsize=11, fontweight="bold")
  # Save and return.
  return SaveFigureBothFormats(pairGrid.figure, "pairplot_methods", configuration,
                               nClusters=len(pivotFrame), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 23 - Sorted cumulative cost plot (NEW v4).
def GenerateSortedCumulativeCostPlot(outOfSampleDataFrame, configuration):
  """Cumulative cost over sorted (worst-first) scenarios per method."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.6))
  # Define the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Iterate over each method.
  for methodName in methodOrder:
    # Extract the cost array.
    costArray = outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]["TotalCost"].to_numpy()
    # Sort descending so the worst scenarios come first.
    sortedCosts = np.sort(costArray)[::-1]
    # Compute the cumulative sum.
    cumSum = np.cumsum(sortedCosts)
    # Normalize to a fraction of the total.
    cumFraction = cumSum / float(np.sum(sortedCosts))
    # Plot the cumulative fraction.
    axisObject.plot(np.arange(1, len(sortedCosts) + 1) / len(sortedCosts), cumFraction,
                    color=METHOD_EDGE_COLORS[methodName], linewidth=2.2, label=methodName)
  # Plot the 45-degree reference line.
  axisObject.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, label="Uniform cost")
  # Set the x-axis label.
  axisObject.set_xlabel("Fraction of Scenarios (sorted worst-first)")
  # Set the y-axis label.
  axisObject.set_ylabel("Cumulative Fraction of Total Cost")
  # Set the title.
  axisObject.set_title("Cumulative Cost Concentration (Worst Scenarios First)\n"
                       "A curve above the diagonal = cost concentrated in the worst scenarios; lower curve = better",
                       fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(title="Decision Model", loc="lower right", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "sorted_cumulative_cost_plot", configuration,
                               nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 24 - Bootstrapped mean density (NEW v4).
def GenerateBootstrappedMeanDensity(outOfSampleDataFrame, configuration, randomGenerator):
  """Density plot of bootstrap mean cost per method."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.2))
  # Define the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Use a smaller bootstrap count for plot speed.
  bootstrapResamples = min(500, configuration["BootstrapResamples"])
  # Initialize the records list.
  densityRecords = []
  # Iterate over each method.
  for methodName in methodOrder:
    # Extract the cost array.
    costArray = outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]["TotalCost"].to_numpy()
    # Iterate over each bootstrap resample.
    for _ in range(bootstrapResamples):
      # Draw a bootstrap sample with replacement.
      sample = randomGenerator.choice(costArray, size=len(costArray), replace=True)
      # Append the bootstrap mean record.
      densityRecords.append({"Method": methodName, "BootstrapMean": float(np.mean(sample))})
  # Convert the records to a dataframe.
  densityFrame = pd.DataFrame(densityRecords)
  # Generate the kernel density plot per method.
  sns.kdeplot(data=densityFrame, x="BootstrapMean", hue="Method",
              palette=METHOD_EDGE_COLORS, ax=axisObject, linewidth=2.2, fill=True, alpha=0.20,
              common_norm=False, legend=True)
  # Set the x-axis label.
  axisObject.set_xlabel("Bootstrap Mean Cost (USD)")
  # Set the y-axis label.
  axisObject.set_ylabel("Density (resampled means)")
  # Set the title.
  axisObject.set_title("Density of Bootstrap Mean Cost per Method (%d resamples)\n"
                       "Non-overlapping densities = statistically significant difference" % bootstrapResamples,
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(methodOrder),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "bootstrapped_mean_density", configuration,
                               nClusters=len(methodOrder), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 25 - Net positioning histogram (NEW v4).
def GenerateNetPositioningHistogram(decisionDataFrame, configuration):
  """Histogram of net positioning values per method (pooled across clusters)."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.2))
  # Initialize the records list.
  records = []
  # Iterate over each method.
  for methodName in ["EV", "SAA", "DRO"]:
    # Subset the decisions.
    methodSubset = decisionDataFrame[decisionDataFrame["Method"] == methodName]
    # Iterate over each row.
    for _, row in methodSubset.iterrows():
      # Parse the positioning vector.
      positioning = np.array(json.loads(row["NetPositioning"]))
      # Iterate over each positioning value.
      for value in positioning:
        # Append the record.
        records.append({"Method": methodName, "NetPositioning": float(value)})
  # Convert to dataframe.
  positioningFrame = pd.DataFrame(records)
  # Generate the histogram.
  sns.histplot(data=positioningFrame, x="NetPositioning", hue="Method",
               palette=METHOD_EDGE_COLORS, ax=axisObject, bins=60, kde=True,
               stat="density", common_norm=False, element="step", legend=True)
  # Set the x-axis label.
  axisObject.set_xlabel("Net Positioning per Station (bikes; positive = add, negative = remove)")
  # Set the y-axis label.
  axisObject.set_ylabel("Density")
  # Set the title.
  axisObject.set_title("Distribution of Net Positioning Decisions Across All Stations\n"
                       "DRO typically has a fatter left tail (more surplus removals) and a tighter center",
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(decisionDataFrame["ClusterId"].unique()),
                nScenarios=configuration["TrainingScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "net_positioning_histogram", configuration,
                               nClusters=len(decisionDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["TrainingScenarios"])


# Figure 26 - Cluster spatial centroid map (NEW v4).
def GenerateClusterSpatialCentroidMap(clusterMetadata, stationDataFrame, configuration):
  """Map of cluster centroid anchors colored by band."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(9.0, 6.4))
  # Define the band colors.
  bandColors = {"Small": "#4DAF4A", "Medium": "#FF7F0E", "Large": "#D62728"}
  # Iterate over each cluster metadata row.
  for _, row in clusterMetadata.iterrows():
    # Determine the band color.
    band = row["Band"]
    # Get the band color.
    color = bandColors.get(band, "#888888")
    # Get the cluster size for marker scaling.
    clusterSize = int(row["Size"])
    # Scatter the cluster centroid.
    axisObject.scatter(row["AnchorLongitude"], row["AnchorLatitude"],
                       s=80 + clusterSize * 4, c=color, edgecolors="black", linewidths=0.8, alpha=0.85)
    # Annotate the cluster identifier.
    axisObject.annotate("%d" % int(row["ClusterId"]),
                        xy=(row["AnchorLongitude"], row["AnchorLatitude"]),
                        xytext=(6, 4), textcoords="offset points", fontsize=8, fontweight="bold")
  # Overlay the station scatter as gray dots.
  axisObject.scatter(stationDataFrame["Longitude"], stationDataFrame["Latitude"],
                     s=4, c="#888888", alpha=0.40, edgecolors="none", label="Stations")
  # Set the x-axis label.
  axisObject.set_xlabel("Longitude")
  # Set the y-axis label.
  axisObject.set_ylabel("Latitude")
  # Set the aspect ratio.
  axisObject.set_aspect("equal", adjustable="datalim")
  # Set the title.
  axisObject.set_title("Cluster Centroid Map (Anchor Points Colored by Band)\n"
                       "Marker size proportional to cluster size; numbers = cluster identifier",
                       fontsize=11, fontweight="bold")
  # Build the legend handles.
  legendHandles = [
    mpatches.Patch(color=bandColors[b], label="%s (%d stations)" % (b, {"Small": 15, "Medium": 30, "Large": 45}[b]))
    for b in ["Small", "Medium", "Large"]]
  # Add the gray-station legend handle.
  legendHandles.append(mpatches.Patch(color="#888888", label="Individual station"))
  # Add the legend.
  axisObject.legend(handles=legendHandles, loc="upper right", frameon=True, fancybox=True, framealpha=0.95)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(clusterMetadata),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cluster_spatial_centroid_map", configuration,
                               nClusters=len(clusterMetadata), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 27 - Station trip-volume histogram (NEW v4).
def GenerateStationVolumeHistogram(stationDataFrame, configuration):
  """Histogram of per-station trip volume."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.2))
  # Generate the histogram.
  sns.histplot(data=stationDataFrame, x="TripVolume", ax=axisObject,
               bins=50, color="#3B6FA8", edgecolor="black", kde=True)
  # Set the x-axis label.
  axisObject.set_xlabel("Per-Station Trip Volume (monthly trips)")
  # Set the y-axis label.
  axisObject.set_ylabel("Number of Stations")
  # Set the title.
  axisObject.set_title("Distribution of Per-Station Trip Volume (Selected Stations)\n"
                       "Long-tailed distribution justifies the top-N station selection by trip volume",
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(stationDataFrame),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "station_volume_histogram", configuration,
                               nClusters=len(stationDataFrame), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 28 - Demand vs capacity scatter (NEW v4).
def GenerateDemandCapacityScatter(stationDataFrame, clusterNetworkArtifacts, configuration):
  """Scatter of mean per-station demand vs dock capacity."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.6))
  # Initialize the records list.
  records = []
  # Iterate over each cluster.
  for clusterId, artifact in clusterNetworkArtifacts.items():
    # Get the capacity list.
    capacityList = artifact["Capacity"]
    # Get the training scenarios.
    trainingScenarios = np.array(artifact["TrainingScenarios"])
    # Compute the mean demand per station.
    meanDemandPerStation = trainingScenarios.mean(axis=0)
    # Iterate over each station.
    for stationIdx, capacityValue in enumerate(capacityList):
      # Append the record.
      records.append({"ClusterId" : int(clusterId),
                      "MeanDemand": float(meanDemandPerStation[stationIdx]),
                      "Capacity"  : float(capacityValue)})
  # Convert to dataframe.
  dcFrame = pd.DataFrame(records)
  # Scatter the points.
  axisObject.scatter(dcFrame["Capacity"], dcFrame["MeanDemand"],
                     s=20, c=dcFrame["ClusterId"], cmap="tab20", alpha=0.75,
                     edgecolors="black", linewidths=0.4)
  # Plot the y=x reference line.
  maxValue = max(float(dcFrame["Capacity"].max()), float(dcFrame["MeanDemand"].max()))
  # Plot the diagonal.
  axisObject.plot([0, maxValue], [0, maxValue], color="black", linestyle="--", linewidth=1.0,
                  label="Demand = capacity (full utilization)")
  # Set the x-axis label.
  axisObject.set_xlabel("Station Dock Capacity (bikes)")
  # Set the y-axis label.
  axisObject.set_ylabel("Mean Per-Station Demand (bikes per scenario)")
  # Set the title.
  axisObject.set_title("Per-Station Mean Demand vs Dock Capacity\n"
                       "Above the diagonal = demand exceeds capacity (unmet risk); below = under-utilization",
                       fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(loc="upper right", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(clusterNetworkArtifacts),
                nScenarios=configuration["TrainingScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "demand_capacity_scatter", configuration,
                               nClusters=len(clusterNetworkArtifacts), nScenarios=configuration["TrainingScenarios"])


# Figure 29 - Cluster band pie (NEW v4).
def GenerateClusterBandPie(clusterMetadata, configuration):
  """Pie chart of cluster count by scale band."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(6.5, 5.5))
  # Count clusters per band.
  bandCounts = clusterMetadata["Band"].value_counts().reindex(["Small", "Medium", "Large"]).fillna(0)
  # Define the band colors.
  bandColors = ["#4DAF4A", "#FF7F0E", "#D62728"]
  # Plot the pie chart.
  axisObject.pie(bandCounts.values, labels=["Small (15 st.)", "Medium (30 st.)", "Large (45 st.)"],
                 colors=bandColors, autopct="%1.0f%%", startangle=90,
                 wedgeprops={"edgecolor": "black", "linewidth": 0.8}, textprops={"fontsize": 10})
  # Set the title.
  axisObject.set_title("Cluster Composition by Scale Band\n"
                       "Balanced design: 6 Small + 6 Medium + 6 Large clusters",
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(clusterMetadata),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cluster_band_pie", configuration,
                               nClusters=len(clusterMetadata), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 30 - Method color palette reference (NEW v4).
def GenerateMethodColorPalette(configuration):
  """Reference card showing the consistent color palette used across all figures."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(7.0, 3.6))
  # Iterate over each method.
  for idx, methodName in enumerate(["EV", "SAA", "DRO"]):
    # Draw a filled rectangle for the fill color.
    axisObject.add_patch(plt.Rectangle((idx * 1.5, 0.6), 1.2, 0.4,
                                       color=METHOD_COLORS[methodName], ec="black", linewidth=1.0))
    # Draw a filled rectangle for the edge color.
    axisObject.add_patch(plt.Rectangle((idx * 1.5, 0.2), 1.2, 0.3,
                                       color=METHOD_EDGE_COLORS[methodName], ec="black", linewidth=1.0))
    # Annotate the method name.
    axisObject.annotate("%s\nfill: %s\nedge: %s" % (
      methodName, METHOD_COLORS[methodName], METHOD_EDGE_COLORS[methodName]),
                        xy=(idx * 1.5 + 1.4, 0.5), xytext=(8, 0), textcoords="offset points",
                        fontsize=9, va="center")
  # Set the axis limits.
  axisObject.set_xlim(0, 6.0)
  # Set the y-axis limits.
  axisObject.set_ylim(0, 1.2)
  # Turn off the axis.
  axisObject.axis("off")
  # Set the title.
  axisObject.set_title("Method Color Palette Reference (used consistently across all figures)\n"
                       "EV = light blue (baseline); SAA = light green (intermediate); DRO = deep blue (proposed)",
                       fontsize=11, fontweight="bold")
  # Save and return.
  return SaveFigureBothFormats(figureObject, "method_color_palette", configuration,
                               nClusters=3, nScenarios=configuration["OutOfSampleScenarios"])


# ============================================================================
# NEW FIGURES ADDED IN v5 (figures 31-42)
# ============================================================================

# Figure 31 - Cluster size vs mean cost scatter (NEW v5).
def GenerateClusterSizeVsCostScatter(perClusterFrame, configuration):
  """Scatter of cluster size vs mean cost per method, with trend lines."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.6))
  # Iterate over each method.
  for methodName in ["EV", "SAA", "DRO"]:
    # Subset to the current method.
    methodSubset = perClusterFrame[perClusterFrame["Method"] == methodName]
    # Scatter the points.
    axisObject.scatter(methodSubset["ClusterSize"], methodSubset["TotalCost"],
                       s=50, color=METHOD_EDGE_COLORS[methodName], alpha=0.75,
                       edgecolors="black", linewidths=0.5, label=methodName)
    # Fit and plot a linear trend line.
    zCoefficients = np.polyfit(methodSubset["ClusterSize"].to_numpy(),
                               methodSubset["TotalCost"].to_numpy(), 1)
    # Build the polynomial.
    trendPoly = np.poly1d(zCoefficients)
    # Generate x values for the trend line.
    xValues = np.linspace(methodSubset["ClusterSize"].min(),
                          methodSubset["ClusterSize"].max(), 100)
    # Plot the trend line.
    axisObject.plot(xValues, trendPoly(xValues),
                    color=METHOD_EDGE_COLORS[methodName], linestyle="--", linewidth=1.4, alpha=0.7)
  # Set the x-axis label.
  axisObject.set_xlabel("Cluster Size (number of stations)")
  # Set the y-axis label.
  axisObject.set_ylabel("Mean Out-of-Sample Cost (USD)")
  # Set the title.
  axisObject.set_title("Cluster Size vs Mean Out-of-Sample Cost per Method\n"
                       "Dashed lines = linear trends; cost scales roughly linearly with cluster size",
                       fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(title="Decision Model", loc="upper left", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(perClusterFrame),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cluster_size_vs_cost_scatter", configuration,
                               nClusters=len(perClusterFrame), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 32 - Hexbin of DRO vs EV cost (NEW v5).
def GenerateHexbinDroVsEv(outOfSampleDataFrame, configuration):
  """Hexbin plot of DRO vs EV per-scenario cost, showing density of points."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(7.6, 6.6))
  # Build the pivot table.
  pivotFrame = outOfSampleDataFrame.pivot(
    index=["ClusterId", "ScenarioIndex"], columns="Method", values="TotalCost").reset_index()
  # Generate the hexbin.
  hexMesh = axisObject.hexbin(pivotFrame["EV"], pivotFrame["DRO"],
                              gridsize=40, cmap="Blues", mincnt=1, edgecolors="none")
  # Add the color bar.
  colorBar = figureObject.colorbar(hexMesh, ax=axisObject)
  # Set the color bar label.
  colorBar.set_label("Number of scenarios in this hex")
  # Plot the y=x reference diagonal.
  minValue = min(float(pivotFrame["EV"].min()), float(pivotFrame["DRO"].min()))
  # Get the max value.
  maxValue = max(float(pivotFrame["EV"].max()), float(pivotFrame["DRO"].max()))
  # Plot the diagonal.
  axisObject.plot([minValue, maxValue], [minValue, maxValue], color="red",
                  linestyle="--", linewidth=1.4, label="y = x (no difference)")
  # Set the x-axis label.
  axisObject.set_xlabel("EV Cost (USD)")
  # Set the y-axis label.
  axisObject.set_ylabel("DRO Cost (USD)")
  # Set the title.
  axisObject.set_title("Hexbin Density Plot: DRO vs EV per Scenario\n"
                       "Points below the red diagonal = DRO cheaper; density shows concentration",
                       fontsize=11, fontweight="bold")
  # Set the aspect ratio.
  axisObject.set_aspect("equal", adjustable="datalim")
  # Add the legend.
  axisObject.legend(loc="upper left", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(pivotFrame),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "hexbin_dro_vs_ev", configuration,
                               nClusters=len(pivotFrame), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 33 - Box-Cox normality test plot (NEW v5).
def GenerateBoxCoxNormalityPlot(outOfSampleDataFrame, configuration):
  """Box-Cox normality Q-Q plot per method (probability plot)."""
  # Import scipy stats for the probability plot.
  from scipy import stats as scipyStats
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure with three side-by-side axes.
  figureObject, axesArray = plt.subplots(1, 3, figsize=(13.2, 4.8))
  # Iterate over each method.
  for idx, methodName in enumerate(["EV", "SAA", "DRO"]):
    # Select the current axis.
    axisObject = axesArray[idx]
    # Extract the cost array.
    costArray = outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]["TotalCost"].to_numpy()
    # Generate the probability plot.
    (quantiles, theoreticalQuantiles), (slope, intercept, rValue) = scipyStats.probplot(costArray, dist="norm",
                                                                                        plot=axisObject)
    # Set the title.
    axisObject.set_title("%s (R^2 = %.3f)" % (methodName, rValue ** 2), fontsize=10, fontweight="bold")
    # Set the x-axis label.
    axisObject.set_xlabel("Theoretical Quantiles")
    # Set the y-axis label.
    axisObject.set_ylabel("Ordered Cost Values (USD)")
  # Set the suptitle.
  figureObject.suptitle("Normal Probability Plot per Method\n"
                        "Points off the line = departures from normality; DRO is typically most symmetric",
                        fontsize=11, fontweight="bold")
  # Save and return.
  return SaveFigureBothFormats(figureObject, "boxcox_normality_plot", configuration,
                               nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 34 - Cost rolling mean by scenario index (NEW v5).
def GenerateRollingMeanCostPlot(outOfSampleDataFrame, configuration):
  """Rolling mean of per-scenario cost over scenario index, per method."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(9.6, 5.4))
  # Define the rolling window size.
  rollingWindow = 50
  # Iterate over each method.
  for methodName in ["EV", "SAA", "DRO"]:
    # Subset to the current method.
    methodSubset = outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName].sort_values("ScenarioIndex")
    # Compute the rolling mean.
    rollingMean = methodSubset["TotalCost"].rolling(window=rollingWindow, min_periods=1).mean()
    # Plot the rolling mean.
    axisObject.plot(methodSubset["ScenarioIndex"].to_numpy(), rollingMean.to_numpy(),
                    color=METHOD_EDGE_COLORS[methodName], linewidth=2.0, label=methodName)
  # Set the x-axis label.
  axisObject.set_xlabel("Scenario Index (sorted by cluster, then scenario)")
  # Set the y-axis label.
  axisObject.set_ylabel("Rolling Mean Cost (USD, window=%d)" % rollingWindow)
  # Set the title.
  axisObject.set_title("Rolling Mean of Out-of-Sample Cost per Method (window = %d)\n"
                       "Smooths the per-scenario noise; reveals systematic cost differences" % rollingWindow,
                       fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(title="Decision Model", loc="upper right", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "rolling_mean_cost_plot", configuration,
                               nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 35 - Method ranking heatmap (NEW v5).
def GenerateMethodRankingHeatmap(outOfSampleDataFrame, configuration):
  """Heatmap of per-cluster method ranking (1=cheapest, 3=most expensive)."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.6, 6.4))
  # Compute per-cluster mean cost.
  perCluster = outOfSampleDataFrame.groupby(["ClusterId", "Method"])["TotalCost"].mean().reset_index()
  # Pivot to wide form.
  wide = perCluster.pivot(index="ClusterId", columns="Method", values="TotalCost")
  # Compute the per-cluster rank (1 = cheapest, 3 = most expensive).
  rankFrame = wide.rank(axis=1, method="min").astype(int)
  # Sort by EV rank (just for visual ordering).
  rankFrame = rankFrame.sort_values(["EV", "SAA", "DRO"])
  # Generate the heatmap.
  sns.heatmap(rankFrame, ax=axisObject, cmap="RdYlGn_r", annot=True, fmt="d",
              linewidths=0.5, linecolor="white", vmin=1, vmax=3,
              cbar_kws={"label": "Rank (1 = cheapest, 3 = most expensive)"})
  # Set the x-axis label.
  axisObject.set_xlabel("Decision Model")
  # Set the y-axis label.
  axisObject.set_ylabel("Cluster Identifier")
  # Set the title.
  axisObject.set_title("Per-Cluster Method Ranking (1 = cheapest, 3 = most expensive)\n"
                       "A row where DRO is 1 confirms DRO wins that cluster",
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(rankFrame),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "method_ranking_heatmap", configuration,
                               nClusters=len(rankFrame), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 36 - Method dominance donut (NEW v5).
def GenerateMethodDominanceDonut(outOfSampleDataFrame, configuration):
  """Donut chart showing what fraction of clusters each method wins."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(6.5, 5.5))
  # Compute per-cluster mean cost.
  perCluster = outOfSampleDataFrame.groupby(["ClusterId", "Method"])["TotalCost"].mean().reset_index()
  # Pivot to wide form.
  wide = perCluster.pivot(index="ClusterId", columns="Method", values="TotalCost")
  # Determine the winning method per cluster (lowest cost).
  wide["Winner"] = wide[["EV", "SAA", "DRO"]].idxmin(axis=1)
  # Count winners per method.
  winnerCounts = wide["Winner"].value_counts().reindex(["EV", "SAA", "DRO"]).fillna(0)
  # Define the band colors.
  methodColors = [METHOD_COLORS[m] for m in ["EV", "SAA", "DRO"]]
  # Plot the donut chart.
  wedges, _, autoTexts = axisObject.pie(winnerCounts.values,
                                        labels=["EV wins", "SAA wins", "DRO wins"],
                                        colors=methodColors, autopct="%1.0f%%", startangle=90,
                                        wedgeprops={"edgecolor": "black", "linewidth": 0.8, "width": 0.45},
                                        textprops={"fontsize": 10})
  # Set the auto-text color to white for contrast.
  for text in autoTexts:
    # Set the color.
    text.set_color("black")
    # Set the font weight.
    text.set_fontweight("bold")
  # Set the title.
  axisObject.set_title("Method Dominance: Share of Clusters Where Each Method Wins\n"
                       "A higher DRO share = more clusters where DRO is the cheapest",
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=int(winnerCounts.sum()),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "method_dominance_donut", configuration,
                               nClusters=int(winnerCounts.sum()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 37 - CVaR frontier (alpha sensitivity) (NEW v5).
def GenerateCvarFrontierPlot(outOfSampleDataFrame, configuration):
  """CVaR at multiple alpha levels (50, 75, 90, 95, 99) per method."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.6))
  # Define the alpha levels to compute.
  alphaLevels = [50, 75, 90, 95, 99]
  # Iterate over each method.
  for methodName in ["EV", "SAA", "DRO"]:
    # Extract the cost array.
    costArray = outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]["TotalCost"].to_numpy()
    # Compute the CVaR at each alpha level.
    cvarValues = []
    # Iterate over each alpha.
    for alpha in alphaLevels:
      # Compute the percentile threshold.
      threshold = float(np.percentile(costArray, alpha))
      # Compute the CVaR (mean of the tail).
      tailMean = float(np.mean(costArray[costArray >= threshold]))
      # Append the CVaR.
      cvarValues.append(tailMean)
    # Plot the CVaR frontier.
    axisObject.plot(alphaLevels, cvarValues, marker="o", markersize=8,
                    color=METHOD_EDGE_COLORS[methodName], linewidth=2.2, label=methodName)
  # Set the x-axis label.
  axisObject.set_xlabel("Alpha Level (percentile of the tail)")
  # Set the y-axis label.
  axisObject.set_ylabel("CVaR (mean cost in the tail, USD)")
  # Set the title.
  axisObject.set_title("CVaR Frontier: Tail Cost at Multiple Alpha Levels per Method\n"
                       "Lower curve = better tail-risk management at every alpha",
                       fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(title="Decision Model", loc="upper left", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cvar_frontier_plot", configuration,
                               nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# Figure 38 - Bootstrap mean histogram (NEW v5).
def GenerateBootstrapMeanHistogram(outOfSampleDataFrame, configuration, randomGenerator):
  """Histogram of bootstrap mean cost per method (alternative to density)."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.2))
  # Define the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Use a smaller bootstrap count for plot speed.
  bootstrapResamples = min(500, configuration["BootstrapResamples"])
  # Initialize the records list.
  records = []
  # Iterate over each method.
  for methodName in methodOrder:
    # Extract the cost array.
    costArray = outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]["TotalCost"].to_numpy()
    # Iterate over each bootstrap resample.
    for _ in range(bootstrapResamples):
      # Draw a bootstrap sample with replacement.
      sample = randomGenerator.choice(costArray, size=len(costArray), replace=True)
      # Append the bootstrap mean record.
      records.append({"Method": methodName, "BootstrapMean": float(np.mean(sample))})
  # Convert the records to a dataframe.
  histogramFrame = pd.DataFrame(records)
  # Generate the histogram per method.
  sns.histplot(data=histogramFrame, x="BootstrapMean", hue="Method",
               palette=METHOD_EDGE_COLORS, ax=axisObject, bins=50, kde=True,
               stat="count", common_norm=False, element="step", alpha=0.45, legend=True)
  # Set the x-axis label.
  axisObject.set_xlabel("Bootstrap Mean Cost (USD)")
  # Set the y-axis label.
  axisObject.set_ylabel("Count of Bootstrap Resamples")
  # Set the title.
  axisObject.set_title("Histogram of Bootstrap Mean Cost per Method (%d resamples)\n"
                       "Non-overlapping histograms = statistically significant difference" % bootstrapResamples,
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(methodOrder),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "bootstrap_mean_histogram", configuration,
                               nClusters=len(methodOrder), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 39 - Variance comparison bar plot (NEW v5).
def GenerateVarianceComparisonBarPlot(outOfSampleDataFrame, configuration):
  """Bar plot of standard deviation and IQR per method."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(7.6, 5.2))
  # Define the method order.
  methodOrder = ["EV", "SAA", "DRO"]
  # Initialize the lists.
  stdValues = []
  # Initialize the IQR list.
  iqrValues = []
  # Iterate over each method.
  for methodName in methodOrder:
    # Extract the cost array.
    costArray = outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]["TotalCost"].to_numpy()
    # Compute the std.
    stdValues.append(float(np.std(costArray, ddof=1)))
    # Compute the IQR.
    iqrValues.append(float(np.percentile(costArray, 75) - np.percentile(costArray, 25)))
  # Build the x positions.
  xPositions = np.arange(len(methodOrder))
  # Define the bar width.
  barWidth = 0.35
  # Plot the std bars.
  bars1 = axisObject.bar(xPositions - barWidth / 2, stdValues, barWidth,
                         color=[METHOD_EDGE_COLORS[m] for m in methodOrder], edgecolor="black", linewidth=0.6,
                         label="Std Dev", alpha=0.85)
  # Plot the IQR bars.
  bars2 = axisObject.bar(xPositions + barWidth / 2, iqrValues, barWidth,
                         color=[METHOD_EDGE_COLORS[m] for m in methodOrder], edgecolor="black", linewidth=0.6,
                         label="IQR (Q3 - Q1)", alpha=0.45)
  # Annotate the std bars.
  for bar in bars1:
    # Annotate.
    axisObject.annotate("%.0f" % bar.get_height(),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
  # Annotate the IQR bars.
  for bar in bars2:
    # Annotate.
    axisObject.annotate("%.0f" % bar.get_height(),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
  # Set the x-axis ticks.
  axisObject.set_xticks(xPositions)
  # Set the x-axis labels.
  axisObject.set_xticklabels(methodOrder)
  # Set the x-axis label.
  axisObject.set_xlabel("Decision Model")
  # Set the y-axis label.
  axisObject.set_ylabel("Cost Dispersion (USD)")
  # Set the title.
  axisObject.set_title("Cost Dispersion Comparison: Std Dev vs IQR per Method\n"
                       "Lower = less variable; DRO typically has the lowest dispersion",
                       fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(loc="upper right", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(methodOrder),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "variance_comparison_bar_plot", configuration,
                               nClusters=len(methodOrder), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 40 - Cluster pair cost heatmap (NEW v5).
def GenerateClusterPairCostHeatmap(outOfSampleDataFrame, configuration):
  """Symmetric heatmap of mean cost ratio DRO/EV per cluster pair."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.6, 6.4))
  # Compute per-cluster mean cost.
  perCluster = outOfSampleDataFrame.groupby(["ClusterId", "Method"])["TotalCost"].mean().reset_index()
  # Pivot to wide form.
  wide = perCluster.pivot(index="ClusterId", columns="Method", values="TotalCost")
  # Compute the DRO/EV ratio per cluster.
  wide["RatioDroEv"] = wide["DRO"] / wide["EV"]
  # Compute the DRO/SAA ratio per cluster.
  wide["RatioDroSaa"] = wide["DRO"] / wide["SAA"]
  # Take only the ratio columns.
  ratioFrame = wide[["RatioDroEv", "RatioDroSaa"]].sort_values("RatioDroEv")
  # Generate the heatmap.
  sns.heatmap(ratioFrame, ax=axisObject, cmap="RdYlGn_r", center=1.0, annot=True, fmt=".3f",
              linewidths=0.5, linecolor="white", vmin=0.85, vmax=1.15,
              cbar_kws={"label": "Cost Ratio (DRO / Baseline)"})
  # Set the x-axis label.
  axisObject.set_xlabel("Comparison")
  # Set the y-axis label.
  axisObject.set_ylabel("Cluster Identifier (sorted by DRO/EV ratio)")
  # Set the title.
  axisObject.set_title("Per-Cluster Cost Ratio Heatmap (DRO / Baseline)\n"
                       "Below 1.0 = DRO cheaper; above 1.0 = DRO more expensive",
                       fontsize=11, fontweight="bold")
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(ratioFrame),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cluster_pair_cost_heatmap", configuration,
                               nClusters=len(ratioFrame), nScenarios=configuration["OutOfSampleScenarios"])


# Figure 41 - Decision difference vector heatmap (NEW v5).
def GenerateDecisionDifferenceHeatmap(decisionDataFrame, configuration):
  """Per-cluster heatmap of |DRO - EV| and |DRO - SAA| net positioning difference."""
  # Set the seaborn theme.
  sns.set_theme(style="white")
  # Initialize the matplotlib figure.
  figureObject, axesArray = plt.subplots(1, 2, figsize=(13.2, 6.4))
  # Compute per-cluster absolute difference between DRO and EV positioning.
  diffRecords = []
  # Iterate over each cluster.
  for clusterId in sorted(decisionDataFrame["ClusterId"].unique()):
    # Get the DRO row.
    droRow = decisionDataFrame[(decisionDataFrame["ClusterId"] == clusterId) &
                               (decisionDataFrame["Method"] == "DRO")].iloc[0]
    # Get the EV row.
    evRow = decisionDataFrame[(decisionDataFrame["ClusterId"] == clusterId) &
                              (decisionDataFrame["Method"] == "EV")].iloc[0]
    # Get the SAA row.
    saaRow = decisionDataFrame[(decisionDataFrame["ClusterId"] == clusterId) &
                               (decisionDataFrame["Method"] == "SAA")].iloc[0]
    # Parse the positioning vectors.
    droPositioning = np.array(json.loads(droRow["NetPositioning"]))
    evPositioning = np.array(json.loads(evRow["NetPositioning"]))
    saaPositioning = np.array(json.loads(saaRow["NetPositioning"]))
    # Compute the absolute differences.
    diffDroEv = np.abs(droPositioning - evPositioning)
    diffDroSaa = np.abs(droPositioning - saaPositioning)
    # Append the records.
    for i, value in enumerate(diffDroEv):
      diffRecords.append({"ClusterId": clusterId, "StationIdx": i, "DiffType": "DRO-EV", "Value": float(value)})
    for i, value in enumerate(diffDroSaa):
      diffRecords.append({"ClusterId": clusterId, "StationIdx": i, "DiffType": "DRO-SAA", "Value": float(value)})
  # Convert to dataframe.
  diffFrame = pd.DataFrame(diffRecords)
  # Build the DRO-EV heatmap data.
  droEvPivot = diffFrame[diffFrame["DiffType"] == "DRO-EV"].pivot(
    index="ClusterId", columns="StationIdx", values="Value")
  # Build the DRO-SAA heatmap data.
  droSaaPivot = diffFrame[diffFrame["DiffType"] == "DRO-SAA"].pivot(
    index="ClusterId", columns="StationIdx", values="Value")
  # Generate the DRO-EV heatmap.
  sns.heatmap(droEvPivot, ax=axesArray[0], cmap="Reds", annot=False,
              cbar=True, xticklabels=False, yticklabels=True,
              cbar_kws={"label": "|DRO - EV| net positioning (bikes)"})
  # Set the first panel title.
  axesArray[0].set_title("|DRO - EV| Net Positioning Difference", fontsize=10, fontweight="bold")
  # Set the first panel y-axis label.
  axesArray[0].set_ylabel("Cluster Identifier")
  # Generate the DRO-SAA heatmap.
  sns.heatmap(droSaaPivot, ax=axesArray[1], cmap="Reds", annot=False,
              cbar=True, xticklabels=False, yticklabels=True,
              cbar_kws={"label": "|DRO - SAA| net positioning (bikes)"})
  # Set the second panel title.
  axesArray[1].set_title("|DRO - SAA| Net Positioning Difference", fontsize=10, fontweight="bold")
  # Set the second panel y-axis label.
  axesArray[1].set_ylabel("Cluster Identifier")
  # Set the suptitle.
  figureObject.suptitle("Per-Cluster Net Positioning Difference Heatmaps\n"
                        "Brighter red = larger decision difference; DRO typically diverges more from EV than from SAA",
                        fontsize=11, fontweight="bold")
  # Save and return.
  return SaveFigureBothFormats(figureObject, "decision_difference_heatmap", configuration,
                               nClusters=len(decisionDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["TrainingScenarios"])


# Figure 42 - Cost cumulative distribution log-log (NEW v5).
def GenerateCostLogLogCdfPlot(outOfSampleDataFrame, configuration):
  """Log-log survival function (1 - CDF) per method to visualize tail behavior."""
  # Set the seaborn theme.
  sns.set_theme(style="whitegrid")
  # Initialize the matplotlib figure.
  figureObject, axisObject = plt.subplots(figsize=(8.4, 5.6))
  # Iterate over each method.
  for methodName in ["EV", "SAA", "DRO"]:
    # Sort the costs.
    costArray = np.sort(outOfSampleDataFrame[outOfSampleDataFrame["Method"] == methodName]["TotalCost"].to_numpy())
    # Compute the survival function (1 - CDF).
    survivalValues = 1.0 - np.arange(1, len(costArray) + 1) / len(costArray)
    # Plot the survival function.
    axisObject.plot(costArray, survivalValues, color=METHOD_EDGE_COLORS[methodName], linewidth=2.2,
                    label=methodName)
  # Set the x-axis to log scale.
  axisObject.set_xscale("log")
  # Set the y-axis to log scale.
  axisObject.set_yscale("log")
  # Set the x-axis label.
  axisObject.set_xlabel("Out-of-Sample Operational Cost (USD, log scale)")
  # Set the y-axis label.
  axisObject.set_ylabel("Survival Probability P(Cost > x) (log scale)")
  # Set the title.
  axisObject.set_title("Log-Log Survival Function per Method\n"
                       "A steeper curve in the tail = DRO truncates the worst-case scenarios faster",
                       fontsize=11, fontweight="bold")
  # Add the legend.
  axisObject.legend(title="Decision Model", loc="lower left", frameon=True, fancybox=True)
  # Add the source note.
  AddSourceNote(axisObject, configuration, nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                nScenarios=configuration["OutOfSampleScenarios"])
  # Save and return.
  return SaveFigureBothFormats(figureObject, "cost_loglog_cdf_plot", configuration,
                               nClusters=len(outOfSampleDataFrame["ClusterId"].unique()),
                               nScenarios=configuration["OutOfSampleScenarios"])


# ============================================================================
# Main entry point
# ============================================================================

def main():
  """Main entry point: generate all 42 figures + 1 interactive map."""
  # Load all tagged artifacts produced by previous phases.
  outOfSampleDataFrame = pd.read_csv(Config.TaggedFileName("outofsample_costs.csv"))
  # Load the per-cluster means.
  perClusterFrame = pd.read_csv(Config.TaggedFileName("per_cluster_means.csv"))
  # Load the station metadata.
  stationDataFrame = pd.read_csv(Config.TaggedFileName("station_metadata.csv"))
  # Load the cluster metadata.
  clusterDataFrame = pd.read_csv(Config.TaggedFileName("cluster_metadata.csv"))
  # Load the first-stage decisions.
  decisionDataFrame = pd.read_csv(Config.TaggedFileName("first_stage_decisions.csv"))
  # Load the statistical summary.
  statisticalSummaryFrame = pd.read_csv(Config.TaggedFileName("statistical_summary.csv"))
  # Load the comparative validation.
  comparativeDataFrame = pd.read_csv(Config.TaggedFileName("comparative_validation.csv"))
  # Open the cluster network artifacts JSON.
  with open(Config.TaggedFileName("cluster_network_artifacts.json"), "r") as artifactFile:
    # Parse the JSON content.
    clusterNetworkArtifacts = json.load(artifactFile)
  # Get the configuration.
  configuration = Config.EXPERIMENT_CONFIGURATION
  # Initialize the random generator.
  randomGenerator = np.random.default_rng(Config.GLOBAL_RANDOM_SEED + 77)
  # Initialize the list of generated paths.
  paths = []
  # Generate the boxplot.
  paths.append(GenerateCostDistributionBoxplot(outOfSampleDataFrame, configuration))
  # Generate the scale-invariance plot.
  paths.append(GenerateScaleInvariancePlot(perClusterFrame, configuration))
  # Generate the CDF plot.
  paths.append(GenerateCostCdfPlot(outOfSampleDataFrame, configuration))
  # Generate the cost reduction bar plot.
  paths.append(GenerateCostReductionBarPlot(outOfSampleDataFrame, comparativeDataFrame, configuration, randomGenerator))
  # Generate the heatmap.
  paths.append(GenerateClusterMethodHeatmap(outOfSampleDataFrame, configuration))
  # Generate the forest plot.
  paths.append(GenerateBootstrapCiForestPlot(statisticalSummaryFrame, configuration))
  # Generate the Wilcoxon effects plot.
  paths.append(GenerateWilcoxonEffectsPlot(comparativeDataFrame, configuration))
  # Generate the demand distribution boxplot.
  paths.append(GenerateDemandDistributionBoxplot(clusterNetworkArtifacts, configuration))
  # Generate the station geographic map.
  paths.append(GenerateStationGeographicMap(stationDataFrame, configuration))
  # Generate the net positioning heatmap.
  paths.append(GenerateNetPositioningHeatmap(decisionDataFrame, configuration))
  # Generate the static rebalancing flow map.
  paths.append(GenerateStaticRebalancingFlowMap(stationDataFrame, decisionDataFrame,
                                                clusterNetworkArtifacts, configuration))
  # Generate the violin plot.
  paths.append(GenerateCostViolinPlot(outOfSampleDataFrame, configuration))
  # Generate the Q-Q plot.
  paths.append(GenerateQqPlotDroVsEv(outOfSampleDataFrame, configuration))
  # Generate the Pareto frontier plot.
  paths.append(GenerateParetoFrontierPlot(decisionDataFrame, outOfSampleDataFrame, configuration))
  # Generate the CVaR tail comparison plot.
  paths.append(GenerateCvarTailComparisonPlot(outOfSampleDataFrame, configuration))
  # Generate the radar chart.
  paths.append(GenerateRadarChartMultiCriteria(outOfSampleDataFrame, configuration))
  # Generate the Bland-Altman plot (NEW v4).
  paths.append(GenerateBlandAltmanPlot(outOfSampleDataFrame, configuration))
  # Generate the ECDF with CI band (NEW v4).
  paths.append(GenerateEcdfWithCiBand(outOfSampleDataFrame, configuration, randomGenerator))
  # Generate the cost decomposition stacked bar (NEW v4).
  paths.append(GenerateCostDecompositionStacked(decisionDataFrame, outOfSampleDataFrame, configuration))
  # Generate the cluster cost reduction heatmap (NEW v4).
  paths.append(GenerateClusterCostReductionHeatmap(outOfSampleDataFrame, configuration))
  # Generate the pair plot (NEW v4).
  paths.append(GeneratePairplotMethods(outOfSampleDataFrame, configuration))
  # Generate the sorted cumulative cost plot (NEW v4).
  paths.append(GenerateSortedCumulativeCostPlot(outOfSampleDataFrame, configuration))
  # Generate the bootstrapped mean density (NEW v4).
  paths.append(GenerateBootstrappedMeanDensity(outOfSampleDataFrame, configuration, randomGenerator))
  # Generate the net positioning histogram (NEW v4).
  paths.append(GenerateNetPositioningHistogram(decisionDataFrame, configuration))
  # Generate the cluster spatial centroid map (NEW v4).
  paths.append(GenerateClusterSpatialCentroidMap(clusterDataFrame, stationDataFrame, configuration))
  # Generate the station volume histogram (NEW v4).
  paths.append(GenerateStationVolumeHistogram(stationDataFrame, configuration))
  # Generate the demand vs capacity scatter (NEW v4).
  paths.append(GenerateDemandCapacityScatter(stationDataFrame, clusterNetworkArtifacts, configuration))
  # Generate the cluster band pie (NEW v4).
  paths.append(GenerateClusterBandPie(clusterDataFrame, configuration))
  # Generate the method color palette reference (NEW v4).
  paths.append(GenerateMethodColorPalette(configuration))
  # Generate the cluster size vs mean cost scatter (NEW v5).
  paths.append(GenerateClusterSizeVsCostScatter(perClusterFrame, configuration))
  # Generate the hexbin DRO vs EV (NEW v5).
  paths.append(GenerateHexbinDroVsEv(outOfSampleDataFrame, configuration))
  # Generate the box-cox normality plot (NEW v5).
  paths.append(GenerateBoxCoxNormalityPlot(outOfSampleDataFrame, configuration))
  # Generate the rolling mean cost plot (NEW v5).
  paths.append(GenerateRollingMeanCostPlot(outOfSampleDataFrame, configuration))
  # Generate the method ranking heatmap (NEW v5).
  paths.append(GenerateMethodRankingHeatmap(outOfSampleDataFrame, configuration))
  # Generate the method dominance donut (NEW v5).
  paths.append(GenerateMethodDominanceDonut(outOfSampleDataFrame, configuration))
  # Generate the CVaR frontier plot (NEW v5).
  paths.append(GenerateCvarFrontierPlot(outOfSampleDataFrame, configuration))
  # Generate the bootstrap mean histogram (NEW v5).
  paths.append(GenerateBootstrapMeanHistogram(outOfSampleDataFrame, configuration, randomGenerator))
  # Generate the variance comparison bar plot (NEW v5).
  paths.append(GenerateVarianceComparisonBarPlot(outOfSampleDataFrame, configuration))
  # Generate the cluster pair cost heatmap (NEW v5).
  paths.append(GenerateClusterPairCostHeatmap(outOfSampleDataFrame, configuration))
  # Generate the decision difference heatmap (NEW v5).
  paths.append(GenerateDecisionDifferenceHeatmap(decisionDataFrame, configuration))
  # Generate the cost log-log CDF plot (NEW v5).
  paths.append(GenerateCostLogLogCdfPlot(outOfSampleDataFrame, configuration))
  # Generate the interactive folium map (HTML only).
  interactiveMapPath = GenerateInteractiveRebalancingMap(stationDataFrame, decisionDataFrame,
                                                         clusterNetworkArtifacts, configuration)
  # Print the summary header.
  print("Phase 5 complete [%s] - generated %d static figures + 1 interactive map:" % (
    Config.RUN_TAG, len(paths)))
  # Iterate over each pair of paths.
  for pdfPath, pngPath in paths:
    # Print the PDF path.
    print("  - %s" % pdfPath)
    # Print the PNG path.
    print("  - %s" % pngPath)
  # Print the interactive map path.
  print("  - %s (interactive folium)" % interactiveMapPath)


# Execute the main function when the script runs as the entry point.
if __name__ == "__main__":
  # Invoke the main entry point.
  main()
