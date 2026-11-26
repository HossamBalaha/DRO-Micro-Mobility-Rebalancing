"""
Phase3BaselinesOutOfSample.py
===============================

Phase 3 - Out-of-sample evaluation of the three first-stage decisions.

OUTPUTS (all prefixed with `{City}_{YearMonthRange}{RunSuffix}_`)
  * outofsample_costs.csv
"""

# Import the numpy library for numerical array operations.
import numpy as np
# Import the pandas library for tabular result storage.
import pandas as pd
# Import the json library for loading network artifacts.
import json
# Import the os library for file path construction.
import os
# Import the shared configuration module for global knobs and paths.
import Config


def EvaluateOutOfSampleCost(netPositioningVector, rebalancingCost, outOfSampleScenarios, unmetPenalty,
                            initialInventoryVector):
  """Evaluate the out-of-sample operational cost for a first-stage decision."""
  # Compute the post-rebalancing target inventory for each station.
  targetInventoryVector = initialInventoryVector + netPositioningVector.reshape(1, -1)
  # Compute the unmet demand for each out-of-sample scenario.
  unmetDemandMatrix = np.maximum(0.0, outOfSampleScenarios - targetInventoryVector)
  # Compute the recourse cost for each out-of-sample scenario.
  recourseCostArray = unmetPenalty * np.sum(unmetDemandMatrix, axis=1)
  # Compute the total operational cost for each out-of-sample scenario.
  totalCostArray = rebalancingCost + recourseCostArray
  # Return the array of total operational costs.
  return totalCostArray


def main():
  """Main entry point: evaluate every (cluster, method) on out-of-sample scenarios."""
  # Load the cluster network artifacts from the tagged JSON file.
  with open(Config.TaggedFileName("cluster_network_artifacts.json"), "r") as artifactFile:
    # Parse the JSON content into a dictionary.
    clusterArtifacts = json.load(artifactFile)
  # Load the first-stage decisions from the tagged CSV file.
  decisionDataFrame = pd.read_csv(Config.TaggedFileName("first_stage_decisions.csv"))
  # Load the station metadata to recover the band labels.
  stationDataFrame = pd.read_csv(Config.TaggedFileName("station_metadata.csv"))
  # Retrieve the experiment configuration dictionary.
  configuration = Config.EXPERIMENT_CONFIGURATION
  # Initialize the list storing out-of-sample cost records.
  outOfSampleRecords = []
  # Iterate over each cluster identifier in the sorted artifact keys.
  for clusterIdentifier in sorted(clusterArtifacts.keys(), key=lambda value: int(value)):
    # Retrieve the artifact dictionary for the current cluster.
    clusterArtifact = clusterArtifacts[clusterIdentifier]
    # Convert the out-of-sample scenarios to a numpy array.
    outOfSampleScenarios = np.array(clusterArtifact["OutOfSampleScenarios"])
    # Recover the initial inventory vector for the current cluster.
    initialInventoryVector = np.array(clusterArtifact["InitialInventory"], dtype=float)
    # Retrieve the band label for the current cluster.
    clusterBand = stationDataFrame[stationDataFrame["ClusterId"] == int(clusterIdentifier)].iloc[0]["Band"]
    # Iterate over each method present in the decision dataframe.
    for method in ["EV", "SAA", "DRO"]:
      # Select the decision row for the current cluster and method.
      decisionRow = decisionDataFrame[
        (decisionDataFrame["ClusterId"] == int(clusterIdentifier)) &
        (decisionDataFrame["Method"] == method)]
      # Continue to the next method when the decision row is missing.
      if (len(decisionRow) == 0):
        # Continue to the next method.
        continue
      # Extract the net positioning vector from the decision row.
      netPositioningVector = np.array(json.loads(decisionRow.iloc[0]["NetPositioning"]))
      # Extract the rebalancing cost from the decision row.
      rebalancingCost = float(decisionRow.iloc[0]["RebalancingCost"])
      # Evaluate the out-of-sample cost array for the current decision.
      totalCostArray = EvaluateOutOfSampleCost(
        netPositioningVector, rebalancingCost, outOfSampleScenarios,
        configuration["UnmetDemandPenalty"], initialInventoryVector)
      # Iterate over each out-of-sample scenario index.
      for scenarioIndex in range(len(totalCostArray)):
        # Append the out-of-sample cost record.
        outOfSampleRecords.append({
          "ClusterId"    : int(clusterIdentifier), "Band": clusterBand,
          "ClusterSize"  : int(clusterArtifact["ClusterSize"]), "Method": method,
          "ScenarioIndex": scenarioIndex,
          "TotalCost"    : float(totalCostArray[scenarioIndex])})
  # Convert the out-of-sample records to a pandas dataframe.
  outOfSampleDataFrame = pd.DataFrame(outOfSampleRecords)
  # Save the out-of-sample costs to a tagged CSV file.
  outOfSampleDataFrame.to_csv(Config.TaggedFileName("outofsample_costs.csv"), index=False)
  # Compute summary statistics for the out-of-sample costs.
  summaryFrame = outOfSampleDataFrame.groupby(["Method"])["TotalCost"].agg(["mean", "std", "median"]).round(2)
  # Print the summary statistics header.
  print("Phase 3 out-of-sample summary by method [%s]:" % Config.RUN_TAG)
  # Print the summary dataframe to the console.
  print(summaryFrame.to_string())
  # Print a confirmation message summarizing the evaluation phase.
  print("Phase 3 complete [%s]: evaluated %d real out-of-sample scenarios across %d clusters and three methods." % (
    Config.RUN_TAG, configuration["OutOfSampleScenarios"], len(clusterArtifacts)))


# Execute the main function when the script runs as the entry point.
if __name__ == "__main__":
  # Invoke the main entry point.
  main()
