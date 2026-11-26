"""
Phase2DroModel.py
===================

Phase 2 - Two-stage linear programming: solve EV, SAA, and DRO models per cluster.

OUTPUTS (all prefixed with `{City}_{YearMonthRange}{RunSuffix}_`)
  * first_stage_decisions.csv
"""

# Import the pulp library for linear programming optimization.
import pulp
# Import the numpy library for numerical array operations.
import numpy as np
# Import the json library for loading network artifacts.
import json
# Import the os library for file path construction.
import os
# Import the pandas library for tabular result storage.
import pandas as pd
# Import the shared configuration module for global knobs and paths.
import Config


def InstantiateHighsSolver():
  """Instantiate the HiGHS LP solver with quiet output."""
  # Return a HiGHS solver instance configured for silent execution.
  return pulp.getSolver("HiGHS", msg=False, timeLimitSeconds=180.0)


def ComputeRecourseUpperBound(unmetPenalty, trainingScenarios):
  """Compute a safe recourse upper bound for the CVaR variable."""
  # Return the penalty times the total maximum scenario demand plus a slack.
  return float(unmetPenalty * np.sum(np.max(trainingScenarios, axis=0)) + 10.0)


def ExtractFirstStageCost(costMatrix, rebalanceVariables, depotVariables, stationCount, depotSourcingCost):
  """Extract the first-stage total cost from solved variables."""
  # Initialize the running first-stage cost accumulator.
  totalFirstStageCost = 0.0
  # Iterate over each origin station index.
  for i in range(stationCount):
    # Iterate over each destination station index.
    for j in range(stationCount):
      # Skip the trivial self-loop entries.
      if (i == j):
        # Continue to the next destination station.
        continue
      # Accumulate the inter-station rebalancing cost.
      totalFirstStageCost += costMatrix[i][j] * float(rebalanceVariables[i][j].value() or 0.0)
    # Accumulate the depot sourcing cost for the current station.
    totalFirstStageCost += depotSourcingCost * float(depotVariables[i].value() or 0.0)
  # Return the accumulated first-stage cost.
  return float(totalFirstStageCost)


def ExtractNetPositioningVector(rebalanceVariables, depotVariables, stationCount):
  """Extract the net positioning vector from solved variables."""
  # Initialize the net positioning vector as zeros.
  netPositioningVector = np.zeros(stationCount)
  # Iterate over each station index.
  for i in range(stationCount):
    # Initialize the net inflow accumulator for the current station.
    netInflow = float(depotVariables[i].value() or 0.0)
    # Iterate over each counterpart station index.
    for j in range(stationCount):
      # Skip the trivial self-loop entries.
      if (i == j):
        # Continue to the next counterpart station.
        continue
      # Add the inter-station inflow from station j into station i.
      netInflow += float(rebalanceVariables[j][i].value() or 0.0)
      # Subtract the inter-station outflow from station i into station j.
      netInflow -= float(rebalanceVariables[i][j].value() or 0.0)
    # Assign the computed net inflow to the positioning vector.
    netPositioningVector[i] = netInflow
  # Return the net positioning vector.
  return netPositioningVector


def SolveExpectedValueModel(stationCount, costMatrix, meanDemandVector, capacityVector,
                            initialInventoryVector, unmetPenalty, depotSourcingCost):
  """Build and solve the deterministic Expected Value baseline."""
  # Initialize the pulp problem instance for the EV baseline.
  expectedValueModel = pulp.LpProblem("ExpectedValueBaseline", pulp.LpMinimize)
  # Compute a safe fleet upper bound from the total mean demand.
  fleetUpperBound = float(3.0 * np.sum(meanDemandVector) + 20.0)
  # Initialize the nested dictionary for first-stage rebalancing variables.
  rebalanceVariables = expectedValueModel.add_variable_dicts(
    "Rebalance", indices=(range(stationCount), range(stationCount)),
    lowBound=0.0, upBound=fleetUpperBound, cat="Continuous")
  # Initialize the dictionary for depot sourcing variables.
  depotVariables = expectedValueModel.add_variable_dicts(
    "Depot", indices=(range(stationCount),), lowBound=0.0,
    upBound=fleetUpperBound, cat="Continuous")
  # Initialize the dictionary for unmet demand variables.
  unmetUpperBound = float(3.0 * np.max(capacityVector) + 20.0)
  unmetVariables = expectedValueModel.add_variable_dicts(
    "Unmet", indices=(range(stationCount),), lowBound=0.0,
    upBound=unmetUpperBound, cat="Continuous")
  # Initialize the dictionary for net inflow expressions.
  netInflowExpressions = {}
  # Iterate over each station index.
  for stationIndex in range(stationCount):
    # Compute the net inflow expression including depot sourcing.
    netInflowExpressions[stationIndex] = depotVariables[stationIndex] + pulp.lpSum(
      [rebalanceVariables[j][stationIndex] for j in range(stationCount) if (j != stationIndex)]) - pulp.lpSum(
      [rebalanceVariables[stationIndex][j] for j in range(stationCount) if (j != stationIndex)])
  # Define the objective function combining rebalancing, depot, and unmet.
  objectiveFunction = pulp.lpSum(
    [costMatrix[i][j] * rebalanceVariables[i][j]
     for i in range(stationCount) for j in range(stationCount) if (i != j)])
  # Add the depot sourcing cost to the objective function.
  objectiveFunction += pulp.lpSum(
    [depotSourcingCost * depotVariables[stationIndex] for stationIndex in range(stationCount)])
  # Add the unmet demand penalty to the objective function.
  objectiveFunction += pulp.lpSum(
    [unmetPenalty * unmetVariables[stationIndex] for stationIndex in range(stationCount)])
  # Set the constructed objective function in the optimization model.
  expectedValueModel += objectiveFunction
  # Enforce the unmet demand constraint based on the mean demand.
  for stationIndex in range(stationCount):
    # Enforce the recourse lower bound using the mean demand less initial inventory.
    expectedValueModel += unmetVariables[stationIndex] >= meanDemandVector[stationIndex] - \
                          initialInventoryVector[stationIndex] - netInflowExpressions[stationIndex]
  # Solve the formulated expected value linear program.
  expectedValueModel.solve(InstantiateHighsSolver())
  # Extract the first-stage total cost from the solved variables.
  firstStageCost = ExtractFirstStageCost(costMatrix, rebalanceVariables, depotVariables,
                                         stationCount, depotSourcingCost)
  # Extract the optimal first-stage net positioning vector.
  netPositioningVector = ExtractNetPositioningVector(rebalanceVariables, depotVariables, stationCount)
  # Return the net positioning vector and the first-stage cost.
  return netPositioningVector, firstStageCost


def SolveSampleAverageApproximationModel(stationCount, costMatrix, trainingScenarios, capacityVector,
                                         initialInventoryVector, unmetPenalty, depotSourcingCost):
  """Build and solve the Sample Average Approximation baseline."""
  # Initialize the pulp problem instance for the SAA model.
  sampleAverageModel = pulp.LpProblem("SampleAverageApproximation", pulp.LpMinimize)
  # Compute a safe fleet upper bound from the total mean demand.
  fleetUpperBound = float(3.0 * np.sum(np.mean(trainingScenarios, axis=0)) + 20.0)
  # Initialize the nested dictionary for first-stage rebalancing variables.
  rebalanceVariables = sampleAverageModel.add_variable_dicts(
    "Rebalance", indices=(range(stationCount), range(stationCount)),
    lowBound=0.0, upBound=fleetUpperBound, cat="Continuous")
  # Initialize the dictionary for depot sourcing variables.
  depotVariables = sampleAverageModel.add_variable_dicts(
    "Depot", indices=(range(stationCount),), lowBound=0.0,
    upBound=fleetUpperBound, cat="Continuous")
  # Initialize the dictionary for second-stage unmet demand variables per scenario.
  unmetUpperBound = float(3.0 * np.max(capacityVector) + 20.0)
  unmetVariables = sampleAverageModel.add_variable_dicts(
    "Unmet", indices=(range(len(trainingScenarios)), range(stationCount)),
    lowBound=0.0, upBound=unmetUpperBound, cat="Continuous")
  # Initialize the dictionary for net inflow expressions.
  netInflowExpressions = {}
  # Iterate over each station index.
  for stationIndex in range(stationCount):
    # Compute the net inflow expression including depot sourcing.
    netInflowExpressions[stationIndex] = depotVariables[stationIndex] + pulp.lpSum(
      [rebalanceVariables[j][stationIndex] for j in range(stationCount) if (j != stationIndex)]) - pulp.lpSum(
      [rebalanceVariables[stationIndex][j] for j in range(stationCount) if (j != stationIndex)])
  # Define the objective function with rebalancing, depot, and empirical average recourse.
  objectiveFunction = pulp.lpSum(
    [costMatrix[i][j] * rebalanceVariables[i][j]
     for i in range(stationCount) for j in range(stationCount) if (i != j)])
  # Add the depot sourcing cost to the objective function.
  objectiveFunction += pulp.lpSum(
    [depotSourcingCost * depotVariables[stationIndex] for stationIndex in range(stationCount)])
  # Add the empirical average unmet demand penalty to the objective function.
  objectiveFunction += (1.0 / len(trainingScenarios)) * pulp.lpSum(
    [unmetPenalty * unmetVariables[s][i]
     for s in range(len(trainingScenarios)) for i in range(stationCount)])
  # Set the constructed objective function in the optimization model.
  sampleAverageModel += objectiveFunction
  # Enforce the recourse constraints for each training scenario and station.
  for scenarioIndex in range(len(trainingScenarios)):
    # Iterate over each station index.
    for stationIndex in range(stationCount):
      # Enforce the unmet demand lower bound for the current scenario.
      sampleAverageModel += unmetVariables[scenarioIndex][stationIndex] >= trainingScenarios[scenarioIndex][
        stationIndex] - initialInventoryVector[stationIndex] - netInflowExpressions[stationIndex]
  # Solve the formulated sample average approximation linear program.
  sampleAverageModel.solve(InstantiateHighsSolver())
  # Extract the first-stage total cost from the solved variables.
  firstStageCost = ExtractFirstStageCost(costMatrix, rebalanceVariables, depotVariables,
                                         stationCount, depotSourcingCost)
  # Extract the optimal first-stage net positioning vector.
  netPositioningVector = ExtractNetPositioningVector(rebalanceVariables, depotVariables, stationCount)
  # Return the net positioning vector and the first-stage cost.
  return netPositioningVector, firstStageCost


def SolveDistributionallyRobustModel(stationCount, costMatrix, trainingScenarios, capacityVector,
                                     initialInventoryVector, unmetPenalty, cvarAlpha, depotSourcingCost):
  """Build and solve the Distributionally Robust Optimization model."""
  # Initialize the pulp problem instance for the DRO model.
  robustModel = pulp.LpProblem("DistributionallyRobustRebalancing", pulp.LpMinimize)
  # Compute a safe fleet upper bound from the total mean demand.
  fleetUpperBound = float(3.0 * np.sum(np.mean(trainingScenarios, axis=0)) + 20.0)
  # Compute a safe recourse upper bound from the total penalty.
  recourseUpperBound = ComputeRecourseUpperBound(unmetPenalty, trainingScenarios)
  # Initialize the nested dictionary for first-stage rebalancing variables.
  rebalanceVariables = robustModel.add_variable_dicts(
    "Rebalance", indices=(range(stationCount), range(stationCount)),
    lowBound=0.0, upBound=fleetUpperBound, cat="Continuous")
  # Initialize the dictionary for depot sourcing variables.
  depotVariables = robustModel.add_variable_dicts(
    "Depot", indices=(range(stationCount),), lowBound=0.0,
    upBound=fleetUpperBound, cat="Continuous")
  # Initialize the dictionary for second-stage unmet demand variables per scenario.
  unmetUpperBound = float(3.0 * np.max(capacityVector) + 20.0)
  unmetVariables = robustModel.add_variable_dicts(
    "Unmet", indices=(range(len(trainingScenarios)), range(stationCount)),
    lowBound=0.0, upBound=unmetUpperBound, cat="Continuous")
  # Initialize the value-at-risk variable for the Rockafellar-Uryasev CVaR formulation.
  valueAtRisk = robustModel.add_variable("ValueAtRisk", lowBound=0.0,
                                         upBound=recourseUpperBound, cat="Continuous")
  # Initialize the dictionary for tail excess variables per scenario.
  tailExcessVariables = robustModel.add_variable_dicts(
    "TailExcess", indices=(range(len(trainingScenarios)),),
    lowBound=0.0, upBound=recourseUpperBound, cat="Continuous")
  # Initialize the dictionary for net inflow expressions.
  netInflowExpressions = {}
  # Iterate over each station index.
  for stationIndex in range(stationCount):
    # Compute the net inflow expression including depot sourcing.
    netInflowExpressions[stationIndex] = depotVariables[stationIndex] + pulp.lpSum(
      [rebalanceVariables[j][stationIndex] for j in range(stationCount) if (j != stationIndex)]) - pulp.lpSum(
      [rebalanceVariables[stationIndex][j] for j in range(stationCount) if (j != stationIndex)])
  # Define the objective function with rebalancing, depot, VaR, and tail excess.
  objectiveFunction = pulp.lpSum(
    [costMatrix[i][j] * rebalanceVariables[i][j]
     for i in range(stationCount) for j in range(stationCount) if (i != j)])
  # Add the depot sourcing cost to the objective function.
  objectiveFunction += pulp.lpSum(
    [depotSourcingCost * depotVariables[stationIndex] for stationIndex in range(stationCount)])
  # Add the value-at-risk anchor to the objective function.
  objectiveFunction += valueAtRisk
  # Add the scaled mean tail excess to the objective function.
  objectiveFunction += (1.0 / ((1.0 - cvarAlpha) * len(trainingScenarios))) * pulp.lpSum(
    [tailExcessVariables[s] for s in range(len(trainingScenarios))])
  # Set the constructed objective function in the optimization model.
  robustModel += objectiveFunction
  # Enforce the tail excess and recourse constraints for each scenario.
  for scenarioIndex in range(len(trainingScenarios)):
    # Compute the per-scenario recourse expression.
    scenarioRecourseExpression = pulp.lpSum(
      [unmetPenalty * unmetVariables[scenarioIndex][i] for i in range(stationCount)])
    # Enforce the tail excess lower bound linking recourse to the value-at-risk.
    robustModel += tailExcessVariables[scenarioIndex] >= scenarioRecourseExpression - valueAtRisk
    # Enforce the unmet demand lower bounds for each station in the scenario.
    for stationIndex in range(stationCount):
      # Enforce the unmet demand lower bound for the current scenario.
      robustModel += unmetVariables[scenarioIndex][stationIndex] >= trainingScenarios[scenarioIndex][
        stationIndex] - initialInventoryVector[stationIndex] - netInflowExpressions[stationIndex]
  # Solve the formulated distributionally robust linear program.
  robustModel.solve(InstantiateHighsSolver())
  # Extract the first-stage total cost from the solved variables.
  firstStageCost = ExtractFirstStageCost(costMatrix, rebalanceVariables, depotVariables,
                                         stationCount, depotSourcingCost)
  # Extract the optimal first-stage net positioning vector.
  netPositioningVector = ExtractNetPositioningVector(rebalanceVariables, depotVariables, stationCount)
  # Return the net positioning vector and the first-stage cost.
  return netPositioningVector, firstStageCost


def main():
  """Main entry point: solve EV / SAA / DRO for every cluster."""
  # Load the cluster network artifacts from the tagged JSON file.
  with open(Config.TaggedFileName("cluster_network_artifacts.json"), "r") as artifactFile:
    # Parse the JSON content into a dictionary.
    clusterArtifacts = json.load(artifactFile)
  # Load the station metadata to recover docking capacities.
  stationDataFrame = pd.read_csv(Config.TaggedFileName("station_metadata.csv"))
  # Retrieve the experiment configuration dictionary.
  configuration = Config.EXPERIMENT_CONFIGURATION
  # Initialize the list storing first-stage decision records.
  decisionRecords = []
  # Iterate over each cluster identifier in the sorted artifact keys.
  for clusterIdentifier in sorted(clusterArtifacts.keys(), key=lambda value: int(value)):
    # Retrieve the artifact dictionary for the current cluster.
    clusterArtifact = clusterArtifacts[clusterIdentifier]
    # Convert the cost matrix to a numpy array.
    costMatrix = np.array(clusterArtifact["CostMatrix"])
    # Convert the training scenarios to a numpy array.
    trainingScenarios = np.array(clusterArtifact["TrainingScenarios"])
    # Retrieve the station count for the current cluster.
    stationCount = int(clusterArtifact["ClusterSize"])
    # Recover the capacity vector for the stations in the current cluster.
    capacityVector = np.array(clusterArtifact["Capacity"], dtype=float)
    # Recover the initial inventory vector for the stations in the current cluster.
    initialInventoryVector = np.array(clusterArtifact["InitialInventory"], dtype=float)
    # Compute the mean demand vector across training scenarios.
    meanDemandVector = np.mean(trainingScenarios, axis=0)
    # Solve the expected value baseline model.
    evPositioning, evCost = SolveExpectedValueModel(
      stationCount, costMatrix, meanDemandVector, capacityVector,
      initialInventoryVector, configuration["UnmetDemandPenalty"],
      configuration["DepotSourcingCost"])
    # Solve the sample average approximation model.
    saaPositioning, saaCost = SolveSampleAverageApproximationModel(
      stationCount, costMatrix, trainingScenarios, capacityVector,
      initialInventoryVector, configuration["UnmetDemandPenalty"],
      configuration["DepotSourcingCost"])
    # Solve the distributionally robust optimization model.
    droPositioning, droCost = SolveDistributionallyRobustModel(
      stationCount, costMatrix, trainingScenarios, capacityVector,
      initialInventoryVector, configuration["UnmetDemandPenalty"],
      configuration["CvarAlpha"], configuration["DepotSourcingCost"])
    # Append the expected value decision record.
    decisionRecords.append({"ClusterId"      : int(clusterIdentifier), "Method": "EV",
                            "RebalancingCost": float(evCost),
                            "NetPositioning" : evPositioning.tolist()})
    # Append the sample average approximation decision record.
    decisionRecords.append({"ClusterId"      : int(clusterIdentifier), "Method": "SAA",
                            "RebalancingCost": float(saaCost),
                            "NetPositioning" : saaPositioning.tolist()})
    # Append the distributionally robust optimization decision record.
    decisionRecords.append({"ClusterId"      : int(clusterIdentifier), "Method": "DRO",
                            "RebalancingCost": float(droCost),
                            "NetPositioning" : droPositioning.tolist()})
    # Print a progress message for the current cluster.
    print("Cluster %s solved: EV cost=%.2f, SAA cost=%.2f, DRO cost=%.2f." % (
      clusterIdentifier, evCost, saaCost, droCost))
  # Convert the decision records to a pandas dataframe.
  decisionDataFrame = pd.DataFrame(decisionRecords)
  # Save the first-stage decisions to a tagged CSV file.
  decisionDataFrame.to_csv(Config.TaggedFileName("first_stage_decisions.csv"), index=False)
  # Print a confirmation message summarizing the optimization phase.
  print("Phase 2 complete [%s]: solved EV, SAA, and DRO models for %d clusters." % (
    Config.RUN_TAG, len(clusterArtifacts)))


# Execute the main function when the script runs as the entry point.
if __name__ == "__main__":
  # Invoke the main entry point.
  main()
