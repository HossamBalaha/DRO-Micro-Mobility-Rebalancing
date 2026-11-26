"""
Phase1RealData.py
===================

Phase 1 - Real-data preparation: cleaning, station catalogue, spatial
clustering, and bootstrap demand scenarios.  When the active range spans
multiple months, MERGES all monthly trip CSVs into a single cleaned
dataframe so downstream analysis captures seasonal variation.

OUTPUTS (all prefixed with `{City}_{YearMonthRange}{RunSuffix}_`)
  * station_metadata.csv
  * cluster_metadata.csv
  * empirical_demand.csv
  * training_pool_trips.csv
  * testing_pool_trips.csv
  * cluster_network_artifacts.json

Note: When RunMulti.py sets DRO_RUN_SUFFIX=_run{i}, all outputs are
uniquely named from the start (e.g. Divvy_202608_run1_station_metadata.csv),
eliminating cross-run overwrite.
"""

# Import the pandas library for tabular data manipulation.
import pandas as pd
# Import the numpy library for numerical array operations.
import numpy as np
# Import the json library for serializing metadata artifacts.
import json
# Import the os library for file path construction.
import os
# Import the shared configuration module for global knobs and paths.
import Config
# Import the phase zero download module to locate the active city CSVs.
import Phase0Download


def LoadAndCleanTripData_Legacy(tripCsvPaths):
  """Load and clean the raw trip data from one or more CSV files."""
  # Accept either a single string path or a list of paths.
  if (isinstance(tripCsvPaths, str)):
    # Wrap the single path in a list for uniform handling.
    tripCsvPaths = [tripCsvPaths]
  # Initialize the list of dataframes to concatenate.
  dataFrames = []
  # Iterate over each CSV path.
  for tripCsvPath in tripCsvPaths:
    # Print a progress message for the current CSV.
    print("  Loading CSV: %s" % os.path.basename(tripCsvPath))
    # Read the raw trip data from the specified CSV file path.
    rawTripDataFrame = pd.read_csv(tripCsvPath)
    # Append the dataframe to the list.
    dataFrames.append(rawTripDataFrame)
  # Concatenate all monthly dataframes into a single trip dataframe.
  rawTripDataFrame = pd.concat(dataFrames, ignore_index=True)
  # Drop rows missing the critical station identifier columns.
  cleanedDataFrame = rawTripDataFrame.dropna(subset=["start_station_id", "end_station_id"])
  # Drop rows missing the critical coordinate columns.
  cleanedDataFrame = cleanedDataFrame.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])
  # Parse the started_at column as datetime objects.
  cleanedDataFrame["started_at"] = pd.to_datetime(cleanedDataFrame["started_at"], errors="coerce")
  # Parse the ended_at column as a datetime object.
  cleanedDataFrame["ended_at"] = pd.to_datetime(cleanedDataFrame["ended_at"], errors="coerce")
  # Drop rows with unparseable timestamps.
  cleanedDataFrame = cleanedDataFrame.dropna(subset=["started_at", "ended_at"])
  # Compute the trip duration in seconds.
  cleanedDataFrame["trip_duration_seconds"] = (
    cleanedDataFrame["ended_at"] - cleanedDataFrame["started_at"]).dt.total_seconds()
  # Filter out trips with durations less than sixty seconds.
  cleanedDataFrame = cleanedDataFrame[cleanedDataFrame["trip_duration_seconds"] >= 60.0]
  # Filter out trips with implausibly long durations exceeding twenty-four hours.
  cleanedDataFrame = cleanedDataFrame[cleanedDataFrame["trip_duration_seconds"] <= 86400.0]
  # Reset the index of the cleaned dataframe.
  cleanedDataFrame = cleanedDataFrame.reset_index(drop=True)
  # Print a summary of the cleaned trip data.
  print("Loaded and cleaned %d real trips from %d monthly CSV(s)." % (
    len(cleanedDataFrame), len(tripCsvPaths)))
  # Return the cleaned trip dataframe.
  return cleanedDataFrame


def LoadAndCleanTripData(tripCsvPaths):
  """Load and clean the raw trip data from one or more CSV files."""
  # Accept either a single string path or a list of paths.
  if (isinstance(tripCsvPaths, str)):
    # Wrap the single path in a list for uniform handling.
    tripCsvPaths = [tripCsvPaths]
  # Initialize the list of dataframes to concatenate.
  dataFrames = []
  # Iterate over each CSV path.
  for tripCsvPath in tripCsvPaths:
    # Print a progress message for the current CSV.
    print("  Loading CSV: %s" % os.path.basename(tripCsvPath))
    # Read the raw trip data with explicit string types for station IDs to prevent mixed-type sorting errors.
    rawTripDataFrame = pd.read_csv(tripCsvPath, dtype={"start_station_id": str, "end_station_id": str},
                                   low_memory=False)
    # Append the dataframe to the list.
    dataFrames.append(rawTripDataFrame)
  # Concatenate all monthly dataframes into a single trip dataframe.
  rawTripDataFrame = pd.concat(dataFrames, ignore_index=True)
  # Drop rows missing the critical station identifier columns.
  cleanedDataFrame = rawTripDataFrame.dropna(subset=["start_station_id", "end_station_id"])
  # Filter out any rows where station IDs are the literal string "nan" or empty.
  cleanedDataFrame = cleanedDataFrame[~cleanedDataFrame["start_station_id"].str.lower().isin(["nan", ""])]
  cleanedDataFrame = cleanedDataFrame[~cleanedDataFrame["end_station_id"].str.lower().isin(["nan", ""])]
  # Drop rows missing the critical coordinate columns.
  cleanedDataFrame = cleanedDataFrame.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])
  # Parse the started_at column as datetime objects.
  cleanedDataFrame["started_at"] = pd.to_datetime(cleanedDataFrame["started_at"], errors="coerce")
  # Parse the ended_at column as a datetime object.
  cleanedDataFrame["ended_at"] = pd.to_datetime(cleanedDataFrame["ended_at"], errors="coerce")
  # Drop rows with unparseable timestamps.
  cleanedDataFrame = cleanedDataFrame.dropna(subset=["started_at", "ended_at"])
  # Compute the trip duration in seconds.
  cleanedDataFrame["trip_duration_seconds"] = (
    cleanedDataFrame["ended_at"] - cleanedDataFrame["started_at"]).dt.total_seconds()
  # Filter out trips with durations less than sixty seconds.
  cleanedDataFrame = cleanedDataFrame[cleanedDataFrame["trip_duration_seconds"] >= 60.0]
  # Filter out trips with implausibly long durations exceeding twenty-four hours.
  cleanedDataFrame = cleanedDataFrame[cleanedDataFrame["trip_duration_seconds"] <= 86400.0]
  # Reset the index of the cleaned dataframe.
  cleanedDataFrame = cleanedDataFrame.reset_index(drop=True)
  # Print a summary of the cleaned trip data.
  print("Loaded and cleaned %d real trips from %d monthly CSV(s)." % (
    len(cleanedDataFrame), len(tripCsvPaths)))
  # Return the cleaned trip dataframe.
  return cleanedDataFrame


def BuildStationCatalogue(cleanedTripDataFrame, configuration, randomGenerator):
  """Build the per-station catalogue with capacity and initial inventory estimates."""
  # Initialize the list storing per-station records.
  stationRecords = []
  # Group the trips by start station identifier.
  startGrouped = cleanedTripDataFrame.groupby("start_station_id")
  # Group the trips by end station identifier.
  endGrouped = cleanedTripDataFrame.groupby("end_station_id")
  # Build a set of all station identifiers appearing as either start or end.
  allStationIds = set(startGrouped.groups.keys()) | set(endGrouped.groups.keys())
  # Iterate over each station identifier.
  for stationIndex, stationId in enumerate(sorted(allStationIds)):
    # Retrieve the start-side trip subset for the current station.
    startSubset = startGrouped.get_group(stationId) if (stationId in startGrouped.groups) else None
    # Retrieve the end-side trip subset for the current station.
    endSubset = endGrouped.get_group(stationId) if (stationId in endGrouped.groups) else None
    # Initialize the latitude accumulator.
    latitudeValues = []
    # Initialize the longitude accumulator.
    longitudeValues = []
    # Collect start-side coordinates when available.
    if (startSubset is not None):
      # Extend the latitude accumulator with start-side latitudes.
      latitudeValues.extend(startSubset["start_lat"].tolist())
      # Extend the longitude accumulator with start-side longitudes.
      longitudeValues.extend(startSubset["start_lng"].tolist())
    # Collect end-side coordinates when available.
    if (endSubset is not None):
      # Extend the latitude accumulator with end-side latitudes.
      latitudeValues.extend(endSubset["end_lat"].tolist())
      # Extend the longitude accumulator with end-side longitudes.
      longitudeValues.extend(endSubset["end_lng"].tolist())
    # Continue to the next station when coordinates are missing.
    if (len(latitudeValues) == 0):
      # Continue to the next station.
      continue
    # Compute the median latitude to suppress GPS jitter.
    stationLatitude = float(np.median(latitudeValues))
    # Compute the median longitude to suppress GPS jitter.
    stationLongitude = float(np.median(longitudeValues))
    # Compute the trip volume as the sum of start and end counts.
    tripVolume = int(
      (0 if (startSubset is None) else len(startSubset)) + (0 if (endSubset is None) else len(endSubset)))
    # Continue to the next station when the trip volume is too small.
    if (tripVolume < 10):
      # Continue to the next station.
      continue
    # Estimate the docking capacity from the trip volume via a clipped square root.
    dockingCapacity = int(min(35, max(15, round(2.0 * np.sqrt(tripVolume)))))
    # Compute the popularity weight as the raw trip volume.
    popularityWeight = float(tripVolume)
    # Sample a deterministic initial inventory fraction between thirty and seventy percent.
    initialInventoryFraction = float(randomGenerator.uniform(0.30, 0.70))
    # Compute the initial morning inventory as the rounded fraction of capacity.
    initialInventory = float(round(dockingCapacity * initialInventoryFraction))
    # Retrieve the station name when available.
    stationName = ""
    # Populate the station name from the start subset when available.
    if (startSubset is not None and len(startSubset) > 0):
      # Assign the first station name encountered.
      stationName = str(startSubset.iloc[0]["start_station_name"])
    # Append the station record to the running list.
    stationRecords.append({"StationId"       : str(stationId), "StationName": stationName,
                           "Latitude"        : stationLatitude, "Longitude": stationLongitude,
                           "Capacity"        : dockingCapacity, "Popularity": popularityWeight,
                           "InitialInventory": initialInventory, "TripVolume": tripVolume})
  # Convert the station records to a pandas dataframe.
  stationDataFrame = pd.DataFrame(stationRecords)
  # Normalize the popularity weights by the maximum trip volume.
  maxPopularity = float(stationDataFrame["Popularity"].max())
  # Divide the popularity weights by the maximum for normalization.
  stationDataFrame["PopularityNormalized"] = stationDataFrame["Popularity"] / maxPopularity
  # Return the station dataframe.
  return stationDataFrame


def SelectTopStations(stationDataFrame, configuration):
  """Select the top-N stations by trip volume."""
  # Retrieve the number of stations to select from the configuration.
  numberOfStations = configuration["NumberOfStationsToSelect"]
  # Sort the station dataframe by descending trip volume.
  sortedDataFrame = stationDataFrame.sort_values("TripVolume", ascending=False).reset_index(drop=True)
  # Take the top stations by trip volume.
  selectedDataFrame = sortedDataFrame.head(numberOfStations).reset_index(drop=True)
  # Print a summary of the selected stations.
  print("Selected top %d stations by trip volume (range %d to %d monthly trips)." % (
    len(selectedDataFrame),
    selectedDataFrame["TripVolume"].min(),
    selectedDataFrame["TripVolume"].max()))
  # Return the selected station dataframe.
  return selectedDataFrame


def ComputeMortonCode(latitude, longitude, latMin, latMax, lngMin, lngMax, bitDepth=16):
  """Compute the Morton space-filling curve code for sorting stations."""
  # Normalize the latitude into the unit interval.
  normLat = (latitude - latMin) / max(1e-9, latMax - latMin)
  # Normalize the longitude into the unit interval.
  normLng = (longitude - lngMin) / max(1e-9, lngMax - lngMin)
  # Clamp the normalized latitude into the unit interval.
  normLat = min(1.0, max(0.0, normLat))
  # Clamp the normalized longitude into the unit interval.
  normLng = min(1.0, max(0.0, normLng))
  # Scale the normalized latitude to the integer grid.
  intLat = int(normLat * ((1 << bitDepth) - 1))
  # Scale the normalized longitude to the integer grid.
  intLng = int(normLng * ((1 << bitDepth) - 1))
  # Initialize the Morton code accumulator.
  mortonCode = 0
  # Iterate over each bit position.
  for bitIndex in range(bitDepth):
    # Extract the latitude bit at the current position.
    latBit = (intLat >> bitIndex) & 1
    # Extract the longitude bit at the current position.
    lngBit = (intLng >> bitIndex) & 1
    # Interleave the bits into the Morton code.
    mortonCode |= (latBit << (2 * bitIndex)) | (lngBit << (2 * bitIndex + 1))
  # Return the Morton code.
  return mortonCode


def PartitionStationsIntoClusters(stationDataFrame, configuration):
  """Partition stations into spatially coherent clusters via Morton codes."""
  # Compute the latitude bounds of the selected stations.
  latMin = float(stationDataFrame["Latitude"].min())
  latMax = float(stationDataFrame["Latitude"].max())
  # Compute the longitude bounds of the selected stations.
  lngMin = float(stationDataFrame["Longitude"].min())
  lngMax = float(stationDataFrame["Longitude"].max())
  # Make a copy of the station dataframe.
  stationDataFrame = stationDataFrame.copy()
  # Assign the Morton code column.
  stationDataFrame["MortonCode"] = stationDataFrame.apply(
    lambda row: ComputeMortonCode(row["Latitude"], row["Longitude"],
                                  latMin, latMax, lngMin, lngMax), axis=1)
  # Sort the stations by ascending Morton code for spatial coherence.
  stationDataFrame = stationDataFrame.sort_values("MortonCode").reset_index(drop=True)
  # Initialize the cluster identifier counter.
  clusterIdentifier = 0
  # Initialize the cluster assignment list.
  clusterAssignment = []
  # Initialize the cluster metadata list.
  clusterMetadata = []
  # Build the ordered list of band names repeated by their cluster counts.
  bandSequence = []
  # Iterate over the ordered band names.
  for bandName in ["Small", "Medium", "Large"]:
    # Append the band name once per cluster in the band.
    for _ in range(configuration["ClusterBandCounts"][bandName]):
      # Append the band name.
      bandSequence.append(bandName)
  # Initialize the running station index.
  runningIndex = 0
  # Iterate over each cluster position in the band sequence.
  for clusterPosition, bandName in enumerate(bandSequence):
    # Advance the cluster identifier.
    clusterIdentifier += 1
    # Retrieve the target size for the current band.
    targetSize = configuration["ClusterSizeBands"][bandName]
    # Slice the sorted stations for the current cluster.
    clusterSlice = stationDataFrame.iloc[runningIndex:runningIndex + targetSize]
    # Compute the anchor latitude as the cluster median.
    anchorLatitude = float(clusterSlice["Latitude"].median())
    # Compute the anchor longitude as the cluster median.
    anchorLongitude = float(clusterSlice["Longitude"].median())
    # Append the cluster metadata record.
    clusterMetadata.append({"ClusterId"      : clusterIdentifier, "Band": bandName,
                            "Size"           : int(len(clusterSlice)),
                            "AnchorLatitude" : anchorLatitude,
                            "AnchorLongitude": anchorLongitude})
    # Assign the cluster identifier to each station in the slice.
    for _ in range(len(clusterSlice)):
      # Append the cluster identifier.
      clusterAssignment.append(clusterIdentifier)
    # Advance the running station index.
    runningIndex += targetSize
  # Truncate the station dataframe to the assigned stations.
  stationDataFrame = stationDataFrame.iloc[:runningIndex].copy()
  # Assign the cluster identifier column.
  stationDataFrame["ClusterId"] = clusterAssignment
  # Assign the band label column.
  stationDataFrame["Band"] = stationDataFrame["ClusterId"].map(
    {m["ClusterId"]: m["Band"] for m in clusterMetadata})
  # Return the partitioned station dataframe and the cluster metadata.
  return stationDataFrame, clusterMetadata


def ComputeHaversineDistanceKilometers(originCoordinates, destinationCoordinates):
  """Compute the Haversine great-circle distance in kilometers."""
  # Define the mean Earth radius in kilometers.
  earthRadiusKilometers = 6371.0088
  # Convert the latitude difference to radians.
  latitudeDifference = np.radians(destinationCoordinates[0] - originCoordinates[0])
  # Convert the longitude difference to radians.
  longitudeDifference = np.radians(destinationCoordinates[1] - originCoordinates[1])
  # Compute the haversine latitude factor.
  latitudeFactor = np.sin(latitudeDifference / 2.0) ** 2
  # Compute the haversine longitude factor with cosine correction.
  longitudeFactor = (np.cos(np.radians(originCoordinates[0])) *
                     np.cos(np.radians(destinationCoordinates[0])) *
                     np.sin(longitudeDifference / 2.0) ** 2)
  # Combine the factors into the haversine argument.
  haversineArgument = latitudeFactor + longitudeFactor
  # Compute the central angle via the two-argument arctangent.
  centralAngle = 2.0 * np.arctan2(np.sqrt(haversineArgument), np.sqrt(1.0 - haversineArgument))
  # Return the great-circle distance in kilometers.
  return earthRadiusKilometers * centralAngle


def ComputeNetworkDistanceMatrix(stationSubsetDataFrame, rebalancingCostPerKilometer, manhattanDetourFactor):
  """Compute the road-proxy distance and cost matrices per cluster."""
  # Extract the station coordinates as a two-dimensional numpy array.
  coordinatesArray = stationSubsetDataFrame[["Latitude", "Longitude"]].to_numpy()
  # Retrieve the number of stations in the current cluster.
  stationCount = len(stationSubsetDataFrame)
  # Initialize a zeroed distance matrix.
  distanceMatrix = np.zeros((stationCount, stationCount))
  # Initialize a zeroed cost matrix.
  costMatrix = np.zeros((stationCount, stationCount))
  # Iterate over each origin station index.
  for originIndex in range(stationCount):
    # Iterate over each destination station index.
    for destinationIndex in range(stationCount):
      # Skip computation when the origin and destination coincide.
      if (originIndex == destinationIndex):
        # Continue to the next destination station.
        continue
      # Compute the Haversine great-circle distance.
      linkDistance = ComputeHaversineDistanceKilometers(coordinatesArray[originIndex],
                                                        coordinatesArray[destinationIndex])
      # Apply the Manhattan detour factor for the urban road grid.
      roadDistance = linkDistance * manhattanDetourFactor
      # Assign the road distance to the matrix.
      distanceMatrix[originIndex, destinationIndex] = roadDistance
      # Assign the rebalancing cost as distance times unit cost.
      costMatrix[originIndex, destinationIndex] = roadDistance * rebalancingCostPerKilometer
  # Return the distance and cost matrices.
  return distanceMatrix, costMatrix


def SplitTripsIntoTrainTestPools(cleanedTripDataFrame, configuration, randomGenerator):
  """Split trips into weekday training and weekend testing pools."""
  # Compute the day of week for each trip from the started_at timestamp.
  dayOfWeekSeries = cleanedTripDataFrame["started_at"].dt.dayofweek
  # Select weekday trips (Monday through Friday) as the training pool.
  trainingPool = cleanedTripDataFrame[dayOfWeekSeries < 5].reset_index(drop=True)
  # Select weekend trips (Saturday and Sunday) as the testing pool.
  testingPool = cleanedTripDataFrame[dayOfWeekSeries >= 5].reset_index(drop=True)
  # Shuffle both pools with the seeded generator to remove ordering bias.
  trainingPool = trainingPool.iloc[randomGenerator.permutation(len(trainingPool))].reset_index(drop=True)
  # Shuffle the testing pool.
  testingPool = testingPool.iloc[randomGenerator.permutation(len(testingPool))].reset_index(drop=True)
  # Print a summary of the weekday and weekend split.
  print(
    "Split trips into %d weekday (training) and %d weekend (testing) pool trips for genuine distributional shift." % (
      len(trainingPool), len(testingPool)))
  # Return the training and testing pools.
  return trainingPool, testingPool


def BootstrapDemandScenario(tripPool, stationIdToIndexMap, stationCount, windowSize, randomGenerator):
  """Bootstrap a per-station demand scenario from a trip pool."""
  # Sample a window of trips with replacement from the pool.
  sampledIndices = randomGenerator.integers(0, len(tripPool), size=windowSize)
  # Extract the sampled start station identifiers.
  sampledStartIds = tripPool.iloc[sampledIndices]["start_station_id"].tolist()
  # Initialize the demand vector to zeros.
  demandVector = np.zeros(stationCount)
  # Iterate over each sampled start station identifier.
  for stationId in sampledStartIds:
    # Retrieve the station index from the map.
    stationIndex = stationIdToIndexMap.get(str(stationId))
    # Skip stations outside the selected catalogue.
    if (stationIndex is None):
      # Continue to the next sampled station.
      continue
    # Increment the demand at the resolved station index.
    demandVector[stationIndex] += 1.0
  # Return the demand vector.
  return demandVector


def GenerateRealDemandScenarios(stationSubsetDataFrame, trainingPool, testingPool, configuration, randomGenerator):
  """Generate the training and out-of-sample scenario matrices per cluster."""
  # Retrieve the number of stations in the current cluster.
  stationCount = len(stationSubsetDataFrame)
  # Build the station identifier to local index map.
  stationIdToIndexMap = {str(row["StationId"]): idx for idx, row in stationSubsetDataFrame.iterrows()}
  # Compute the bootstrap window size from the configuration.
  windowSize = int(configuration["ScenarioWindowSizeFraction"] * len(trainingPool))
  # Initialize the training scenario matrix.
  trainingScenarios = np.zeros((configuration["TrainingScenarios"], stationCount))
  # Iterate over each training scenario index.
  for scenarioIndex in range(configuration["TrainingScenarios"]):
    # Bootstrap a training scenario demand vector.
    trainingScenarios[scenarioIndex] = BootstrapDemandScenario(
      trainingPool, stationIdToIndexMap, stationCount, windowSize, randomGenerator)
  # Initialize the out-of-sample scenario matrix.
  outOfSampleScenarios = np.zeros((configuration["OutOfSampleScenarios"], stationCount))
  # Iterate over each out-of-sample scenario index.
  for scenarioIndex in range(configuration["OutOfSampleScenarios"]):
    # Bootstrap a testing scenario demand vector.
    outOfSampleScenarios[scenarioIndex] = BootstrapDemandScenario(
      testingPool, stationIdToIndexMap, stationCount, windowSize, randomGenerator)
  # Return the training and out-of-sample scenario matrices.
  return trainingScenarios, outOfSampleScenarios


def AggregateOriginDestinationDemand(stationDataFrame, cleanedTripDataFrame, configuration):
  """Aggregate the empirical origin-destination demand distribution per cluster."""
  # Initialize the list storing the empirical demand tuples.
  empiricalDemandRecords = []
  # Iterate over each unique cluster identifier.
  for clusterIdentifier in sorted(stationDataFrame["ClusterId"].unique()):
    # Collect the station identifiers belonging to the current cluster.
    clusterStationIds = set(
      stationDataFrame[stationDataFrame["ClusterId"] == clusterIdentifier]["StationId"].astype(str).tolist())
    # Subset the cleaned trips to those with both endpoints in the current cluster.
    clusterTrips = cleanedTripDataFrame[
      cleanedTripDataFrame["start_station_id"].astype(str).isin(clusterStationIds) &
      cleanedTripDataFrame["end_station_id"].astype(str).isin(clusterStationIds)]
    # Group the cluster trips by start and end station identifiers.
    groupedTrips = clusterTrips.groupby(["start_station_id", "end_station_id"]).size().reset_index(name="TripCount")
    # Compute the total trip count for normalization.
    totalTripCount = float(groupedTrips["TripCount"].sum())
    # Continue to the next cluster when no intra-cluster trips exist.
    if (totalTripCount == 0):
      # Continue to the next cluster.
      continue
    # Iterate over each grouped trip row.
    for _, row in groupedTrips.iterrows():
      # Compute the empirical probability.
      empiricalProbability = float(row["TripCount"] / totalTripCount)
      # Append the empirical demand record.
      empiricalDemandRecords.append({
        "OriginStation"       : str(row["start_station_id"]),
        "DestinationStation"  : str(row["end_station_id"]),
        "ClusterId"           : int(clusterIdentifier),
        "EmpiricalProbability": empiricalProbability,
        "TripCount"           : int(row["TripCount"])})
  # Convert the empirical demand records to a pandas dataframe.
  empiricalDemandDataFrame = pd.DataFrame(empiricalDemandRecords)
  # Return the empirical demand dataframe.
  return empiricalDemandDataFrame


def main():
  """Main entry point: build all real-data artifacts for the active run."""
  # Instantiate a fresh random generator from the shared configuration.
  randomGenerator = np.random.default_rng(Config.GLOBAL_RANDOM_SEED)
  # Locate the list of trip CSVs for the active city across all year-months.
  tripCsvPaths = Phase0Download.LocateActiveCityTripCsvs()
  # Load and clean the merged raw trip data.
  cleanedTripDataFrame = LoadAndCleanTripData(tripCsvPaths)
  # Retrieve the experiment configuration dictionary.
  configuration = Config.EXPERIMENT_CONFIGURATION
  # Build the station catalogue from the cleaned trip data.
  stationDataFrame = BuildStationCatalogue(cleanedTripDataFrame, configuration, randomGenerator)
  # Select the top stations by trip volume.
  stationDataFrame = SelectTopStations(stationDataFrame, configuration)
  # Partition the selected stations into spatially coherent clusters.
  stationDataFrame, clusterMetadata = PartitionStationsIntoClusters(stationDataFrame, configuration)
  # Convert the cluster metadata to a pandas dataframe.
  clusterDataFrame = pd.DataFrame(clusterMetadata)
  # Aggregate the empirical origin-destination demand distribution.
  empiricalDemandDataFrame = AggregateOriginDestinationDemand(stationDataFrame, cleanedTripDataFrame, configuration)
  # Save the station catalogue to a tagged CSV file.
  stationDataFrame.to_csv(Config.TaggedFileName("station_metadata.csv"), index=False)
  # Save the cluster metadata to a tagged CSV file.
  clusterDataFrame.to_csv(Config.TaggedFileName("cluster_metadata.csv"), index=False)
  # Save the empirical demand distribution to a tagged CSV file.
  empiricalDemandDataFrame.to_csv(Config.TaggedFileName("empirical_demand.csv"), index=False)
  # Split the cleaned trips into training and testing pools.
  trainingPool, testingPool = SplitTripsIntoTrainTestPools(cleanedTripDataFrame, configuration, randomGenerator)
  # Save the training pool trip identifiers for reproducibility.
  trainingPool[["ride_id", "start_station_id", "started_at"]].to_csv(
    Config.TaggedFileName("training_pool_trips.csv"), index=False)
  # Save the testing pool trip identifiers for reproducibility.
  testingPool[["ride_id", "start_station_id", "started_at"]].to_csv(
    Config.TaggedFileName("testing_pool_trips.csv"), index=False)
  # Initialize the cluster network artifact dictionary.
  clusterNetworkArtifacts = {}
  # Iterate over each unique cluster identifier.
  for clusterIdentifier in sorted(stationDataFrame["ClusterId"].unique()):
    # Subset the station dataframe to the current cluster.
    clusterStationSubset = stationDataFrame[stationDataFrame["ClusterId"] == clusterIdentifier].reset_index(drop=True)
    # Compute the distance and cost matrices for the current cluster.
    distanceMatrix, costMatrix = ComputeNetworkDistanceMatrix(clusterStationSubset,
                                                              configuration["RebalancingCostPerKilometer"],
                                                              configuration["ManhattanDetourFactor"])
    # Generate the training and out-of-sample scenario matrices.
    trainingScenarios, outOfSampleScenarios = GenerateRealDemandScenarios(
      clusterStationSubset, trainingPool, testingPool, configuration, randomGenerator)
    # Store the matrices and scenarios in the artifact dictionary.
    clusterNetworkArtifacts[str(clusterIdentifier)] = {
      "DistanceMatrix"      : distanceMatrix.tolist(),
      "CostMatrix"          : costMatrix.tolist(),
      "TrainingScenarios"   : trainingScenarios.tolist(),
      "OutOfSampleScenarios": outOfSampleScenarios.tolist(),
      "StationIds"          : clusterStationSubset["StationId"].astype(str).tolist(),
      "ClusterSize"         : int(len(clusterStationSubset)),
      "InitialInventory"    : clusterStationSubset["InitialInventory"].astype(float).tolist(),
      "Capacity"            : clusterStationSubset["Capacity"].astype(float).tolist()}
  # Save the cluster network artifacts as a tagged JSON file.
  with open(Config.TaggedFileName("cluster_network_artifacts.json"), "w") as artifactFile:
    # Dump the artifact dictionary to the JSON file.
    json.dump(clusterNetworkArtifacts, artifactFile)
  # Print a confirmation message summarizing the real-data phase.
  print(
    "Phase 1 complete [%s]: built %d stations across %d clusters from real %s %s trip data with bootstrapped demand scenarios." % (
      Config.RUN_TAG, len(stationDataFrame), len(clusterDataFrame),
      configuration["ActiveCityKey"], Config.YEAR_MONTH_TAG))


# Execute the main function when the script runs as the entry point.
if __name__ == "__main__":
  # Invoke the main entry point.
  main()
