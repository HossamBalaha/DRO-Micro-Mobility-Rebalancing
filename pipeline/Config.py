"""
Config.py
=========

Central configuration module for the DRO micro-mobility rebalancing pipeline (v5).

PURPOSE
-------
Defines every global knob, path, and data source that the rest of the
pipeline reads at runtime.  Supports multi-month ranges, multi-run
mean +/- std reporting with Run1/Run2/RunN folder organization, and
auto-generated .txt/.tex interpretation files.

KEY FIX IN v5
-------------
Adds the RUN_SUFFIX env var.  When RunMulti.py sets DRO_RUN_SUFFIX=_run{i},
every Phase 1-5 subprocess writes its outputs as Divvy_202608_run{i}_*.* 
files DIRECTLY to data/ and data/figures/.  This eliminates the v4 bug
where MoveRunOutputsToPerRunFolder ran after each phase batch and re-cleared
the Run{i} folders.  Each run's outputs are now uniquely named from the
start, so the folder organization step just COPIES files into the Run{i}/
folders without ever deleting prior runs' files.

USAGE
-----
  * Direct edit: change DEFAULT_CITY_KEY / DEFAULT_START_YEAR_MONTH /
    DEFAULT_END_YEAR_MONTH below.
  * CLI override: python RunAll.py --city JC --start-year-month 202501 \\
      --end-year-month 202512 --seed 42
  * Multi-run: python RunMulti.py --city Divvy --start-year-month 202601 \\
      --end-year-month 202612 --n-runs 5

READ BY
-------
Phase0Download.py, Phase1RealData.py, Phase2DroModel.py,
Phase3BaselinesOutOfSample.py, Phase4StatisticalValidation.py,
Phase5Visualization.py, Phase6MultiRunAggregator.py, Interpretation.py,
RunAll.py, RunMulti.py.
"""

# Import the os library for file system interactions and environment lookup.
import os
# Import the numpy library for numerical array operations.
import numpy as np

# Section 1 - Active run parameters.

# Define the default city used when no override is provided.
DEFAULT_CITY_KEY = "Divvy"
# Define the default start year-month (YYYYMM) of trip data to download.
DEFAULT_START_YEAR_MONTH = "202608"
# Define the default end year-month (YYYYMM) of trip data to download.
DEFAULT_END_YEAR_MONTH = "202608"
# Define the default global random seed for full reproducibility.
DEFAULT_RANDOM_SEED = 20260801

# Allow CLI / environment override of the active city for parametric runs.
ACTIVE_CITY_KEY = os.environ.get("DRO_ACTIVE_CITY", DEFAULT_CITY_KEY)
# Allow CLI / environment override of the start year-month for multi-month runs.
ACTIVE_START_YEAR_MONTH = os.environ.get("DRO_START_YEAR_MONTH", DEFAULT_START_YEAR_MONTH)
# Allow CLI / environment override of the end year-month for multi-month runs.
ACTIVE_END_YEAR_MONTH = os.environ.get("DRO_END_YEAR_MONTH", DEFAULT_END_YEAR_MONTH)
# Allow CLI / environment override of the global random seed for multi-run mode.
GLOBAL_RANDOM_SEED = int(os.environ.get("DRO_RANDOM_SEED", DEFAULT_RANDOM_SEED))
# Allow CLI / environment override of the per-run suffix (e.g. "_run1") for multi-run mode.
# When empty (the default), single-run mode is assumed and the run tag is "Divvy_202608".
# When set to "_run1" by RunMulti.py, the run tag becomes "Divvy_202608_run1", which
# means every Phase 1-5 subprocess writes its outputs with the unique per-run suffix
# already injected.  This eliminates the v4 bug of cross-run overwrite.
RUN_SUFFIX = os.environ.get("DRO_RUN_SUFFIX", "")


def IterateYearMonthRange(startYearMonth, endYearMonth):
  """Return the list of YYYYMM strings spanning the inclusive range."""
  # Parse the start year from the YYYYMM string.
  startYear = int(startYearMonth[:4])
  # Parse the start month from the YYYYMM string.
  startMonth = int(startYearMonth[4:])
  # Parse the end year from the YYYYMM string.
  endYear = int(endYearMonth[:4])
  # Parse the end month from the YYYYMM string.
  endMonth = int(endYearMonth[4:])
  # Initialize the result list.
  results = []
  # Initialize the running year and month to the start values.
  runningYear = startYear
  # Initialize the running month.
  runningMonth = startMonth
  # Loop while the running year-month has not passed the end values.
  while ((runningYear, runningMonth) <= (endYear, endMonth)):
    # Append the formatted YYYYMM string to the results.
    results.append("%04d%02d" % (runningYear, runningMonth))
    # Advance the running month by one.
    runningMonth += 1
    # Wrap the running month to the next year when it exceeds twelve.
    if (runningMonth > 12):
      # Reset the running month to January.
      runningMonth = 1
      # Advance the running year by one.
      runningYear += 1
  # Return the list of YYYYMM strings.
  return results


# Compute the list of year-months in the active range (one or many).
ACTIVE_YEAR_MONTH_LIST = IterateYearMonthRange(ACTIVE_START_YEAR_MONTH, ACTIVE_END_YEAR_MONTH)

# Build the year-month tag used in display strings.
if (ACTIVE_START_YEAR_MONTH == ACTIVE_END_YEAR_MONTH):
  # Use the single-month tag.
  YEAR_MONTH_TAG = ACTIVE_START_YEAR_MONTH
else:
  # Use the multi-month range tag.
  YEAR_MONTH_TAG = "%s-%s" % (ACTIVE_START_YEAR_MONTH, ACTIVE_END_YEAR_MONTH)

# Build the run tag used to prefix every output artifact.
# Single-run example:  "Divvy_202608".
# Multi-run example:   "Divvy_202608_run3" (when DRO_RUN_SUFFIX=_run3 is set).
RUN_TAG = "%s_%s%s" % (ACTIVE_CITY_KEY, YEAR_MONTH_TAG, RUN_SUFFIX)

# Build the base run tag (without the per-run suffix) for aggregated outputs.
BASE_RUN_TAG = "%s_%s" % (ACTIVE_CITY_KEY, YEAR_MONTH_TAG)

# Instantiate a seeded random number generator shared across all phases.
RANDOM_GENERATOR = np.random.default_rng(GLOBAL_RANDOM_SEED)

# Section 2 - Filesystem layout.

# Define the base directory anchoring all project artifacts.
BASE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Define the data directory for persistent numerical artifacts.
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, "data")
# Define the raw data directory holding the downloaded trip CSV files.
RAW_DIRECTORY = os.path.join(DATA_DIRECTORY, "raw")
# Define the figures directory for publication-ready visual outputs.
FIGURES_DIRECTORY = os.path.join(DATA_DIRECTORY, "figures")
# Define the multi-run results directory used by phase6.
MULTIRUN_DIRECTORY = os.path.join(DATA_DIRECTORY, "multirun")
# Ensure the data directory exists before any phase writes artifacts.
os.makedirs(DATA_DIRECTORY, exist_ok=True)
# Ensure the figures directory exists before any visualization phase runs.
os.makedirs(FIGURES_DIRECTORY, exist_ok=True)
# Ensure the multi-run directory exists for the aggregator phase.
os.makedirs(MULTIRUN_DIRECTORY, exist_ok=True)

# Section 3 - Trip data source catalogue.

# Define the catalog of real trip data sources keyed by city identifier.
TRIP_DATA_SOURCES = {
  "Divvy"      : {
    "CityName"           : "Chicago, Illinois, USA",
    "DownloadUrlTemplate": "https://divvy-tripdata.s3.amazonaws.com/{YearMonth}-divvy-tripdata.zip",
    "LocalZipTemplate"   : "divvy_{YearMonth}.zip",
    "ExtractSubdir"      : "divvy",
    "CityCenter"         : (41.8781, -87.6298),
  },
  "JC"         : {
    "CityName"           : "Jersey City, NJ, USA",
    "DownloadUrlTemplate": "https://s3.amazonaws.com/tripdata/JC-{YearMonth}-citibike-tripdata.csv.zip",
    "LocalZipTemplate"   : "jc_citibike_{YearMonth}.zip",
    "ExtractSubdir"      : "jc",
    "CityCenter"         : (40.7282, -74.0776),
  },
  "CitibikeNYC": {
    "CityName"           : "New York, NY, USA",
    "DownloadUrlTemplate": "https://s3.amazonaws.com/tripdata/{YearMonth}-citibike-tripdata.zip",
    "LocalZipTemplate"   : "nyc_citibike_{YearMonth}.zip",
    "ExtractSubdir"      : "nyc",
    "CityCenter"         : (40.7128, -74.0060),
  },
}

# Section 4 - Experimental knobs.

# Define the configuration dictionary collecting all fixed experimental knobs.
EXPERIMENT_CONFIGURATION = {
  "TargetCity"                 : "Chicago, Illinois, USA",
  "ActiveCityKey"              : ACTIVE_CITY_KEY,
  "ActiveStartYearMonth"       : ACTIVE_START_YEAR_MONTH,
  "ActiveEndYearMonth"         : ACTIVE_END_YEAR_MONTH,
  "ActiveYearMonthList"        : ACTIVE_YEAR_MONTH_LIST,
  "RunTag"                     : RUN_TAG,
  "BaseRunTag"                 : BASE_RUN_TAG,
  "RunSuffix"                  : RUN_SUFFIX,
  "NumberOfStationsToSelect"   : 540,
  "NumberOfClusters"           : 18,
  "ClusterSizeBands"           : {"Small": 15, "Medium": 30, "Large": 45},
  "ClusterBandCounts"          : {"Small": 6, "Medium": 6, "Large": 6},
  "TrainingScenarios"          : 30,
  "OutOfSampleScenarios"       : 100,
  "TrainTestSplitFraction"     : 0.80,
  "ScenarioWindowSizeFraction" : 0.033,
  "UnmetDemandPenalty"         : 12.50,
  "RebalancingCostPerKilometer": 0.85,
  "ManhattanDetourFactor"      : 1.32,
  "WassersteinRadius"          : 1.5,
  "CvarAlpha"                  : 0.75,
  "DepotSourcingCost"          : 4.0,
  "SolverVerbosity"            : 0,
  "BootstrapResamples"         : 2000,
  "BootstrapConfidence"        : 0.95,
  "RandomSeed"                 : GLOBAL_RANDOM_SEED,
}


# Section 5 - Output path helpers used by every downstream phase.

def TaggedFileName(baseName):
  """Return a data-directory path with the run tag prefix to avoid overwriting."""
  # Return the joined path with the run tag prefix.
  return os.path.join(DATA_DIRECTORY, "%s_%s" % (RUN_TAG, baseName))


def TaggedFigureName(baseName):
  """Return a figures-directory path with the run tag prefix."""
  # Return the joined path with the run tag prefix.
  return os.path.join(FIGURES_DIRECTORY, "%s_%s" % (RUN_TAG, baseName))


def TaggedRawName(baseName):
  """Return a raw-data-directory path with the run tag prefix."""
  # Return the joined path with the run tag prefix.
  return os.path.join(RAW_DIRECTORY, "%s_%s" % (RUN_TAG, baseName))


def MultirunRunDirectory(runIndex):
  """Return the path to the per-run subfolder under data/multirun/."""
  # Return the joined path for the Run{N} subfolder.
  return os.path.join(MULTIRUN_DIRECTORY, "Run%d" % runIndex)


def MultirunRunFiguresDirectory(runIndex):
  """Return the path to the figures subfolder inside a per-run directory."""
  # Return the joined path for the figures subfolder of Run{N}.
  return os.path.join(MultirunRunDirectory(runIndex), "figures")


def MultirunAggregatedName(cityKey, yearMonthTag, nRuns, suffix):
  """Return a path for an aggregated multi-run artifact in data/multirun/."""
  # Build the file name with the multirun suffix.
  fileName = "%s_%s_multirun%d_%s" % (cityKey, yearMonthTag, nRuns, suffix)
  # Return the joined path.
  return os.path.join(MULTIRUN_DIRECTORY, fileName)


def MultirunAggregatedFigureName(cityKey, yearMonthTag, nRuns, suffix):
  """Return a path for an aggregated multi-run figure in data/figures/."""
  # Build the file name with the multirun suffix.
  fileName = "%s_%s_multirun%d_%s" % (cityKey, yearMonthTag, nRuns, suffix)
  # Return the joined path.
  return os.path.join(FIGURES_DIRECTORY, fileName)
