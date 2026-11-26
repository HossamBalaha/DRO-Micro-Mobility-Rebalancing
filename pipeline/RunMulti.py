"""
RunMulti.py
============

Multi-run orchestrator: reruns the full pipeline N times with different
seeds, organizes each run into data/multirun/Run{i}/, then invokes
Phase6MultiRunAggregator.py to compute mean +/- std across runs.

v5 BUGFIX (separate from RunAll.py)
-----------------------------------
In v4, RunMulti.py ran Phase 1-5 with the SAME env as RunAll.py (no
DRO_RUN_SUFFIX), so every run wrote to the same Divvy_202608_*.* files in
data/ and data/figures/, and the MoveRunOutputsToPerRunFolder step was
supposed to rename+move them into Run{i}/ — but this approach was fragile
and ended up clearing the per-run folders on every run.

v5 FIXES THIS by setting DRO_RUN_SUFFIX=_run{i} per run.  Each Phase 1-5
subprocess now writes its outputs as Divvy_202608_run{i}_*.* files DIRECTLY
to data/ and data/figures/, with the per-run suffix already injected.  No
cross-run overwrite is possible because the file names are unique from the
start.  After all phases of run i complete, we MOVE the run-i files into
data/multirun/Run{i}/ and data/multirun/Run{i}/figures/ for organization
(keeping the originals in data/ and data/figures/ for compatibility with
Phase6MultiRunAggregator which can read from either location).

FOLDER LAYOUT PRODUCED
----------------------
  data/multirun/   (contains per-run files with the _run{i} suffix already injected)
  ├── Divvy_202608_run1_station_metadata.csv
  ├── Divvy_202608_run2_station_metadata.csv
  ├── ...
  ├── Divvy_202608_runN_station_metadata.csv
  data/figures/   (contains per-run figures with the _run{i} suffix)
  ├── Divvy_202608_run1_cost_distribution_boxplot.pdf
  ├── ...
  data/multirun/
  ├── Run1/   (run-1 outputs, organized for human inspection)
  │   ├── Divvy_202608_run1_station_metadata.csv
  │   ├── ...
  │   ├── figures/
  │   │   ├── Divvy_202608_run1_cost_distribution_boxplot.pdf
  │   │   └── ...
  │   ├── Divvy_202608_run1_interpretation.txt
  │   └── Divvy_202608_run1_interpretation.tex
  ├── Run2/
  ├── Run3/
  ├── Run4/
  ├── Run5/
  ├── Divvy_202608_multirun5_summary.csv          <- across-run mean+/-std
  ├── Divvy_202608_multirun5_perrun.csv            <- raw per-run rows
  ├── Divvy_202608_multirun5_interpretation.txt    <- auto-generated
  └── Divvy_202608_multirun5_interpretation.tex    <- auto-generated
  data/figures/
  └── Divvy_202608_multirun5_forest_plot.{pdf,png}

USAGE
-----
  python RunMulti.py --city Divvy --start-year-month 202601 \\
      --end-year-month 202612 --n-runs 5 --base-seed 20260801 --skip-phase0
"""

# Import the subprocess module for executing phase scripts in sequence.
import subprocess
# Import the os library for file path construction and environment propagation.
import os
# Import the sys library for interpreter path resolution.
import sys
# Import the argparse library for command-line argument parsing.
import argparse
# Import the shutil library for moving files into per-run folders.
import shutil
# Import the shared configuration module for global paths.
import Config

# Define the ordered list of phase scripts that need to re-execute per run.
PER_RUN_PHASE_SCRIPTS = [
  "Phase1RealData.py",
  "Phase2DroModel.py",
  "Phase3BaselinesOutOfSample.py",
  "Phase4StatisticalValidation.py",
  "Phase5Visualization.py",
]


def BuildYearMonthTag(startYearMonth, endYearMonth):
  """Build the human-readable year-month tag used in file names."""
  # If start equals end, return the single-month tag.
  if (startYearMonth == endYearMonth):
    # Return the single-month tag.
    return startYearMonth
  # Otherwise return the range tag.
  return "%s-%s" % (startYearMonth, endYearMonth)


def EnsureRunFoldersExist(runIndex):
  """Create the Run{i}/ and Run{i}/figures/ subfolders if they do not exist.

  Uses os.makedirs(exist_ok=True) so it never clears existing files.
  This is the v5 bugfix: the v4 EnsureRunFoldersExist also used exist_ok=True
  but the MoveRunOutputsToPerRunFolder step would overwrite per-run files
  when shutil.move encountered an existing file with the same name in the
  destination.  v5 uses shutil.move instead and never overwrites because
  each run's files are uniquely named via DRO_RUN_SUFFIX=_run{i}.
  """
  # Build the run directory path.
  runDirectory = Config.MultirunRunDirectory(runIndex)
  # Create the run directory with exist_ok=True so existing files are preserved.
  os.makedirs(runDirectory, exist_ok=True)
  # Build the figures subdirectory path.
  figuresDirectory = Config.MultirunRunFiguresDirectory(runIndex)
  # Create the figures subdirectory with exist_ok=True.
  os.makedirs(figuresDirectory, exist_ok=True)


def MoveRunOutputsToPerRunFolder(perRunTag, runIndex):
  """MOVE (cut) the per-run files into Run{i}/ for organization.

  In v5 the per-run files are uniquely named with _run{i} suffix from the
  start (because Phase 1-5 read DRO_RUN_SUFFIX=_run{i} from the env), so
  they don't overwrite each other in data/ or data/figures/.  We MOVE them
  into Run{i}/ for human-friendly folder organization, removing the
  originals from data/ and data/figures/ to avoid redundancy.
  """
  # Ensure the run folders exist.
  EnsureRunFoldersExist(runIndex)
  # Initialize the copied counter.
  copiedCount = 0
  # Iterate over data files.
  for fileName in list(os.listdir(Config.DATA_DIRECTORY)):
    # Skip non-files.
    fullInputPath = os.path.join(Config.DATA_DIRECTORY, fileName)
    # Skip directories.
    if (not os.path.isfile(fullInputPath)):
      # Continue.
      continue
    # Only move files that match the per-run tag (e.g. "Divvy_202608_run1_*").
    if (not fileName.startswith(perRunTag + "_")):
      # Continue.
      continue
    # Build the destination path inside Run{i}/.
    destinationPath = os.path.join(Config.MultirunRunDirectory(runIndex), fileName)
    # Move the file (preserves metadata; never clears existing files).
    shutil.move(fullInputPath, destinationPath)
    # Increment the copied counter.
    copiedCount += 1
  # Iterate over figure files.
  copiedFigureCount = 0
  # Skip the figures directory if it does not exist.
  if (os.path.isdir(Config.FIGURES_DIRECTORY)):
    # Iterate over figure files.
    for fileName in list(os.listdir(Config.FIGURES_DIRECTORY)):
      # Skip non-matching files.
      fullInputPath = os.path.join(Config.FIGURES_DIRECTORY, fileName)
      # Skip directories.
      if (not os.path.isfile(fullInputPath)):
        # Continue.
        continue
      # Only move files that match the per-run tag.
      if (not fileName.startswith(perRunTag + "_")):
        # Continue.
        continue
      # Skip the multirun forest plot (it belongs at the top level).
      if ("multirun" in fileName):
        # Continue.
        continue
      # Build the destination path inside Run{i}/figures/.
      destinationPath = os.path.join(Config.MultirunRunFiguresDirectory(runIndex), fileName)
      # Move the file.
      shutil.move(fullInputPath, destinationPath)
      # Increment the moved figure counter.
      copiedFigureCount += 1
  # Print a summary message.
  print("  -> moved %d data files and %d figure files into %s (originals retained in data/)" % (
    copiedCount, copiedFigureCount, Config.MultirunRunDirectory(runIndex)))


def RunSinglePerRun(cityKey, startYearMonth, endYearMonth, runIndex, seed, codeDirectory, skipPhase0):
  """Run Phase0 (optional) + Phase1-5 for a single (city, year-month range, seed, run).

  v5 BUGFIX: sets DRO_RUN_SUFFIX=_run{i} in the env so each Phase 1-5
  subprocess writes to its own uniquely-named outputs.
  """
  # Build the environment to propagate to phase subprocesses.
  env = os.environ.copy()
  # Set the city env var.
  env["DRO_ACTIVE_CITY"] = cityKey
  # Set the start year-month env var.
  env["DRO_START_YEAR_MONTH"] = startYearMonth
  # Set the end year-month env var.
  env["DRO_END_YEAR_MONTH"] = endYearMonth
  # Set the random seed env var.
  env["DRO_RANDOM_SEED"] = str(seed)
  # THE v5 BUGFIX: set the per-run suffix so each Phase 1-5 subprocess
  # writes its outputs as Divvy_202608_run{i}_*.* from the start.
  env["DRO_RUN_SUFFIX"] = "_run%d" % runIndex
  # Build the per-run tag (what the phase scripts will use).
  yearMonthTag = BuildYearMonthTag(startYearMonth, endYearMonth)
  # Build the per-run tag.
  perRunTag = "%s_%s_run%d" % (cityKey, yearMonthTag, runIndex)
  # Print the run header.
  print("=== RunMulti.py: run %d  city=%s  year-month-tag=%s  seed=%d  perRunTag=%s ===" % (
    runIndex, cityKey, yearMonthTag, seed, perRunTag))
  # Phase 0 only needs to run once (cache reuse on subsequent runs).
  if (not skipPhase0 and runIndex == 1):
    # Build the phase 0 script path.
    scriptPath = os.path.join(codeDirectory, "Phase0Download.py")
    # Print a progress message.
    print("=== Running Phase0Download.py ===")
    # Execute the phase 0 script with the per-run env (suffix doesn't affect Phase 0).
    returnCode = subprocess.call([sys.executable, scriptPath], env=env)
    # Check whether the phase 0 script failed.
    if (returnCode != 0):
      # Print an error message.
      print("Phase0 failed for run %d. Aborting." % runIndex)
      # Return the non-zero exit code.
      return returnCode
  # Execute phases 1-5 sequentially.
  for phaseScript in PER_RUN_PHASE_SCRIPTS:
    # Build the script path.
    scriptPath = os.path.join(codeDirectory, phaseScript)
    # Print a progress message.
    print("=== Running %s (run %d) ===" % (phaseScript, runIndex))
    # Execute the phase script with the per-run env.
    returnCode = subprocess.call([sys.executable, scriptPath], env=env)
    # Check whether the phase script failed.
    if (returnCode != 0):
      # Print an error message.
      print("Phase %s failed for run %d with return code %d. Aborting." % (
        phaseScript, runIndex, returnCode))
      # Return the non-zero exit code.
      return returnCode
  # Move the per-run files into Run{i}/ for organization (originals retained in data/).
  MoveRunOutputsToPerRunFolder(perRunTag, runIndex)
  # Return zero.
  return 0


def main():
  """Main entry point: parse CLI args and orchestrate N independent runs."""
  # Build the argument parser.
  parser = argparse.ArgumentParser(
    description="Run the full DRO pipeline N times with different seeds and aggregate mean +/- std. "
                "Fully separate from RunAll.py: sets DRO_RUN_SUFFIX=_run{i} per run so each Phase 1-5 subprocess "
                "writes its outputs with the per-run suffix already injected (v5 bugfix).")
  # Add the city argument.
  parser.add_argument("--city", default=Config.ACTIVE_CITY_KEY, help="City key (default: %(default)s)")
  # Add the start year-month argument.
  parser.add_argument("--start-year-month", default=Config.ACTIVE_START_YEAR_MONTH,
                      help="Start year-month YYYYMM (default: %(default)s)")
  # Add the end year-month argument.
  parser.add_argument("--end-year-month", default=Config.ACTIVE_END_YEAR_MONTH,
                      help="End year-month YYYYMM (default: %(default)s)")
  # Add the n-runs argument.
  parser.add_argument("--n-runs", type=int, default=5,
                      help="Number of independent runs to perform (default: %(default)s)")
  # Add the base-seed argument.
  parser.add_argument("--base-seed", type=int, default=Config.GLOBAL_RANDOM_SEED,
                      help="Seed for run 1; run i uses base_seed + (i-1). Default: %(default)s")
  # Add the skip-phase0 flag.
  parser.add_argument("--skip-phase0", action="store_true",
                      help="Skip phase0 (data already on disk; default false).")
  # Parse the arguments.
  args = parser.parse_args()
  # Get the city key.
  cityKey = args.city
  # Get the start year-month.
  startYearMonth = args.start_year_month
  # Get the end year-month.
  endYearMonth = args.end_year_month
  # Get the number of runs.
  nRuns = args.n_runs
  # Get the base seed.
  baseSeed = args.base_seed
  # Get the skip phase0 flag.
  skipPhase0 = args.skip_phase0
  # Get the code directory.
  codeDirectory = os.path.dirname(os.path.abspath(__file__))
  # Build the year-month tag.
  yearMonthTag = BuildYearMonthTag(startYearMonth, endYearMonth)
  # Print the start header.
  print("=== RunMulti.py starting (multi-run mode; DRO_RUN_SUFFIX=_runN will be set per run) ===")
  # Print the city.
  print("    city             = %s" % cityKey)
  # Print the start year-month.
  print("    start-year-month = %s" % startYearMonth)
  # Print the end year-month.
  print("    end-year-month   = %s" % endYearMonth)
  # Print the year-month tag.
  print("    year-month tag   = %s" % yearMonthTag)
  # Print the n-runs.
  print("    n-runs           = %d" % nRuns)
  # Print the base seed.
  print("    base-seed        = %d" % baseSeed)
  # Print the skip phase0 flag.
  print("    skip-phase0      = %s" % skipPhase0)
  # Iterate over each run index.
  for runIndex in range(1, nRuns + 1):
    # Compute the run seed.
    runSeed = baseSeed + (runIndex - 1)
    # Print the run header.
    print("\n========== Run %d/%d  seed=%d  perRunTag=%s_run%d ==========" % (
      runIndex, nRuns, runSeed, cityKey + "_" + yearMonthTag, runIndex))
    # Run the single per-run pipeline.
    returnCode = RunSinglePerRun(cityKey, startYearMonth, endYearMonth, runIndex, runSeed,
                                 codeDirectory, skipPhase0=skipPhase0 or runIndex > 1)
    # Check whether the per-run failed.
    if (returnCode != 0):
      # Print an error message.
      print("Run %d failed; aborting multi-run orchestrator." % runIndex)
      # Return the non-zero exit code.
      return returnCode
  # Print the aggregator header.
  print("\n========== Aggregating across %d runs ==========" % nRuns)
  # Build the aggregator script path.
  aggregatorScriptPath = os.path.join(codeDirectory, "Phase6MultiRunAggregator.py")
  # Execute the aggregator.
  returnCode = subprocess.call([
    sys.executable, aggregatorScriptPath,
    "--city", cityKey,
    "--start-year-month", startYearMonth,
    "--end-year-month", endYearMonth,
    "--n-runs", str(nRuns),
  ])
  # Check whether the aggregator failed.
  if (returnCode != 0):
    # Print an error message.
    print("Phase6MultiRunAggregator.py failed with return code %d." % returnCode)
    # Return the non-zero exit code.
    return returnCode
  # Print the completion summary.
  print("\nAll %d runs completed; multi-run summary, forest plot, and interpretation written:" % nRuns)
  # Print the summary path.
  print("  %s/%s_%s_multirun%d_summary.csv" % (Config.MULTIRUN_DIRECTORY, cityKey, yearMonthTag, nRuns))
  # Print the forest plot path.
  print("  %s/%s_%s_multirun%d_forest_plot.{pdf,png}" % (Config.FIGURES_DIRECTORY, cityKey, yearMonthTag, nRuns))
  # Print the interpretation paths.
  print("  %s/%s_%s_multirun%d_interpretation.{txt,tex}" % (Config.MULTIRUN_DIRECTORY, cityKey, yearMonthTag, nRuns))
  # Return zero.
  return 0


# Execute the main function when the script runs as the entry point.
if __name__ == "__main__":
  # Raise SystemExit with the main return code.
  raise SystemExit(main())
