"""
RunAll.py
==========

Single-run orchestrator: executes Phase0 through Phase5 in sequence for one
(city, start-year-month, end-year-month, seed) tuple.

PURPOSE
-------
Run the entire DRO micro-mobility rebalancing pipeline end-to-end in a
single command.  The active year-month range lets the user merge multiple
months of trip data (e.g. all of 2026: --start-year-month 202601
--end-year-month 202612) to produce a richer demand distribution.

SEPARATION FROM RunMulti.py
---------------------------
RunAll.py is the SINGLE-RUN orchestrator.  It writes outputs with the run
tag {City}_{YearMonthRange} (no _run{i} suffix) directly to data/ and
data/figures/.  RunMulti.py is the MULTI-RUN orchestrator and is fully
separate: it sets DRO_RUN_SUFFIX=_run{i} per run so each Phase 1-5
subprocess writes to its own uniquely-named outputs without cross-run
overwrite (v5 bugfix).  RunAll.py never sets DRO_RUN_SUFFIX.

USAGE
-----
  # Default run (uses Config.py defaults: Divvy, 202608, single month).
  python RunAll.py
  # All 12 months of 2026 for Divvy (data is merged across months).
  python RunAll.py --city Divvy --start-year-month 202601 --end-year-month 202612
  # Jersey City, August 2025, custom seed.
  python RunAll.py --city JC --start-year-month 202508 --end-year-month 202508 --seed 42
  # Skip the heavy download phase (data already on disk).
  python RunAll.py --skip-phase0
  # Convenience: --year-month sets both start and end to the same month.
  python RunAll.py --year-month 202608
"""

# Import the subprocess module for executing phase scripts in sequence.
import subprocess
# Import the os library for file path construction and environment propagation.
import os
# Import the sys library for interpreter path resolution.
import sys
# Import the argparse library for command-line argument parsing.
import argparse
# Import the shared configuration module for global paths.
import Config

# Define the ordered list of phase script file names for the real-data pipeline.
PHASE_SCRIPTS = [
  "Phase0Download.py",
  "Phase1RealData.py",
  "Phase2DroModel.py",
  "Phase3BaselinesOutOfSample.py",
  "Phase4StatisticalValidation.py",
  "Phase5Visualization.py",
]


def main():
  """Main entry point: parse CLI args and execute every phase in order."""
  # Build the argument parser.
  parser = argparse.ArgumentParser(description="Run the full DRO pipeline for one city / year-month range.")
  # Add the city argument.
  parser.add_argument("--city", default=None, help="City key (Divvy, JC, CitibikeNYC).")
  # Add the start year-month argument.
  parser.add_argument("--start-year-month", default=None, help="Start year-month YYYYMM (e.g. 202601).")
  # Add the end year-month argument.
  parser.add_argument("--end-year-month", default=None, help="End year-month YYYYMM (e.g. 202612).")
  # Add the convenience --year-month argument.
  parser.add_argument("--year-month", default=None,
                      help="Convenience flag: sets both --start-year-month and --end-year-month to the same value.")
  # Add the seed argument.
  parser.add_argument("--seed", type=int, default=None, help="Global random seed for full reproducibility.")
  # Add the skip-phase0 flag.
  parser.add_argument("--skip-phase0", action="store_true", help="Skip the download phase.")
  # Parse the arguments.
  args = parser.parse_args()
  # Build the environment to propagate to phase subprocesses.
  env = os.environ.copy()
  # Explicitly clear the per-run suffix so single-run mode never collides with multi-run outputs.
  env["DRO_RUN_SUFFIX"] = ""
  # Override the city if provided.
  if (args.city is not None):
    # Set the active city env var.
    env["DRO_ACTIVE_CITY"] = args.city
  # Override the start year-month if provided.
  if (args.start_year_month is not None):
    # Set the start year-month env var.
    env["DRO_START_YEAR_MONTH"] = args.start_year_month
  # Override the end year-month if provided.
  if (args.end_year_month is not None):
    # Set the end year-month env var.
    env["DRO_END_YEAR_MONTH"] = args.end_year_month
  # Convenience: --year-month sets both.
  if (args.year_month is not None):
    # Set the start year-month env var.
    env["DRO_START_YEAR_MONTH"] = args.year_month
    # Set the end year-month env var.
    env["DRO_END_YEAR_MONTH"] = args.year_month
  # Override the seed if provided.
  if (args.seed is not None):
    # Set the random seed env var.
    env["DRO_RANDOM_SEED"] = str(args.seed)
  # Read the resolved values from environment so Config defaults are honored.
  resolvedCity = env.get("DRO_ACTIVE_CITY", Config.ACTIVE_CITY_KEY)
  # Read the resolved start year-month.
  resolvedStartYearMonth = env.get("DRO_START_YEAR_MONTH", Config.ACTIVE_START_YEAR_MONTH)
  # Read the resolved end year-month.
  resolvedEndYearMonth = env.get("DRO_END_YEAR_MONTH", Config.ACTIVE_END_YEAR_MONTH)
  # Read the resolved seed.
  resolvedSeed = env.get("DRO_RANDOM_SEED", str(Config.GLOBAL_RANDOM_SEED))
  # Build the year-month tag.
  if (resolvedStartYearMonth == resolvedEndYearMonth):
    # Use the single-month tag.
    yearMonthTag = resolvedStartYearMonth
  else:
    # Build the range tag.
    yearMonthTag = "%s-%s" % (resolvedStartYearMonth, resolvedEndYearMonth)
  # Print the start header.
  print("=== RunAll.py starting (single-run mode; DRO_RUN_SUFFIX cleared) ===")
  # Print the city.
  print("    city             = %s" % resolvedCity)
  # Print the start year-month.
  print("    start-year-month = %s" % resolvedStartYearMonth)
  # Print the end year-month.
  print("    end-year-month   = %s" % resolvedEndYearMonth)
  # Print the seed.
  print("    seed             = %s" % resolvedSeed)
  # Print the run tag.
  print("    run tag          = %s_%s" % (resolvedCity, yearMonthTag))
  # Get the directory containing this orchestrator script.
  codeDirectory = os.path.dirname(os.path.abspath(__file__))
  # Iterate over each phase script in the ordered list.
  for phaseScript in PHASE_SCRIPTS:
    # Skip phase0 when the user requested to.
    if (phaseScript == "Phase0Download.py" and args.skip_phase0):
      # Print a skip message.
      print("=== Skipping %s (--skip-phase0) ===" % phaseScript)
      # Continue to the next phase.
      continue
    # Build the absolute path to the phase script.
    scriptPath = os.path.join(codeDirectory, phaseScript)
    # Print a progress message.
    print("=== Running %s ===" % phaseScript)
    # Execute the phase script as a subprocess of the current interpreter.
    returnCode = subprocess.call([sys.executable, scriptPath], env=env)
    # Check whether the phase script failed.
    if (returnCode != 0):
      # Print an error message and abort.
      print("Phase %s failed with return code %d. Aborting pipeline." % (phaseScript, returnCode))
      # Return the non-zero exit code.
      return returnCode
  # Print a final confirmation message.
  print("All phases completed successfully [%s_%s]. Artifacts written to %s." % (
    resolvedCity, yearMonthTag, Config.DATA_DIRECTORY))
  # Print the run tag.
  print("Run tag for this run: %s_%s" % (resolvedCity, yearMonthTag))
  # Return a zero exit code.
  return 0


# Execute the main function when the script runs as the entry point.
if __name__ == "__main__":
  # Invoke the main entry point and propagate the exit code.
  raise SystemExit(main())
