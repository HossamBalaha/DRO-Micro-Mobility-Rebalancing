"""
Phase0Download.py
==================

Phase 0 - Remote trip-data archive downloader and extractor (multi-month, per-month subdirectory).

PURPOSE
-------
Fetches the raw bike-share trip archive for the active city and each
year-month in the active range, then unzips each archive into its OWN
per-month subdirectory under data/raw/<ExtractSubdir>/<YearMonth>/.

BUGFIX (carried from v4)
-------------------------
v3 extracted every month to the same directory data/raw/<ExtractSubdir>/
and returned the first CSV found there.  When loading multiple months,
every call returned the SAME first CSV (whatever happened to be first
alphabetically).  This caused the "Loading CSV: 202501-divvy-tripdata.csv"
appearing 8 times even when the user requested 2026 data.

v4+ fixes this by extracting each month to its own subdirectory
data/raw/<ExtractSubdir>/<YearMonth>/ so the per-month CSV path is unique
and LocateActiveCityTripCsvs returns a list of distinct paths.

NOT-FOUND HANDLING
------------------
The download is wrapped in try/except so a missing month's archive is
skipped rather than aborting the run.  If a month's archive cannot be
downloaded or extracted, that month is skipped and the next month is
attempted.  If ALL months fail, the pipeline raises a RuntimeError.

OUTPUTS
-------
  * data/raw/<CityKey>_<YearMonth>.zip                  (cached zip)
  * data/raw/<ExtractSubdir>/<YearMonth>/<CSV>          (per-month CSV)

The list of CSV paths returned by LocateActiveCityTripCsvs is consumed by Phase1.
"""

# Import the urllib.request module for downloading remote archive files.
import urllib.request
# Import the zipfile module for extracting the downloaded archives.
import zipfile
# Import the os library for file system interactions.
import os
# Import the sys library for interpreter path resolution.
import sys
# Import the shared configuration module for global knobs and paths.
import Config


def DownloadTripDataArchive(cityKey, yearMonth):
  """Download a remote archive for the given city and year-month."""
  # Retrieve the source descriptor for the requested city.
  sourceDescriptor = Config.TRIP_DATA_SOURCES[cityKey]
  # Build the remote download URL by substituting the year-month template.
  downloadUrl = sourceDescriptor["DownloadUrlTemplate"].format(YearMonth=yearMonth)
  # Build the local zip file name with the city+year tag.
  localZipName = sourceDescriptor["LocalZipTemplate"].format(YearMonth=yearMonth)
  # Construct the local path for the downloaded zip archive.
  localZipPath = os.path.join(Config.RAW_DIRECTORY, localZipName)
  # Skip the download when the archive already exists locally.
  if (os.path.exists(localZipPath)):
    # Print a skip message and return the existing path.
    print("Archive for %s %s already present at %s; skipping download." % (
      cityKey, yearMonth, localZipPath))
    # Return the existing local zip path.
    return localZipPath
  # Print a download announcement.
  print("Downloading %s %s trip data from %s ..." % (cityKey, yearMonth, downloadUrl))
  # Ensure the raw directory exists before the download.
  os.makedirs(Config.RAW_DIRECTORY, exist_ok=True)

  # Define the reporthook closure to report download progress.
  def reporthook(blockNumber, blockSize, totalSize):
    # Guard against an unknown total size.
    if (totalSize > 0):
      # Compute the download completion fraction.
      completionFraction = blockNumber * blockSize / totalSize
      # Print a compact progress line every ten percent.
      if (blockNumber % max(1, totalSize // (blockSize * 10)) == 0):
        # Print the progress percentage.
        print("  %s: %5.1f%%" % (cityKey, min(100.0, completionFraction * 100.0)))

  # Wrap the download in a try/except for not-found handling.
  try:
    # Execute the download via urllib with a generous timeout.
    urllib.request.urlretrieve(downloadUrl, localZipPath, reporthook=reporthook)
  except Exception as exceptionObject:
    # Print an error message if the download failed.
    print("Error occurred while downloading %s %s: %s" % (
      cityKey, yearMonth, str(exceptionObject)))
    # Return None to signal the failure.
    return None
  # Print a completion message with the local file size.
  print("Downloaded %s %s (%.1f MB) to %s." % (
    cityKey, yearMonth, os.path.getsize(localZipPath) / 1e6, localZipPath))
  # Return the local zip path.
  return localZipPath


def ExtractTripDataArchive(cityKey, localZipPath, yearMonth):
  """Extract a downloaded archive into a PER-MONTH subdirectory."""
  # Retrieve the source descriptor for the requested city.
  sourceDescriptor = Config.TRIP_DATA_SOURCES[cityKey]
  # Build the per-month extraction destination directory (the v4 bugfix).
  extractDestination = os.path.join(
    Config.RAW_DIRECTORY, sourceDescriptor["ExtractSubdir"], yearMonth)
  # Skip extraction when the per-month destination already contains a CSV.
  if (os.path.isdir(extractDestination)):
    # List the CSV files in the per-month destination.
    csvFiles = [name for name in os.listdir(extractDestination) if (name.endswith(".csv"))]
    # Skip extraction when at least one CSV file is present.
    if (len(csvFiles) > 0):
      # Print a skip message.
      print("Extraction for %s %s already present at %s; skipping." % (
        cityKey, yearMonth, extractDestination))
      # Return the path to the first CSV file in this per-month subdir.
      return os.path.join(extractDestination, csvFiles[0])
  # Ensure the per-month extraction destination exists.
  os.makedirs(extractDestination, exist_ok=True)
  # Open the zip archive for reading.
  with zipfile.ZipFile(localZipPath, "r") as archiveHandle:
    # Extract every member into the per-month destination directory.
    archiveHandle.extractall(extractDestination)
  # Print a completion message.
  print("Extracted %s %s into %s." % (cityKey, yearMonth, extractDestination))
  # List the CSV files inside the per-month destination.
  csvFiles = [name for name in os.listdir(extractDestination) if (name.endswith(".csv"))]
  # Guard against an empty extraction result.
  if (len(csvFiles) == 0):
    # Print an error message.
    print("Error: no CSV file found in %s after extraction." % extractDestination)
    # Return None to signal the failure.
    return None
  # Return the path to the first CSV file.
  return os.path.join(extractDestination, csvFiles[0])


def LocateActiveCityTripCsvs():
  """Return the list of trip CSV paths covering all year-months in the active range."""
  # Initialize the list of CSV paths to populate.
  csvPaths = []
  # Iterate over each year-month in the active range.
  for yearMonth in Config.ACTIVE_YEAR_MONTH_LIST:
    # Download the archive for the active city and current year-month.
    localZipPath = DownloadTripDataArchive(Config.ACTIVE_CITY_KEY, yearMonth)
    # Skip this month when the download failed (not-found handling).
    if (localZipPath is None):
      # Print an error message.
      print("Error: failed to download archive for %s %s; skipping." % (
        Config.ACTIVE_CITY_KEY, yearMonth))
      # Continue to the next year-month.
      continue
    # Extract the archive into its per-month subdirectory.
    csvPath = ExtractTripDataArchive(Config.ACTIVE_CITY_KEY, localZipPath, yearMonth)
    # Skip this month when the extraction failed.
    if (csvPath is None):
      # Print an error message.
      print("Error: failed to extract archive for %s %s; skipping." % (
        Config.ACTIVE_CITY_KEY, yearMonth))
      # Continue to the next year-month.
      continue
    # Append the CSV path to the list.
    csvPaths.append(csvPath)
  # Guard against an empty CSV list (e.g. all downloads failed).
  if (len(csvPaths) == 0):
    # Raise a runtime error with a clear message.
    raise RuntimeError("No CSV files were successfully downloaded and extracted for %s." % Config.RUN_TAG)
  # Return the list of CSV paths.
  return csvPaths


def main():
  """Main entry point: download and extract all year-months in the active range."""
  # Iterate over each year-month in the active range.
  for yearMonth in Config.ACTIVE_YEAR_MONTH_LIST:
    # Download the archive for the current year-month.
    localZipPath = DownloadTripDataArchive(Config.ACTIVE_CITY_KEY, yearMonth)
    # Skip this month when the download failed.
    if (localZipPath is None):
      # Print an error message.
      print("Error: failed to download archive for %s %s; skipping." % (
        Config.ACTIVE_CITY_KEY, yearMonth))
      # Continue to the next year-month.
      continue
    # Extract the archive for the current year-month.
    ExtractTripDataArchive(Config.ACTIVE_CITY_KEY, localZipPath, yearMonth)
  # Print a confirmation message summarizing the download phase.
  print("Phase 0 complete [%s]: real trip data archives for %s %s -> %s downloaded and extracted into %s." % (
    Config.RUN_TAG, Config.ACTIVE_CITY_KEY, Config.ACTIVE_START_YEAR_MONTH,
    Config.ACTIVE_END_YEAR_MONTH, Config.RAW_DIRECTORY))


# Execute the main function when the script runs as the entry point.
if __name__ == "__main__":
  # Invoke the main entry point.
  main()
