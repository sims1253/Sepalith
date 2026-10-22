library(targets)
library(tarchetypes)

settings <- config::get(file = "config.yml", config = Sys.getenv("R_CONFIG_ACTIVE", "default"))

tar_option_set(
  packages = c("dplyr", "readr", "readxl", "jsonlite", "lubridate", "ggplot2", "stringr"),
  format = "qs",
  error = "abridge",
  memory = "transient",
  garbage_collection = TRUE,
  controller = crew::crew_controller_local(workers = settings$workers %||% 2L)
)

tar_source("R")

list(
  # raw inputs, tracked as files so edits trigger a rebuild
  tar_files(
    arrivals_files,
    list.files(settings$inbox, pattern = "^arrivals_.*\\.csv$", full.names = TRUE)
  ),
  tar_target(berths_file, settings$berth_register, format = "file"),
  tar_target(tides_file, settings$tide_feed, format = "file"),
  tar_target(
    arrivals_raw,
    read_arrivals_csv(arrivals_files),
    pattern = map(arrivals_files),
    iteration = "vector"
  ),
  tar_target(berths, read_berths_xlsx(berths_file, sheet = settings$berth_sheet)),
  tar_target(tides, read_tides_json(tides_file)),
  # cleaning and enrichment
  tar_target(arrivals_clean, clean_arrivals(arrivals_raw, tz = settings$timezone)),
  tar_target(arrivals_berths, join_berths(arrivals_clean, berths)),
  tar_target(
    arrivals,
    arrivals_berths |>
      compute_dwell() |>
      flag_delays(threshold_hours = settings$delay_threshold_h)
  ),
  # summaries
  tar_target(by_berth, summarise_by_berth(arrivals)),
  tar_target(by_day, summarise_by_day(arrivals)),
  tar_target(
    by_berth_csv,
    write_summary_csv(by_berth, file.path(settings$output_dir, "by_berth.csv")),
    format = "file"
  ),
  tar_target(
    by_day_csv,
    write_summary_csv(by_day, file.path(settings$output_dir, "by_day.csv")),
    format = "file"
  ),
  # figures
  tar_target(fig_daily_volume, plot_daily_volume(by_day)),
  tar_target(fig_berth_dwell, plot_berth_dwell(by_berth)),
  # report
  tar_quarto(
    weekly_report,
    path = "reports/weekly.qmd",
    execute_params = list(
      week_ending = format(max(by_day$arrival_date)),
      delay_threshold_h = settings$delay_threshold_h
    ),
    quiet = TRUE
  )
)
