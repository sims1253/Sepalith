# Millrace catchment: import, convert and quality-control the raw logger
# exports from the three weather stations and the outlet gauge.

library(dplyr)
library(readr)
library(tidyr)
library(lubridate)
library(gaugewise)

raw_dir <- "data/raw"
out_dir <- "data/clean"

station_limits <- list(
  tmin = list(lower = -35, upper = 35, max_step = 15),
  tmax = list(lower = -30, upper = 42, max_step = 15),
  precip = list(lower = 0, upper = 250, max_step = 200),
  wind = list(lower = 0, upper = 60, max_step = 25),
  rh = list(lower = 0, upper = 100, max_step = 60)
)

read_logger_csv <- function(path, station_id) {
  # Loggers write three header lines and switched timestamp format in 2019
  read_csv(
    path,
    skip = 3,
    col_names = c("timestamp", "tmin_f", "tmax_f", "precip_in", "wind_mph", "rh"),
    col_types = cols(timestamp = col_character(), .default = col_double()),
    na = c("", "NA", "-9999", "NAN")
  ) %>%
    mutate(
      station = station_id,
      date = as_date(parse_date_time(timestamp, orders = c("ymd HMS", "mdy HM")))
    ) %>%
    select(-timestamp)
}

read_gauge_csv <- function(path, gauge_id) {
  # "Ice" marks days when the stage-discharge relation does not hold
  read_csv(
    path,
    skip = 1,
    col_names = c("date", "flow_cfs", "grade"),
    col_types = cols(date = col_date(), flow_cfs = col_double(), grade = col_character()),
    na = c("", "NA", "-9999", "Ice")
  ) %>%
    mutate(
      gauge = gauge_id,
      flow = convert_discharge(flow_cfs, from = "cfs", to = "m3_s")
    ) %>%
    select(-flow_cfs)
}

to_metric <- function(df) {
  df %>%
    mutate(
      tmin = convert_temperature(tmin_f, from = "F", to = "C"),
      tmax = convert_temperature(tmax_f, from = "F", to = "C"),
      precip = convert_precipitation(precip_in, from = "inch", to = "mm"),
      wind = convert_speed(wind_mph, from = "mph", to = "m_s")
    ) %>%
    select(station, date, tmin, tmax, precip, wind, rh)
}

complete_daily <- function(df, key = "station") {
  # Collapse sub-daily records to daily means, then make missing days explicit
  df %>%
    group_by(.data[[key]], date) %>%
    summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    mutate(across(where(is.numeric), ~ ifelse(is.nan(.x), NA_real_, .x))) %>%
    group_by(.data[[key]]) %>%
    complete(date = seq(min(date), max(date), by = "day")) %>%
    ungroup()
}

mask_flagged <- function(df, vars, worst_ok = 1L) {
  for (v in vars) {
    flag_col <- paste0(v, "_qc")
    if (!flag_col %in% names(df)) {
      next
    }

    # Code 9 means the value was already missing; nothing to mask
    bad <- !is.na(df[[flag_col]]) & df[[flag_col]] > worst_ok & df[[flag_col]] != 9L
    df[[v]][bad] <- NA_real_
  }
  df
}

fill_station_gaps <- function(df, neighbours, vars = c("tmin", "tmax")) {
  pieces <- split(df, df$station)

  for (st in names(pieces)) {
    piece <- pieces[[st]]
    nb_id <- neighbours[[st]]
    for (v in vars) {
      # Align the neighbour's series to this station's dates
      nb <- NULL
      if (!is.null(nb_id) && nb_id %in% names(pieces)) {
        nb_df <- pieces[[nb_id]]
        nb <- nb_df[[v]][match(piece$date, nb_df$date)]
      }

      filled <- fill_cascade(piece[[v]], piece$date, neighbour = nb, max_gap = 2L)
      piece[[v]] <- filled$value
      piece[[paste0(v, "_source")]] <- filled$source
    }
    pieces[[st]] <- piece
  }

  bind_rows(pieces)
}

station_year_summary <- function(df, min_coverage = 0.9) {
  df <- df %>% mutate(year = year(date))
  keys <- df %>% distinct(station, year) %>% arrange(station, year)
  rows <- vector("list", nrow(keys))

  for (i in seq_len(nrow(keys))) {
    sub <- df %>% filter(station == keys$station[i], year == keys$year[i])
    n_days <- nrow(sub)

    coverage <- c(
      tmin = mean(!is.na(sub$tmin)),
      tmax = mean(!is.na(sub$tmax)),
      precip = mean(!is.na(sub$precip))
    )
    # A year is only "complete" when every key variable is well covered
    if (all(coverage >= min_coverage)) {
      status <- "complete"
    } else if (any(coverage >= min_coverage)) {
      status <- "partial"
    } else {
      status <- "sparse"
    }

    filled_share <- if ("tmax_source" %in% names(sub)) {
      mean(!is.na(sub$tmax_source) & sub$tmax_source != "observed")
    } else {
      NA_real_
    }

    # Precipitation totals with gaps would be biased low, so drop them
    rows[[i]] <- tibble(
      station = keys$station[i],
      year = keys$year[i],
      n_days = n_days,
      tmin_mean = mean(sub$tmin, na.rm = TRUE),
      tmax_mean = mean(sub$tmax, na.rm = TRUE),
      precip_total = if (coverage[["precip"]] >= min_coverage) {
        sum(sub$precip, na.rm = TRUE)
      } else {
        NA_real_
      },
      filled_share = filled_share,
      status = status
    )
  }

  bind_rows(rows)
}

write_clean <- function(df, name) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  path <- file.path(out_dir, paste0(name, ".csv"))

  write_csv(df, path)
  message("Wrote ", nrow(df), " rows to ", path)
  invisible(path)
}

# Stations -----------------------------------------------------------------

station_files <- list.files(raw_dir, pattern = "^wx_.*\\.csv$", full.names = TRUE)
station_ids <- sub("^wx_(.*)\\.csv$", "\\1", basename(station_files))

stations <- purrr::map2_dfr(station_files, station_ids, read_logger_csv) %>%
  to_metric() %>%
  complete_daily("station")

stations_qc <- stations %>%
  group_by(station) %>%
  group_modify(~ qc_station(.x, station_limits)) %>%
  ungroup()

print(qc_report(stations_qc))

stations_clean <- stations_qc %>%
  mask_flagged(names(station_limits)) %>%
  fill_station_gaps(neighbours = list(upper = "mill", mill = "upper", lower = "mill"))

write_clean(stations_clean, "stations_daily")
write_clean(station_year_summary(stations_clean), "station_years")

# Gauge --------------------------------------------------------------------

gauge <- read_gauge_csv(file.path(raw_dir, "gauge_outlet.csv"), "outlet") %>%
  complete_daily("gauge")

gauge$flow <- fill_linear(gauge$flow, max_gap = 5L)$value
write_clean(gauge, "gauge_daily")
