#' Compute dwell and waiting times
#'
#' Dwell is the time between actual arrival and departure; waiting is the
#' delay between the announced and the actual arrival. Both are in hours.
#'
#' @param arrivals A tibble of cleaned arrivals with berth details.
#' @return `arrivals` with `dwell_h` and `wait_h` columns added.
#' @keywords internal
compute_dwell <- function(arrivals) {
  arrivals %>%
    dplyr::mutate(
      dwell_h = as.numeric(difftime(atd, ata, units = "hours")),
      wait_h = as.numeric(difftime(ata, eta, units = "hours")),
      # Negative dwell means ATD was keyed before ATA; treat it as unknown
      dwell_h = dplyr::if_else(dwell_h < 0, NA_real_, dwell_h)
    )
}

#' Flag delayed arrivals
#'
#' @param arrivals A tibble of arrivals with `wait_h` computed.
#' @param threshold_hours Waiting time above which a call counts as delayed.
#' @return `arrivals` with a logical `delayed` column added.
#' @keywords internal
flag_delays <- function(arrivals, threshold_hours = 2) {
  stopifnot(is.numeric(threshold_hours), length(threshold_hours) == 1)

  # Unknown waiting time is not a delay
  arrivals %>%
    dplyr::mutate(delayed = !is.na(wait_h) & wait_h > threshold_hours)
}

#' Summarise arrivals by berth
#'
#' @param arrivals A tibble of arrivals with dwell and delay columns.
#' @return A tibble with one row per berth.
#' @rdname summarise_arrivals
#' @keywords internal
summarise_by_berth <- function(arrivals) {
  arrivals %>%
    dplyr::group_by(terminal, berth) %>%
    dplyr::summarise(
      calls = dplyr::n(),
      median_dwell_h = stats::median(dwell_h, na.rm = TRUE),
      mean_wait_h = mean(wait_h, na.rm = TRUE),
      delayed_share = mean(delayed),
      .groups = "drop"
    ) %>%
    dplyr::arrange(terminal, berth)
}

#' Summarise arrivals by day
#'
#' @param arrivals A tibble of arrivals with dwell and delay columns.
#' @rdname summarise_arrivals
#' @keywords internal
summarise_by_day <- function(arrivals) {
  arrivals %>%
    dplyr::group_by(arrival_date) %>%
    dplyr::summarise(
      calls = dplyr::n(),
      median_dwell_h = stats::median(dwell_h, na.rm = TRUE),
      mean_wait_h = mean(wait_h, na.rm = TRUE),
      delayed_share = mean(delayed),
      .groups = "drop"
    ) %>%
    dplyr::arrange(arrival_date)
}

#' Plot daily arrival volume
#'
#' @param daily A tibble returned by [summarise_by_day()].
#' @return A ggplot object.
#' @keywords internal
plot_daily_volume <- function(daily) {
  ggplot2::ggplot(daily, ggplot2::aes(x = arrival_date, y = calls)) +
    ggplot2::geom_col(fill = "grey40") +
    # 7-day trailing mean; assumes one row per day with no gaps
    ggplot2::geom_line(
      ggplot2::aes(y = zoo::rollmean(calls, k = 7, fill = NA, align = "right")),
      colour = "firebrick"
    ) +
    ggplot2::labs(x = NULL, y = "Calls per day", title = "Daily vessel arrivals") +
    ggplot2::theme_minimal()
}

#' Plot dwell times by berth
#'
#' @param by_berth A tibble returned by [summarise_by_berth()].
#' @return A ggplot object.
#' @keywords internal
plot_berth_dwell <- function(by_berth) {
  ggplot2::ggplot(
    by_berth,
    ggplot2::aes(x = stats::reorder(berth, median_dwell_h), y = median_dwell_h, fill = terminal)
  ) +
    ggplot2::geom_col() +
    ggplot2::coord_flip() +
    ggplot2::labs(x = NULL, y = "Median dwell (hours)", title = "Dwell time by berth") +
    ggplot2::theme_minimal()
}

#' Write a summary table for the report
#'
#' Writes `x` as CSV and returns the path, so the function can be used in a
#' `tar_target(format = "file")`.
#'
#' Only plain double columns are rounded; dates and date-times are doubles
#' internally but must be written unchanged.
#'
#' @param x A data frame to write.
#' @param path Path of the CSV file to write.
#' @param digits Number of decimal places kept for numeric columns.
#' @return `path`.
#' @keywords internal
write_summary_csv <- function(x, path, digits = 2) {
  fs::dir_create(fs::path_dir(path))

  is_plain_double <- function(col) is.double(col) && !inherits(col, c("Date", "POSIXt"))
  x %>%
    dplyr::mutate(dplyr::across(dplyr::where(is_plain_double), ~ round(.x, digits))) %>%
    readr::write_csv(path, na = "")

  path
}
