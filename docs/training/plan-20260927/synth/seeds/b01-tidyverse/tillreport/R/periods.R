#' Find the start of the reporting period containing each date
#'
#' Weeks start on Monday. Fiscal years start on the first day of
#' `fiscalStartMonth`.
#'
#' @param dates A vector of `Date` values.
#' @param unit One of `"week"`, `"month"`, `"quarter"` or `"fiscalYear"`.
#' @param fiscalStartMonth Month (1-12) in which the fiscal year begins.
#' @return A `Date` vector the same length as `dates`.
#' @export
periodStart <- function(dates, unit = c("week", "month", "quarter", "fiscalYear"),
                        fiscalStartMonth = 2L) {
  unit <- match.arg(unit)
  dates <- lubridate::as_date(dates)
  switch(unit,
    week = lubridate::floor_date(dates, "week", week_start = 1),
    month = lubridate::floor_date(dates, "month"),
    quarter = lubridate::floor_date(dates, "quarter"),
    fiscalYear = {
      yr <- lubridate::year(dates) - (lubridate::month(dates) < fiscalStartMonth)
      lubridate::make_date(yr, fiscalStartMonth, 1L)
    }
  )
}

#' Number weeks within the fiscal year
#'
#' Week 1 is the week containing the first day of the fiscal year, so a
#' fiscal year can have 53 weeks.
#'
#' @param dates A vector of `Date` values.
#' @param fiscalStartMonth Month (1-12) in which the fiscal year begins.
#' @return An integer vector of fiscal week numbers.
#' @export
fiscalWeek <- function(dates, fiscalStartMonth = 2L) {
  yearStart <- periodStart(dates, "fiscalYear", fiscalStartMonth)
  firstMonday <- lubridate::floor_date(yearStart, "week", week_start = 1)
  as.integer((lubridate::as_date(dates) - firstMonday) %/% 7) + 1L
}

#' Add a reporting-period column to a sales table
#'
#' @param sales A data frame of till transactions.
#' @param dateCol <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the sale date.
#' @param unit One of `"week"`, `"month"`, `"quarter"` or `"fiscalYear"`.
#' @param name Name of the new column.
#' @return `sales` with an added period column.
#' @export
addPeriod <- function(sales, dateCol = saleDate, unit = "month", name = "period") {
  sales |>
    dplyr::mutate("{name}" := periodStart({{ dateCol }}, unit = unit))
}

#' Compare each period with the previous period
#'
#' @param summary A data frame with one row per period.
#' @param valueCol <[`data-masking`][dplyr::dplyr_data_masking]> Column to compare.
#' @param periodCol <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the period start.
#' @return `summary` with `previous`, `change` and `pctChange` columns.
#' @family period comparisons
#' @export
compareToPrevious <- function(summary, valueCol = revenue, periodCol = period) {
  summary |>
    dplyr::arrange({{ periodCol }}) |>
    dplyr::mutate(
      previous = dplyr::lag({{ valueCol }}),
      change = {{ valueCol }} - previous,
      pctChange = dplyr::if_else(previous == 0, NA_real_, change / previous)
    )
}

#' Compare each period with the same period one year earlier
#'
#' Periods are matched on their start date shifted back by one year, so
#' this works for weekly, monthly and quarterly summaries alike.
#'
#' @param summary A data frame with one row per period.
#' @param valueCol <[`data-masking`][dplyr::dplyr_data_masking]> Column to compare.
#' @param periodCol <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the period start.
#' @return `summary` with `previous`, `change` and `pctChange` columns.
#' @family period comparisons
#' @export
compareToLastYear <- function(summary, valueCol = revenue, periodCol = period) {
  lastYear <- summary |>
    dplyr::transmute(
      joinKey = {{ periodCol }} + lubridate::years(1),
      previous = {{ valueCol }}
    )
  summary |>
    dplyr::mutate(joinKey = {{ periodCol }}) |>
    dplyr::left_join(lastYear, by = "joinKey") |>
    dplyr::select(-"joinKey") |>
    dplyr::mutate(
      change = {{ valueCol }} - previous,
      pctChange = dplyr::if_else(previous == 0, NA_real_, change / previous)
    )
}

#' Fill in periods with no sales
#'
#' Summaries built from transactions skip periods without any sales; this
#' inserts those periods with zero values so charts and comparisons line up.
#'
#' @param summary A data frame with one row per period.
#' @param periodCol <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the period start.
#' @param unit One of `"week"`, `"month"` or `"quarter"`.
#' @param fill A named list of values used for the inserted rows.
#' @return `summary` with one row for every period in its range.
#' @export
completePeriods <- function(summary, periodCol = period,
                            unit = c("month", "week", "quarter"),
                            fill = list(revenue = 0, units = 0, transactions = 0L)) {
  unit <- match.arg(unit)
  step <- switch(unit,
    week = "1 week",
    month = "1 month",
    quarter = "3 months"
  )
  periods <- dplyr::pull(summary, {{ periodCol }})
  if (length(periods) == 0) {
    return(summary)
  }
  allPeriods <- seq(min(periods), max(periods), by = step)
  summary |>
    tidyr::complete({{ periodCol }} := allPeriods, fill = fill) |>
    dplyr::arrange({{ periodCol }})
}

#' Label weekly periods for reports
#'
#' @param starts A vector of period start dates.
#' @return A character vector such as `"2026-W07"`.
#' @family period labels
#' @export
labelWeeks <- function(starts) {
  starts <- lubridate::as_date(starts)
  sprintf("%d-W%02d", lubridate::isoyear(starts), lubridate::isoweek(starts))
}

#' Label monthly periods for reports
#'
#' @param starts A vector of period start dates.
#' @return A character vector such as `"2026-02"`.
#' @family period labels
#' @export
labelMonths <- function(starts) {
  starts <- lubridate::as_date(starts)
  sprintf("%d-%02d", lubridate::year(starts), lubridate::month(starts))
}

#' Label quarterly periods for reports
#'
#' @param starts A vector of period start dates.
#' @return A character vector such as `"2026-Q1"`.
#' @family period labels
#' @export
labelQuarters <- function(starts) {
  starts <- lubridate::as_date(starts)
  sprintf("%d-Q%d", lubridate::year(starts), lubridate::quarter(starts))
}
