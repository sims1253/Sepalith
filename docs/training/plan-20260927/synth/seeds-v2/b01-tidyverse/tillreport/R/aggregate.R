#' Summarise sales by arbitrary grouping columns
#'
#' The workhorse behind the `summariseBy*()` helpers: computes revenue,
#' units, transaction count and average basket value per group.
#'
#' @export
#' @param sales A data frame of till transactions with `revenue`, `units`
#'   and `transactionId` columns.
#' @param ... <[`data-masking`][dplyr::dplyr_data_masking]> Grouping columns.
#' @return A tibble with one row per group, sorted by descending revenue.
summariseSales <- function(sales, ...) {
  checkSalesColumns(sales)

  sales |>
    dplyr::group_by(...) |>
    dplyr::summarise(
      revenue = sum(.data$revenue, na.rm = TRUE),
      units = sum(.data$units, na.rm = TRUE),
      transactions = dplyr::n_distinct(.data$transactionId),
      .groups = "drop"
    ) |>
    dplyr::mutate(avgBasket = .data$revenue / .data$transactions) |>
    dplyr::arrange(dplyr::desc(.data$revenue))
}

#' Summarise sales per store
#'
#' @export
#' @param sales A data frame of till transactions with `revenue`, `units`
#'   and `transactionId` columns.
#' @param storeCol <[`data-masking`][dplyr::dplyr_data_masking]> Column identifying the store.
#' @return A tibble with one row per store.
summariseByStore <- function(sales, storeCol = storeId) {
  summariseSales(sales, {{ storeCol }})
}

#' Summarise sales per product category
#'
#' @export
#' @param sales A data frame of till transactions with `revenue`, `units`
#'   and `transactionId` columns.
#' @param categoryCol <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the category.
#' @return A tibble with one row per category.
summariseByCategory <- function(sales, categoryCol = category) {
  summariseSales(sales, {{ categoryCol }})
}

#' Summarise sales per ISO week
#'
#' @export
#' @param sales A data frame of till transactions with `revenue`, `units`
#'   and `transactionId` columns.
#' @param dateCol <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the sale date.
#' @return A tibble with one row per week, in calendar order.
summariseByWeek <- function(sales, dateCol = saleDate) {
  # Weeks start on Monday to match the trading calendar.
  sales |>
    dplyr::mutate(week = lubridate::floor_date({{ dateCol }}, "week", week_start = 1)) |>
    summariseSales(week) |>
    dplyr::arrange(week)
}

#' Find the best-selling products
#'
#' Ranks products by a chosen metric, optionally within each group, and
#' keeps the top `n` per group. Ties share the same rank.
#'
#' @export
#' @param sales A data frame of till transactions with `revenue`, `units`
#'   and `transactionId` columns.
#' @param n Number of products to keep per group.
#' @param by Metric to rank on, either `"revenue"` or `"units"`.
#' @param groupCol <[`data-masking`][dplyr::dplyr_data_masking]> Optional grouping column.
#' @return A tibble with `productId`, the metric and its `rank`.
topProducts <- function(sales, n = 10L, by = c("revenue", "units"), groupCol = NULL) {
  by <- match.arg(by)
  totals <- sales |>
    dplyr::group_by({{ groupCol }}, .data$productId) |>
    dplyr::summarise(value = sum(.data[[by]], na.rm = TRUE), .groups = "drop_last")

  # "drop_last" leaves the data grouped by `groupCol`, so ranks are per group.
  totals |>
    dplyr::mutate(rank = dplyr::min_rank(dplyr::desc(.data$value))) |>
    dplyr::filter(.data$rank <= n) |>
    dplyr::arrange(.data$rank, .by_group = TRUE) |>
    dplyr::ungroup() |>
    dplyr::rename("{by}" := value)
}

#' Find the worst-selling products
#'
#' Ranks products by a chosen metric, optionally within each group, and
#' keeps the bottom `n` per group. Ties share the same rank.
#'
#' @export
#' @param sales A data frame of till transactions with `revenue`, `units`
#'   and `transactionId` columns.
#' @param n Number of products to keep per group.
#' @param by Metric to rank on, either `"revenue"` or `"units"`.
#' @param groupCol <[`data-masking`][dplyr::dplyr_data_masking]> Optional grouping column.
#' @return A tibble with `productId`, the metric and its `rank`.
bottomProducts <- function(sales, n = 10L, by = c("revenue", "units"), groupCol = NULL) {
  by <- match.arg(by)
  totals <- sales |>
    dplyr::group_by({{ groupCol }}, .data$productId) |>
    dplyr::summarise(value = sum(.data[[by]], na.rm = TRUE), .groups = "drop_last")

  totals |>
    dplyr::mutate(rank = dplyr::min_rank(.data$value)) |>
    dplyr::filter(.data$rank <= n) |>
    dplyr::arrange(.data$rank, .by_group = TRUE) |>
    dplyr::ungroup() |>
    dplyr::rename("{by}" := value)
}

#' Add each row's share of a group total
#'
#' @export
#' @param summary A summarised sales table, e.g. from [summariseSales()].
#' @param valueCol <[`data-masking`][dplyr::dplyr_data_masking]> Column to compute shares of.
#' @param withinCol <[`data-masking`][dplyr::dplyr_data_masking]> Optional column defining
#'   the groups the shares sum to one within.
#' @return `summary` with an added `share` column.
addShare <- function(summary, valueCol = revenue, withinCol = NULL) {
  summary |>
    dplyr::group_by({{ withinCol }}) |>
    dplyr::mutate(share = {{ valueCol }} / sum({{ valueCol }}, na.rm = TRUE)) |>
    dplyr::ungroup()
}

#' Check that a sales table has the columns the summaries need
#'
#' @param sales A data frame of till transactions with `revenue`, `units`
#'   and `transactionId` columns.
#' @noRd
checkSalesColumns <- function(sales) {
  needed <- c("revenue", "units", "transactionId")
  missing <- setdiff(needed, names(sales))
  if (length(missing) > 0) {
    rlang::abort(c(
      "Sales data is missing required columns.",
      x = paste("Missing:", paste(missing, collapse = ", "))
    ))
  }
  invisible(sales)
}

#' Reconcile till takings against recorded sales
#'
#' Compares the cash and card totals declared at each till close with the
#' revenue recorded in the transaction log for the same store and day, and
#' classifies each difference.
#'
#' @export
#' @param sales A data frame of till transactions with `revenue`, `units`
#'   and `transactionId` columns.
#' @param closes A data frame of till closes with `storeId`, `closeDate`,
#'   `cash` and `card` columns.
#' @param tolerance Absolute difference below which a till counts as balanced.
#' @return A tibble with one row per store and day and a `status` column.
reconcileTills <- function(sales, closes, tolerance = 1) {
  checkSalesColumns(sales)
  needed <- c("storeId", "closeDate", "cash", "card")
  missing <- setdiff(needed, names(closes))
  if (length(missing) > 0) {
    rlang::abort(c(
      "Till close data is missing required columns.",
      x = paste("Missing:", paste(missing, collapse = ", "))
    ))
  }

  recorded <- sales |>
    dplyr::mutate(closeDate = lubridate::as_date(.data$saleDate)) |>
    dplyr::group_by(.data$storeId, .data$closeDate) |>
    dplyr::summarise(recorded = sum(.data$revenue, na.rm = TRUE), .groups = "drop")
  declared <- closes |>
    dplyr::mutate(closeDate = lubridate::as_date(.data$closeDate)) |>
    dplyr::group_by(.data$storeId, .data$closeDate) |>
    dplyr::summarise(
      declared = sum(.data$cash, na.rm = TRUE) + sum(.data$card, na.rm = TRUE),
      .groups = "drop"
    )

  # A full join keeps days with a close but no sales (and vice versa) so
  # that they show up in the status column instead of disappearing.
  dplyr::full_join(recorded, declared, by = c("storeId", "closeDate")) |>
    dplyr::mutate(
      difference = .data$declared - .data$recorded,
      status = dplyr::case_when(
        is.na(.data$declared) ~ "no close",
        is.na(.data$recorded) ~ "no sales",
        abs(.data$difference) < tolerance ~ "balanced",
        .data$difference > 0 ~ "over",
        .default = "short"
      ),
      status = factor(.data$status,
        levels = c("balanced", "over", "short", "no close", "no sales"))
    ) |>
    dplyr::arrange(.data$closeDate, .data$storeId)
}
