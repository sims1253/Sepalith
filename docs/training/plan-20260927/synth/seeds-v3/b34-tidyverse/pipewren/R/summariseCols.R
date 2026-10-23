#' Compute one summary statistic
#'
#' @param x A numeric vector.
#' @param stat One of `"n"`, `"mean"`, `"sd"`, `"median"`, `"min"` or
#'   `"max"`. Missing values are always dropped first.
#' @return A single number; `NA` when `x` has too few values.
#' @family column summaries
#' @export
statValue <- function(x, stat = c("n", "mean", "sd", "median", "min", "max")) {
  stat <- match.arg(stat)
  x <- x[!is.na(x)]
  if (stat != "n" && length(x) == 0L) {
    return(NA_real_)
  }
  switch(stat,
    n = as.numeric(length(x)),
    mean = mean(x),
    sd = if (length(x) < 2L) NA_real_ else stats::sd(x),
    median = stats::median(x),
    min = min(x),
    max = max(x)
  )
}

#' Summarise numeric columns into a long table
#'
#' @param data A data frame.
#' @param cols Names of numeric columns to summarise.
#' @param stats Statistics passed to [statValue()].
#' @return A data frame with a `column` column followed by one column per
#'   element of `stats`, one row per element of `cols`.
#' @family column summaries
#' @export
summaryTable <- function(data, cols, stats = c("n", "mean", "sd", "min", "max")) {
  unknown <- setdiff(cols, names(data))
  if (length(unknown) > 0L) {
    stop("Unknown columns: ", toString(unknown), call. = FALSE)
  }
  out <- data.frame(column = cols, stringsAsFactors = FALSE)
  for (stat in stats) {
    out[[stat]] <- vapply(cols, function(col) {
      statValue(data[[col]], stat)
    }, numeric(1), USE.NAMES = FALSE)
  }
  out
}

#' Summarise sales metrics by group
#'
#' @description
#' Computes the count, mean, median and standard deviation of `units`,
#' `revenue`, `cost` and `discount` within each group.
#'
#' @param data A data frame of sales with columns `units`, `revenue`, `cost`
#'   and `discount`.
#' @param groupVars Character vector of grouping columns.
#' @return A tibble with the grouping columns followed by `units_n`,
#'   `units_mean`, `units_median`, `units_sd`, `revenue_n`, `revenue_mean`,
#'   `revenue_median`, `revenue_sd`, `cost_n`, `cost_mean`, `cost_median`,
#'   `cost_sd`, `discount_n`, `discount_mean`, `discount_median` and
#'   `discount_sd`.
#' @family column summaries
#' @importFrom dplyr group_by across all_of summarise
#' @export
summariseSales <- function(data, groupVars = "region") {
  data |>
    group_by(across(all_of(groupVars))) |>
    summarise(
      units_n = sum(!is.na(units)),
      units_mean = mean(units, na.rm = TRUE),
      units_median = stats::median(units, na.rm = TRUE),
      units_sd = stats::sd(units, na.rm = TRUE),
      revenue_n = sum(!is.na(revenue)),
      revenue_mean = mean(revenue, na.rm = TRUE),
      revenue_median = stats::median(revenue, na.rm = TRUE),
      revenue_sd = stats::sd(revenue, na.rm = TRUE),
      cost_n = sum(!is.na(cost)),
      cost_mean = mean(cost, na.rm = TRUE),
      cost_median = stats::median(cost, na.rm = TRUE),
      cost_sd = stats::sd(cost, na.rm = TRUE),
      discount_n = sum(!is.na(discount)),
      discount_mean = mean(discount, na.rm = TRUE),
      discount_median = stats::median(discount, na.rm = TRUE),
      discount_sd = stats::sd(discount, na.rm = TRUE),
      .groups = "drop"
    )
}

#' Share of rows, units and revenue per group
#'
#' @param data A data frame of sales with columns `units` and `revenue`.
#' @param group Name of the grouping column.
#' @return A tibble sorted by decreasing revenue with columns `group`,
#'   `rows`, `rows_share`, `rows_cumshare`, `units`, `units_share`,
#'   `units_cumshare`, `revenue`, `revenue_share` and `revenue_cumshare`.
#' @family column summaries
#' @importFrom dplyr summarise mutate arrange desc n
#' @export
summariseShares <- function(data, group) {
  totals <- data |>
    group_by(group = .data[[group]]) |>
    summarise(
      rows = n(),
      units = sum(units, na.rm = TRUE),
      revenue = sum(revenue, na.rm = TRUE),
      .groups = "drop"
    ) |>
    arrange(desc(revenue))

  # Shares and running shares in revenue order
  totals |>
    mutate(
      rows_share = rows / sum(rows),
      rows_cumshare = cumsum(rows_share),
      units_share = units / sum(units),
      units_cumshare = cumsum(units_share),
      revenue_share = revenue / sum(revenue),
      revenue_cumshare = cumsum(revenue_share)
    ) |>
    dplyr::select(
      group,
      rows, rows_share, rows_cumshare,
      units, units_share, units_cumshare,
      revenue, revenue_share, revenue_cumshare
    )
}

#' Summarise columns within groups using base R
#'
#' @param data A data frame.
#' @param group Name of the grouping column.
#' @param cols Names of numeric columns to summarise.
#' @param stat A statistic passed to [statValue()].
#' @param overall Append a row named `"(all)"` computed over all rows?
#' @return A data frame with the columns `group` and `n` followed by one
#'   column per element of `cols`, one row per group in sorted order.
#' @family column summaries
#' @export
summariseByGroup <- function(data, group, cols, stat = "mean", overall = TRUE) {
  unknown <- setdiff(c(group, cols), names(data))
  if (length(unknown) > 0L) {
    stop("Unknown columns: ", toString(unknown), call. = FALSE)
  }
  pieces <- split(data[cols], data[[group]], drop = TRUE)

  # One row per group
  out <- data.frame(group = names(pieces), stringsAsFactors = FALSE)
  out$n <- vapply(pieces, nrow, integer(1), USE.NAMES = FALSE)
  for (col in cols) {
    out[[col]] <- vapply(pieces, function(piece) {
      statValue(piece[[col]], stat)
    }, numeric(1), USE.NAMES = FALSE)
  }

  # The same statistics over all rows
  if (overall) {
    total <- data.frame(group = "(all)", stringsAsFactors = FALSE)
    total$n <- nrow(data)
    for (col in cols) {
      total[[col]] <- statValue(data[[col]], stat)
    }
    out <- rbind(out, total)
  }
  out
}

#' Compare column summaries before and after a change
#'
#' @description
#' Summarises the same columns in two data frames and reports the change in
#' count, mean and median.
#'
#' @param before A data frame before the change.
#' @param after A data frame after the change.
#' @param cols Names of numeric columns present in both.
#' @return A data frame with columns `column`, `n_before`, `n_after`,
#'   `n_change`, `mean_before`, `mean_after`, `mean_change`,
#'   `median_before`, `median_after` and `median_change`.
#' @family column summaries
#' @export
compareSummaries <- function(before, after, cols) {
  missingBefore <- setdiff(cols, names(before))
  if (length(missingBefore) > 0L) {
    stop("Unknown columns in `before`: ", toString(missingBefore), call. = FALSE)
  }
  missingAfter <- setdiff(cols, names(after))
  if (length(missingAfter) > 0L) {
    stop("Unknown columns in `after`: ", toString(missingAfter), call. = FALSE)
  }

  stats <- c("n", "mean", "median")
  lhs <- summaryTable(before, cols, stats)
  rhs <- summaryTable(after, cols, stats)
  out <- data.frame(column = cols, stringsAsFactors = FALSE)

  # Count
  out$n_before <- lhs$n
  out$n_after <- rhs$n
  out$n_change <- rhs$n - lhs$n

  # Mean
  out$mean_before <- lhs$mean
  out$mean_after <- rhs$mean
  out$mean_change <- rhs$mean - lhs$mean

  # Median
  out$median_before <- lhs$median
  out$median_after <- rhs$median
  out$median_change <- rhs$median - lhs$median
  out
}
