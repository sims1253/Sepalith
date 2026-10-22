#' Summarise numeric columns by group
#'
#' Computes one summary statistic for every column in `cols` within each
#' group defined by `by`, using `.SD` and `.SDcols`.
#'
#' @param dt A data.table of records to summarise.
#' @param cols Character vector of numeric columns to summarise.
#' @param by Character vector of grouping columns.
#' @param stat Summary statistic to apply.
#' @return A data.table with one row per group.
#' @export
summarise_by = function(dt, cols, by, stat = c("sum", "mean", "median", "min", "max")) {
  stat = match.arg(stat)
  check_has_cols(dt, c(cols, by), "summarise_by")
  fun = switch(stat,
    sum = function(x) sum(x, na.rm = TRUE),
    mean = function(x) mean(x, na.rm = TRUE),
    median = function(x) stats::median(x, na.rm = TRUE),
    min = function(x) suppressWarnings(min(x, na.rm = TRUE)),
    max = function(x) suppressWarnings(max(x, na.rm = TRUE))
  )
  out = dt[, lapply(.SD, fun), by = by, .SDcols = cols]
  data.table::setnames(out, cols, paste(cols, stat, sep = "_"))
  out
}

#' Count rows and distinct values by group
#'
#' @param dt A data.table of records to summarise.
#' @param by Character vector of grouping columns.
#' @param distinct Optional column whose distinct values are counted per group.
#' @return A data.table with columns `n` and, if requested, `n_distinct`.
#' @export
count_by = function(dt, by, distinct = NULL) {
  check_has_cols(dt, c(by, distinct), "count_by")
  if (is.null(distinct)) {
    out = dt[, .(n = .N), by = by]
  } else {
    out = dt[, .(n = .N, n_distinct = data.table::uniqueN(get(distinct))), by = by]
  }
  data.table::setorderv(out, "n", order = -1L)
  out
}

#' Add a running total within groups
#'
#' Rows are ordered by `order_col` before the cumulative sum is taken, and the
#' result is written back by reference.
#'
#' @param dt A data.table of records to summarise.
#' @param value_col Name of the numeric column to accumulate.
#' @param order_col Name of the column defining row order within groups.
#' @param by Character vector of grouping columns.
#' @return `dt`, modified by reference and returned invisibly.
#' @export
add_running_total = function(dt, value_col, order_col, by = NULL) {
  check_has_cols(dt, c(value_col, order_col, by), "running_total")
  data.table::setorderv(dt, c(by, order_col))
  new_col = paste0(value_col, "_cum")
  dt[, (new_col) := cumsum(data.table::fcoalesce(get(value_col), 0)), by = by]
  invisible(dt)
}

#' Add a lagged copy of a column within groups
#'
#' @param dt A data.table of records to summarise.
#' @param value_col Name of the column to shift.
#' @param order_col Name of the column defining row order within groups.
#' @param by Character vector of grouping columns.
#' @param n Number of rows to shift by.
#' @return `dt`, modified by reference and returned invisibly.
#' @export
add_lag = function(dt, value_col, order_col, by = NULL, n = 1L) {
  check_has_cols(dt, c(value_col, order_col, by), "lag")
  data.table::setorderv(dt, c(by, order_col))
  new_col = paste0(value_col, "_lag", n)
  dt[, (new_col) := data.table::shift(get(value_col), n = n, type = "lag"), by = by]
  invisible(dt)
}

#' Add a leading copy of a column within groups
#'
#' @param dt A data.table of records to summarise.
#' @param value_col Name of the column to shift.
#' @param order_col Name of the column defining row order within groups.
#' @param by Character vector of grouping columns.
#' @param n Number of rows to shift by.
#' @return `dt`, modified by reference and returned invisibly.
#' @export
add_lead = function(dt, value_col, order_col, by = NULL, n = 1L) {
  check_has_cols(dt, c(value_col, order_col, by), "lead")
  data.table::setorderv(dt, c(by, order_col))
  new_col = paste0(value_col, "_lead", n)
  dt[, (new_col) := data.table::shift(get(value_col), n = n, type = "lead"), by = by]
  invisible(dt)
}

#' Pivot a long table to wide with one column per category
#'
#' A wrapper around [data.table::dcast()] that builds the formula from
#' character arguments and fills empty cells with `fill`.
#'
#' @param dt A data.table of records to summarise.
#' @param id_cols Character vector of columns identifying each output row.
#' @param names_from Name of the column whose values become new columns.
#' @param values_from Name of the column holding the cell values.
#' @param fill Value used for missing combinations.
#' @return A wide data.table keyed by `id_cols`.
#' @export
pivot_wide_dt = function(dt, id_cols, names_from, values_from, fill = NA) {
  check_has_cols(dt, c(id_cols, names_from, values_from), "pivot_wide")
  dupes = dt[, .N, by = c(id_cols, names_from)][N > 1L]
  if (nrow(dupes) > 0L) {
    warning(nrow(dupes), " duplicated cell(s); keeping the first value.", call. = FALSE)
  }
  lhs = paste(id_cols, collapse = " + ")
  f = stats::as.formula(paste(lhs, "~", names_from))
  out = data.table::dcast(
    dt,
    f,
    value.var = values_from,
    fun.aggregate = function(x) x[1L],
    fill = fill
  )
  data.table::setkeyv(out, id_cols)
  out
}

#' Pivot a wide table to long format
#'
#' @param dt A data.table of records to summarise.
#' @param id_cols Character vector of columns identifying each output row.
#' @param names_to Name of the new column holding former column names.
#' @param values_to Name of the new column holding the values.
#' @return A long data.table.
#' @export
pivot_long_dt = function(dt, id_cols, names_to = "name", values_to = "value") {
  check_has_cols(dt, id_cols, "pivot_long")
  measure = setdiff(names(dt), id_cols)
  if (length(measure) == 0L) {
    stop("No columns left to pivot after removing `id_cols`.", call. = FALSE)
  }
  out = data.table::melt(
    dt,
    id.vars = id_cols,
    measure.vars = measure,
    variable.name = names_to,
    value.name = values_to,
    variable.factor = FALSE
  )
  out
}

#' Build a daily report with period-over-period changes
#'
#' Aggregates `value_cols` to one row per day and group, fills in days with no
#' records, and adds the change against the previous day and the same weekday
#' one week earlier.
#'
#' @param dt A data.table of records to summarise.
#' @param date_col Name of the date or date-time column.
#' @param value_cols Character vector of numeric columns to summarise.
#' @param by Character vector of grouping columns.
#' @return A data.table with one row per day and group.
#' @export
daily_report = function(dt, date_col, value_cols, by = NULL) {
  check_has_cols(dt, c(date_col, value_cols, by), "daily_report")
  work = dt[, c(by, date_col, value_cols), with = FALSE]
  work[, day := as.Date(get(date_col))]
  daily = work[, lapply(.SD, sum, na.rm = TRUE), by = c(by, "day"), .SDcols = value_cols]
  # complete the calendar so lags line up with real days
  days = seq(min(daily$day), max(daily$day), by = "day")
  if (is.null(by)) {
    grid = data.table::data.table(day = days)
  } else {
    groups = unique(daily[, by, with = FALSE])
    grid = groups[, .(day = days), by = by]
  }
  daily = daily[grid, on = c(by, "day")]
  for (col in value_cols) {
    data.table::set(daily, which(is.na(daily[[col]])), col, 0)
  }
  data.table::setorderv(daily, c(by, "day"))
  for (col in value_cols) {
    d1 = paste0(col, "_d1")
    w1 = paste0(col, "_w1")
    daily[, (d1) := get(col) - data.table::shift(get(col), 1L), by = by]
    daily[, (w1) := get(col) - data.table::shift(get(col), 7L), by = by]
  }
  daily[, weekday := weekdays(day, abbreviate = TRUE)]
  data.table::setcolorder(daily, c(by, "day", "weekday"))
  daily
}
