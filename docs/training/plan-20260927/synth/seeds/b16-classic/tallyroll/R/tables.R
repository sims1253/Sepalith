#' Range of each row of a matrix
#'
#' @param m A numeric matrix.
#' @param na.rm If `TRUE`, missing values are dropped before computing.
#' @return A two-column matrix with columns `min` and `max`, one row per row
#'   of `m`.
#' @export
rowSpans <- function(m, na.rm = FALSE) {
  out <- t(apply(m, 1, range, na.rm = na.rm))
  colnames(out) <- c("min", "max")
  rownames(out) <- rownames(m)
  out
}

#' Range of each column of a matrix
#'
#' @param m A numeric matrix.
#' @param na.rm If `TRUE`, missing values are dropped before computing.
#' @return A two-column matrix with columns `min` and `max`, one row per
#'   column of `m`.
#' @export
colSpans <- function(m, na.rm = FALSE) {
  out <- t(apply(m, 2, range, na.rm = na.rm))
  colnames(out) <- c("min", "max")
  rownames(out) <- colnames(m)
  out
}

#' Mean of a column within groups
#'
#' @param data A data frame of transactions.
#' @param value Name of the numeric column to summarise.
#' @param by Name of the grouping column.
#' @return A named numeric vector with one mean per group.
#' @export
groupMeans <- function(data, value, by) {
  stopifnot(value %in% names(data), by %in% names(data))
  tapply(data[[value]], data[[by]], mean, na.rm = TRUE)
}

#' Total of a column within groups
#'
#' @param data A data frame of transactions.
#' @param value Name of the numeric column to summarise.
#' @param by Name of the grouping column.
#' @return A named numeric vector with one total per group.
#' @export
groupTotals <- function(data, value, by) {
  stopifnot(value %in% names(data), by %in% names(data))
  tapply(data[[value]], data[[by]], sum, na.rm = TRUE)
}

#' Monthly totals of a value column
#'
#' Adds a `month` column in `YYYY-MM` form and sums the value column within
#' each month and optional extra grouping columns.
#'
#' @param data A data frame of transactions.
#' @param value Name of the numeric column to summarise.
#' @param date Name of the `Date` column.
#' @param by Optional character vector of extra grouping columns.
#' @return A data frame with the grouping columns, `month` and the summed
#'   value column, ordered by month.
#' @export
monthlyTotals <- function(data, value, date = "date", by = NULL) {
  if (!inherits(data[[date]], "Date")) {
    stop("column '", date, "' must be of class Date", call. = FALSE)
  }
  data$month <- format(data[[date]], "%Y-%m")
  groups <- c(by, "month")
  f <- reformulate(groups, response = value)
  out <- aggregate(f, data = data, FUN = sum)
  out[order(out$month), , drop = FALSE]
}

#' Attach labels from a lookup table
#'
#' Left-joins `lookup` onto `data` and reports keys that found no match.
#'
#' @param data A data frame of transactions.
#' @param lookup A data frame with one row per key.
#' @param by Name of the key column shared by both tables.
#' @param warnMissing If `TRUE`, warn about keys absent from `lookup`.
#' @return `data` with the lookup columns added, in its original row order.
#' @export
attachLookup <- function(data, lookup, by, warnMissing = TRUE) {
  if (anyDuplicated(lookup[[by]])) {
    stop("lookup keys in '", by, "' must be unique", call. = FALSE)
  }
  data$.row <- seq_len(nrow(data))
  out <- merge(data, lookup, by = by, all.x = TRUE, sort = FALSE)
  missing <- setdiff(unique(data[[by]]), lookup[[by]])
  if (warnMissing && length(missing) > 0) {
    warning(sprintf("%d key(s) not found in lookup: %s", length(missing),
                    paste(head(missing, 5), collapse = ", ")), call. = FALSE)
  }
  out <- out[order(out$.row), , drop = FALSE]
  out$.row <- NULL
  rownames(out) <- NULL
  out
}

#' Reshape a long table to wide
#'
#' @param data A data frame in long form.
#' @param id Name of the column identifying each unit.
#' @param time Name of the column whose values become new columns.
#' @param value Name of the column holding the measurements.
#' @return A wide data frame with one row per unit.
#' @export
toWide <- function(data, id, time, value) {
  sub <- data[, c(id, time, value)]
  wide <- reshape(sub, idvar = id, timevar = time, v.names = value,
                  direction = "wide", sep = "_")
  rownames(wide) <- NULL
  attr(wide, "reshapeWide") <- NULL
  wide
}

#' Reshape a wide table to long
#'
#' @param data A data frame in wide form.
#' @param id Name of the column identifying each unit.
#' @param cols Names of the columns to stack.
#' @param value Name of the new column holding the measurements.
#' @param time Name of the new column holding the former column names.
#' @return A long data frame with one row per unit and stacked column.
#' @export
toLong <- function(data, id, cols, value = "value", time = "key") {
  long <- reshape(data[, c(id, cols)], idvar = id, varying = cols,
                  v.names = value, timevar = time, times = cols,
                  direction = "long")
  rownames(long) <- NULL
  attr(long, "reshapeLong") <- NULL
  long[order(long[[id]], long[[time]]), , drop = FALSE]
}
