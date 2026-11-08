#' Guarded column statistics
#'
#' Each wrapper drops missing values when `naRm` is `TRUE`, returns `NA`
#' when fewer than `minN` values remain, rounds to `digits` when given and
#' records the number of values used and dropped in the attributes `"n"`
#' and `"missing"`.
#'
#' @param x A numeric vector.
#' @param naRm Drop missing values first?
#' @param minN Minimum number of values required.
#' @param digits Optional number of decimal places for rounding.
#' @return A single number with attributes `"n"` and `"missing"`.
#' @name wrenStats
#' @family statistic wrappers
NULL

#' @rdname wrenStats
#' @export
wrenMean <- function(x, naRm = TRUE, minN = 1L, digits = NULL) {
  if (!is.numeric(x)) {
    stop("`x` must be numeric, not ", class(x)[1L], ".", call. = FALSE)
  }
  if (!is.numeric(minN) || length(minN) != 1L || minN < 0) {
    stop("`minN` must be a single non-negative number.", call. = FALSE)
  }

  missing <- sum(is.na(x))
  if (naRm) {
    x <- x[!is.na(x)]
  }
  if (length(x) < minN) {
    return(structure(NA_real_, n = length(x), missing = missing))
  }

  value <- sum(x) / length(x)
  if (!is.null(digits)) {
    value <- round(value, digits)
  }
  structure(value, n = length(x), missing = missing)
}

#' @rdname wrenStats
#' @export
wrenMedian <- function(x, naRm = TRUE, minN = 1L, digits = NULL) {
  if (!is.numeric(x)) {
    stop("`x` must be numeric, not ", class(x)[1L], ".", call. = FALSE)
  }
  if (!is.numeric(minN) || length(minN) != 1L || minN < 0) {
    stop("`minN` must be a single non-negative number.", call. = FALSE)
  }

  missing <- sum(is.na(x))
  if (naRm) {
    x <- x[!is.na(x)]
  }
  if (length(x) < minN) {
    return(structure(NA_real_, n = length(x), missing = missing))
  }

  value <- stats::median(x)
  if (!is.null(digits)) {
    value <- round(value, digits)
  }
  structure(value, n = length(x), missing = missing)
}

#' @rdname wrenStats
#' @export
wrenSd <- function(x, naRm = TRUE, minN = 2L, digits = NULL) {
  if (!is.numeric(x)) {
    stop("`x` must be numeric, not ", class(x)[1L], ".", call. = FALSE)
  }
  if (!is.numeric(minN) || length(minN) != 1L || minN < 0) {
    stop("`minN` must be a single non-negative number.", call. = FALSE)
  }

  missing <- sum(is.na(x))
  if (naRm) {
    x <- x[!is.na(x)]
  }
  if (length(x) < minN) {
    return(structure(NA_real_, n = length(x), missing = missing))
  }

  value <- stats::sd(x)
  if (!is.null(digits)) {
    value <- round(value, digits)
  }
  structure(value, n = length(x), missing = missing)
}

#' @rdname wrenStats
#' @export
wrenIqr <- function(x, naRm = TRUE, minN = 2L, digits = NULL) {
  if (!is.numeric(x)) {
    stop("`x` must be numeric, not ", class(x)[1L], ".", call. = FALSE)
  }
  if (!is.numeric(minN) || length(minN) != 1L || minN < 0) {
    stop("`minN` must be a single non-negative number.", call. = FALSE)
  }

  missing <- sum(is.na(x))
  if (naRm) {
    x <- x[!is.na(x)]
  }
  if (length(x) < minN) {
    return(structure(NA_real_, n = length(x), missing = missing))
  }

  value <- stats::IQR(x)
  if (!is.null(digits)) {
    value <- round(value, digits)
  }
  structure(value, n = length(x), missing = missing)
}

#' Look up a guarded statistic by name
#'
#' @param stat One of `"mean"`, `"median"`, `"sd"` or `"iqr"`.
#' @return The matching function from [wrenStats].
#' @family statistic wrappers
#' @keywords internal
statFunction <- function(stat = c("mean", "median", "sd", "iqr")) {
  stat <- match.arg(stat)
  switch(stat,
    mean = wrenMean,
    median = wrenMedian,
    sd = wrenSd,
    iqr = wrenIqr
  )
}

#' Describe numeric columns with the guarded statistics
#'
#' @param data A data frame.
#' @param cols Names of numeric columns.
#' @param minN Minimum number of values; passed to every wrapper.
#' @return A data frame with one row per column and the columns `column`,
#'   `n`, `mean`, `median`, `sd`, `iqr` and `cv` (the coefficient of
#'   variation, `sd / mean`).
#' @family statistic wrappers
#' @export
describeColumns <- function(data, cols, minN = 2L) {
  unknown <- setdiff(cols, names(data))
  if (length(unknown) > 0L) {
    stop("Unknown columns: ", toString(unknown), call. = FALSE)
  }
  out <- data.frame(column = cols, stringsAsFactors = FALSE)

  out$n <- vapply(cols, function(col) {
    sum(!is.na(data[[col]]))
  }, integer(1), USE.NAMES = FALSE)
  out$mean <- vapply(cols, function(col) {
    as.numeric(wrenMean(data[[col]], minN = minN))
  }, numeric(1), USE.NAMES = FALSE)
  out$median <- vapply(cols, function(col) {
    as.numeric(wrenMedian(data[[col]], minN = minN))
  }, numeric(1), USE.NAMES = FALSE)
  out$sd <- vapply(cols, function(col) {
    as.numeric(wrenSd(data[[col]], minN = minN))
  }, numeric(1), USE.NAMES = FALSE)
  out$iqr <- vapply(cols, function(col) {
    as.numeric(wrenIqr(data[[col]], minN = minN))
  }, numeric(1), USE.NAMES = FALSE)

  # A zero mean has no meaningful coefficient of variation
  out$cv <- ifelse(is.na(out$mean) | out$mean == 0, NA_real_, out$sd / out$mean)
  out
}

#' Apply guarded statistics to groups of rows
#'
#' @param data A data frame.
#' @param group Name of the grouping column.
#' @param col Name of the numeric column.
#' @param stats Names passed to [statFunction()].
#' @return A matrix with one row per group and the columns `n` followed by
#'   one column per statistic.
#' @family statistic wrappers
#' @export
groupStats <- function(data, group, col, stats = c("mean", "median", "sd", "iqr")) {
  if (!all(c(group, col) %in% names(data))) {
    stop("Unknown columns: ", toString(setdiff(c(group, col), names(data))), call. = FALSE)
  }
  pieces <- split(data[[col]], data[[group]], drop = TRUE)

  # One row per group, one column per statistic after the count
  out <- matrix(
    NA_real_,
    nrow = length(pieces),
    ncol = length(stats) + 1L,
    dimnames = list(names(pieces), c("n", stats))
  )

  # Count first, then the statistics in the order given
  out[, "n"] <- vapply(pieces, function(x) sum(!is.na(x)), integer(1))
  for (stat in stats) {
    fun <- statFunction(stat)
    out[, stat] <- vapply(pieces, function(x) as.numeric(fun(x)), numeric(1))
  }
  out
}
