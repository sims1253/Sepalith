#' Classify a column for profiling
#'
#' @param x A column.
#' @keywords internal
#' @noRd
column_type <- function(x) {
  if (is.logical(x)) {
    "logical"
  } else if (is.factor(x)) {
    "factor"
  } else if (is.numeric(x)) {
    "numeric"
  } else if (is.character(x)) {
    "character"
  } else {
    "other"
  }
}

#' Profile a numeric column
#'
#' @param x A column.
#' @param name The column name used in the output.
#' @return A one-row data frame with columns `column`, `type`, `n`,
#'   `n_missing`, `n_distinct`, `center` and `spread`.
#' @family column profilers
#' @export
profile_numeric <- function(x, name = "x") {
  ok <- x[!is.na(x)]

  # Center and spread of the non-missing values
  center <- if (length(ok) > 0L) mean(ok) else NA_real_
  spread <- if (length(ok) > 1L) stats::sd(ok) else NA_real_

  data.frame(
    column = name,
    type = "numeric",
    n = length(x),
    n_missing = sum(is.na(x)),
    n_distinct = length(unique(ok)),
    center = center,
    spread = spread,
    stringsAsFactors = FALSE
  )
}

#' Profile a logical column
#'
#' The center is the proportion of `TRUE` values and the spread is the
#' Bernoulli standard deviation.
#'
#' @inheritParams profile_numeric
#' @return A one-row data frame with columns `column`, `type`, `n`,
#'   `n_missing`, `n_distinct`, `center` and `spread`.
#' @family column profilers
#' @export
profile_logical <- function(x, name = "x") {
  ok <- x[!is.na(x)]

  # Center and spread of the non-missing values
  center <- if (length(ok) > 0L) mean(ok) else NA_real_
  spread <- if (length(ok) > 0L) sqrt(center * (1 - center)) else NA_real_

  data.frame(
    column = name,
    type = "logical",
    n = length(x),
    n_missing = sum(is.na(x)),
    n_distinct = length(unique(ok)),
    center = center,
    spread = spread,
    stringsAsFactors = FALSE
  )
}

#' Profile a factor column
#'
#' The center is the proportion of the modal level and the spread is the
#' Shannon entropy of the level frequencies divided by its maximum.
#'
#' @inheritParams profile_numeric
#' @return A one-row data frame with columns `column`, `type`, `n`,
#'   `n_missing`, `n_distinct`, `center` and `spread`.
#' @family column profilers
#' @export
profile_factor <- function(x, name = "x") {
  ok <- x[!is.na(x)]

  # Center and spread of the non-missing values
  freq <- tabulate(as.integer(ok), nbins = nlevels(x)) / max(length(ok), 1L)
  freq <- freq[freq > 0]
  center <- if (length(ok) > 0L) max(freq) else NA_real_
  spread <- if (length(freq) > 1L) -sum(freq * log(freq)) / log(length(freq)) else 0

  data.frame(
    column = name,
    type = "factor",
    n = length(x),
    n_missing = sum(is.na(x)),
    n_distinct = length(unique(ok)),
    center = center,
    spread = spread,
    stringsAsFactors = FALSE
  )
}

#' Profile a character column
#'
#' The center and spread describe the string lengths in characters.
#'
#' @inheritParams profile_numeric
#' @return A one-row data frame with columns `column`, `type`, `n`,
#'   `n_missing`, `n_distinct`, `center` and `spread`.
#' @family column profilers
#' @export
profile_character <- function(x, name = "x") {
  ok <- x[!is.na(x)]

  # Center and spread of the non-missing values
  len <- nchar(ok)
  center <- if (length(ok) > 0L) mean(len) else NA_real_
  spread <- if (length(ok) > 1L) stats::sd(len) else NA_real_

  data.frame(
    column = name,
    type = "character",
    n = length(x),
    n_missing = sum(is.na(x)),
    n_distinct = length(unique(ok)),
    center = center,
    spread = spread,
    stringsAsFactors = FALSE
  )
}

#' Profile every column of a data frame
#'
#' @param data A data frame.
#' @param cols Character vector of column names to profile.
#' @return A data frame with one row per column, binding the results of the
#'   `profile_*()` family.
#' @export
profile_columns <- function(data, cols = names(data)) {
  absent <- setdiff(cols, names(data))
  if (length(absent) > 0L) {
    stop("Unknown columns: ", paste(absent, collapse = ", "), call. = FALSE)
  }

  rows <- vector("list", length(cols))
  for (i in seq_along(cols)) {
    x <- data[[cols[i]]]
    rows[[i]] <- switch(column_type(x),
      numeric = profile_numeric(x, cols[i]),
      logical = profile_logical(x, cols[i]),
      factor = profile_factor(x, cols[i]),
      character = profile_character(x, cols[i]),
      stop(sprintf("Column `%s` has an unsupported type.", cols[i]), call. = FALSE)
    )
  }
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' Locate the range of numeric columns
#'
#' @param data A data frame.
#' @param cols Character vector of numeric column names.
#' @param finite Logical; ignore infinite values as well as missing ones?
#' @return A data frame with columns `column`, `lower`, `lower_row`,
#'   `lower_ties`, `lower_gap`, `upper`, `upper_row`, `upper_ties` and
#'   `upper_gap`. The `_row` columns give the first row holding the extreme
#'   value, `_ties` how many rows hold it, and `_gap` the distance to the
#'   next distinct value inwards.
#' @export
column_ranges <- function(data, cols = names(data), finite = TRUE) {
  k <- length(cols)
  lower <- upper <- lower_gap <- upper_gap <- rep(NA_real_, k)
  lower_row <- upper_row <- lower_ties <- upper_ties <- rep(NA_integer_, k)
  for (i in seq_len(k)) {
    v <- data[[cols[i]]]
    keep <- if (finite) is.finite(v) else !is.na(v)
    if (!any(keep)) {
      next
    }

    values <- sort(unique(v[keep]))
    # Lower bound
    lower[i] <- values[1L]
    lower_row[i] <- which(keep & v == lower[i])[1L]
    lower_ties[i] <- sum(keep & v == lower[i])
    lower_gap[i] <- if (length(values) > 1L) values[2L] - values[1L] else NA_real_
    # Upper bound
    upper[i] <- values[length(values)]
    upper_row[i] <- which(keep & v == upper[i])[1L]
    upper_ties[i] <- sum(keep & v == upper[i])
    upper_gap[i] <- if (length(values) > 1L) {
      values[length(values)] - values[length(values) - 1L]
    } else {
      NA_real_
    }
  }
  data.frame(
    column = cols,
    lower = lower,
    lower_row = lower_row,
    lower_ties = lower_ties,
    lower_gap = lower_gap,
    upper = upper,
    upper_row = upper_row,
    upper_ties = upper_ties,
    upper_gap = upper_gap,
    stringsAsFactors = FALSE
  )
}

#' Compare column profiles before and after a processing step
#'
#' @param before A data frame before processing.
#' @param after A data frame after processing.
#' @param cols Character vector of column names present in both.
#' @return A data frame with columns `column`, `type`, `n_before`,
#'   `n_after`, `missing_before`, `missing_after`, `center_before`,
#'   `center_after`, `center_change`, `spread_before`, `spread_after` and
#'   `spread_ratio`.
#' @export
compare_profiles <- function(before, after,
                             cols = intersect(names(before), names(after))) {
  absent <- setdiff(cols, intersect(names(before), names(after)))
  if (length(absent) > 0L) {
    stop("Columns missing from `before` or `after`: ", paste(absent, collapse = ", "),
         call. = FALSE)
  }

  # Profile `before`
  prof_before <- profile_columns(before, cols)
  # Profile `after`
  prof_after <- profile_columns(after, cols)

  changed <- prof_before$type != prof_after$type
  if (any(changed)) {
    warning("Column types changed: ", paste(cols[changed], collapse = ", "), call. = FALSE)
  }

  data.frame(
    column = cols,
    type = prof_after$type,
    n_before = prof_before$n,
    n_after = prof_after$n,
    missing_before = prof_before$n_missing,
    missing_after = prof_after$n_missing,
    center_before = prof_before$center,
    center_after = prof_after$center,
    center_change = prof_after$center - prof_before$center,
    spread_before = prof_before$spread,
    spread_after = prof_after$spread,
    spread_ratio = prof_after$spread / prof_before$spread,
    stringsAsFactors = FALSE
  )
}
