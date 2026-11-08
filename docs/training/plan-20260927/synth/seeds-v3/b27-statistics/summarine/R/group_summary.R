#' Apply a statistic to a vector with a minimum size
#'
#' @param v A numeric vector.
#' @param fun A function returning a single number.
#' @param min_n Minimum number of non-missing values required.
#' @keywords internal
#' @noRd
safe_stat <- function(v, fun, min_n = 1L) {
  v <- v[!is.na(v)]
  if (length(v) < min_n) {
    return(NA_real_)
  }
  as.numeric(fun(v))
}

#' Summarise a numeric vector within groups
#'
#' Splits `x` by `group` and computes the count, mean, standard deviation,
#' standard error, quartiles, minimum and maximum for every level.
#'
#' @param x A numeric vector.
#' @param group A factor or vector of group labels, the same length as `x`.
#' @param na_rm Logical; drop missing values before summarising?
#' @return A data frame with columns `group`, `n`, `mean`, `sd`, `se`,
#'   `min`, `q1`, `median`, `q3` and `max`, one row per level of `group`.
#' @export
group_summary <- function(x, group, na_rm = TRUE) {
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }
  if (length(group) != length(x)) {
    stop("`group` must have the same length as `x`.", call. = FALSE)
  }
  group <- as.factor(group)
  pieces <- split(x, group)

  q1 <- function(v) stats::quantile(v, 0.25, names = FALSE)
  q3 <- function(v) stats::quantile(v, 0.75, names = FALSE)

  n <- vapply(pieces, function(v) sum(!is.na(v)), integer(1))
  out <- data.frame(
    group = levels(group),
    n = n,
    mean = vapply(pieces, safe_stat, numeric(1), fun = mean),
    sd = vapply(pieces, safe_stat, numeric(1), fun = stats::sd, min_n = 2L),
    se = vapply(pieces, safe_stat, numeric(1), fun = stats::sd, min_n = 2L) / sqrt(n),
    min = vapply(pieces, safe_stat, numeric(1), fun = min),
    q1 = vapply(pieces, safe_stat, numeric(1), fun = q1),
    median = vapply(pieces, safe_stat, numeric(1), fun = stats::median),
    q3 = vapply(pieces, safe_stat, numeric(1), fun = q3),
    max = vapply(pieces, safe_stat, numeric(1), fun = max),
    stringsAsFactors = FALSE
  )
  # Without `na_rm`, any missing value poisons the group statistics
  if (!na_rm) {
    has_na <- vapply(pieces, anyNA, logical(1))
    stat_cols <- c("mean", "sd", "se", "min", "q1", "median", "q3", "max")
    out[has_na, stat_cols] <- NA_real_
  }
  rownames(out) <- NULL
  out
}

#' Quantiles of a numeric vector within groups
#'
#' @param x A numeric vector.
#' @param group A factor or vector of group labels, the same length as `x`.
#' @param probs Numeric vector of probabilities in `[0, 1]`.
#' @param type Quantile algorithm passed to [stats::quantile()].
#' @return A matrix with one row per group and one column per probability,
#'   with column names such as `"q25"`.
#' @export
group_quantiles <- function(x, group, probs = c(0.25, 0.5, 0.75), type = 7L) {
  if (any(probs < 0 | probs > 1)) {
    stop("`probs` must lie in [0, 1].", call. = FALSE)
  }
  group <- as.factor(group)
  pieces <- split(x, group)

  out <- matrix(NA_real_, nrow = length(pieces), ncol = length(probs))
  for (i in seq_along(pieces)) {
    v <- pieces[[i]]
    v <- v[!is.na(v)]
    if (length(v) > 0L) {
      out[i, ] <- stats::quantile(v, probs = probs, type = type, names = FALSE)
    }
  }
  dimnames(out) <- list(levels(group), paste0("q", round(100 * probs)))
  out
}

#' Pooled standard deviation of two samples
#'
#' @param a,b Numeric vectors.
#' @keywords internal
#' @noRd
pooled_sd <- function(a, b) {
  a <- a[!is.na(a)]
  b <- b[!is.na(b)]
  df <- length(a) + length(b) - 2
  if (df < 1) {
    return(NA_real_)
  }
  ss_a <- if (length(a) > 1L) (length(a) - 1) * stats::var(a) else 0
  ss_b <- if (length(b) > 1L) (length(b) - 1) * stats::var(b) else 0
  sqrt((ss_a + ss_b) / df)
}

#' Compare every group with a reference group
#'
#' @param x A numeric vector.
#' @param group A factor or vector of group labels, the same length as `x`.
#' @param ref Label of the reference group; defaults to the first level.
#' @return A data frame with one row per non-reference group and columns
#'   `group`, `mean_diff`, `median_diff`, `sd_ratio` and `effect_size`.
#' @details
#' 1. `mean_diff` is the group mean minus the reference mean.
#' 2. `median_diff` is the group median minus the reference median.
#' 3. `sd_ratio` is the group standard deviation over the reference one.
#' 4. `effect_size` is `mean_diff` divided by the pooled standard deviation.
#' @export
group_compare <- function(x, group, ref = NULL) {
  group <- as.factor(group)
  if (is.null(ref)) {
    ref <- levels(group)[1L]
  }
  if (!ref %in% levels(group)) {
    stop("`ref` must be a level of `group`.", call. = FALSE)
  }
  pieces <- lapply(split(x, group), function(v) v[!is.na(v)])
  base <- pieces[[ref]]
  others <- setdiff(levels(group), ref)

  # 1. Difference in means
  mean_diff <- vapply(others, function(g) {
    mean(pieces[[g]]) - mean(base)
  }, numeric(1))
  # 2. Difference in medians
  median_diff <- vapply(others, function(g) {
    stats::median(pieces[[g]]) - stats::median(base)
  }, numeric(1))
  # 3. Ratio of standard deviations
  sd_ratio <- vapply(others, function(g) {
    stats::sd(pieces[[g]]) / stats::sd(base)
  }, numeric(1))
  # 4. Standardised mean difference
  effect_size <- vapply(others, function(g) {
    (mean(pieces[[g]]) - mean(base)) / pooled_sd(pieces[[g]], base)
  }, numeric(1))

  data.frame(
    group = others,
    mean_diff = unname(mean_diff),
    median_diff = unname(median_diff),
    sd_ratio = unname(sd_ratio),
    effect_size = unname(effect_size),
    stringsAsFactors = FALSE
  )
}

#' Summarise several columns of a data frame by groups
#'
#' @param data A data frame.
#' @param value_cols Character vector of numeric column names to summarise.
#' @param by Name of the grouping column.
#' @param stat Statistic to compute: one of `"mean"`, `"median"`, `"sd"`,
#'   `"iqr"` or `"mad"`.
#' @return A data frame with the column `by` followed by one column per
#'   entry of `value_cols`.
#' @export
summarise_by <- function(data, value_cols, by,
                         stat = c("mean", "median", "sd", "iqr", "mad")) {
  stat <- match.arg(stat)
  missing_cols <- setdiff(c(value_cols, by), names(data))
  if (length(missing_cols) > 0L) {
    stop("Unknown columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  summarise_one <- switch(stat,
    mean = function(v) {
      v <- v[!is.na(v)]
      if (length(v) < 1L) return(NA_real_)
      mean(v)
    },
    median = function(v) {
      v <- v[!is.na(v)]
      if (length(v) < 1L) return(NA_real_)
      stats::median(v)
    },
    sd = function(v) {
      v <- v[!is.na(v)]
      if (length(v) < 2L) return(NA_real_)
      stats::sd(v)
    },
    iqr = function(v) {
      v <- v[!is.na(v)]
      if (length(v) < 1L) return(NA_real_)
      stats::IQR(v)
    },
    mad = function(v) {
      v <- v[!is.na(v)]
      if (length(v) < 1L) return(NA_real_)
      stats::mad(v)
    }
  )

  groups <- as.factor(data[[by]])
  out <- data.frame(levels(groups), stringsAsFactors = FALSE)
  names(out) <- by
  for (col in value_cols) {
    pieces <- split(data[[col]], groups)
    out[[col]] <- vapply(pieces, summarise_one, numeric(1), USE.NAMES = FALSE)
  }
  out
}

#' Count observations per group
#'
#' @param group A factor or vector of group labels.
#' @param na_label Label used for missing group values, or `NULL` to drop them.
#' @export
group_counts <- function(group, na_label = NULL) {
  if (!is.null(na_label)) {
    group <- as.character(group)
    group[is.na(group)] <- na_label
  }
  counts <- table(group, useNA = "no")
  stats::setNames(as.integer(counts), names(counts))
}
