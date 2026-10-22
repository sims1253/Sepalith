#' Complete-case item matrix
#'
#' Extracts the scale items as a numeric matrix and drops respondents with
#' any missing response, which is what classical reliability formulas assume.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items A character vector of item names making up the scale.
#' @return A numeric matrix with one column per item.
#' @noRd
item_matrix <- function(data, items) {
  x <- as.matrix(data[items])
  storage.mode(x) <- "double"
  x[stats::complete.cases(x), , drop = FALSE]
}

#' Cronbach's alpha computed by hand
#'
#' Uses the classical formula based on the ratio of summed item variances to
#' the variance of the total score.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items A character vector of item names making up the scale.
#' @return A single numeric value.
#' @family reliability
#' @export
cronbach_alpha <- function(data, items) {
  x <- item_matrix(data, items)
  k <- ncol(x)
  if (k < 2) {
    stop("Alpha needs at least two items.", call. = FALSE)
  }
  item_var <- apply(x, 2, stats::var)
  total_var <- stats::var(rowSums(x))
  (k / (k - 1)) * (1 - sum(item_var) / total_var)
}

#' Alpha if each item is deleted
#'
#' Recomputes Cronbach's alpha once per item with that item left out, which
#' highlights items that drag the internal consistency down.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items A character vector of item names making up the scale.
#' @return A named numeric vector, one value per item.
#' @family reliability
#' @export
alpha_if_deleted <- function(data, items) {
  if (length(items) < 3) {
    stop("Alpha-if-deleted needs at least three items.", call. = FALSE)
  }
  out <- numeric(length(items))
  names(out) <- items
  for (i in seq_along(items)) {
    out[i] <- cronbach_alpha(data, items[-i])
  }
  out
}

#' Item-total correlations
#'
#' Correlates each item with the total score. The corrected version removes
#' the item from the total before correlating, avoiding part-whole inflation.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items A character vector of item names making up the scale.
#' @param corrected If `TRUE`, correlate with the rest score instead of the total.
#' @return A named numeric vector, one value per item.
#' @family reliability
#' @export
item_total_cor <- function(data, items, corrected = TRUE) {
  x <- item_matrix(data, items)
  total <- rowSums(x)
  out <- numeric(length(items))
  names(out) <- items
  for (i in seq_along(items)) {
    criterion <- if (corrected) total - x[, i] else total
    out[i] <- stats::cor(x[, i], criterion)
  }
  out
}

#' Spearman-Brown prophecy
#'
#' Predicts the reliability of a test lengthened (or shortened) by `factor`.
#'
#' @param rel Reliability of the current test.
#' @param factor Ratio of new test length to current test length.
#' @return The predicted reliability.
#' @family reliability
#' @export
spearman_brown <- function(rel, factor) {
  stopifnot(all(rel > -1 & rel < 1), all(factor > 0))
  (factor * rel) / (1 + (factor - 1) * rel)
}

#' Inverse Spearman-Brown prophecy
#'
#' Finds the lengthening factor needed to move from one reliability to another.
#'
#' @param rel Reliability of the current test.
#' @param target Desired reliability of the new test.
#' @return The required ratio of new test length to current test length.
#' @family reliability
#' @export
spearman_brown_factor <- function(rel, target) {
  stopifnot(all(rel > -1 & rel < 1), all(target > -1 & target < 1))
  (target * (1 - rel)) / (rel * (1 - target))
}

#' Split-half reliability
#'
#' Splits the items into two halves, correlates the half scores and steps the
#' result up to full length with [spearman_brown()].
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items A character vector of item names making up the scale.
#' @param split How to form the halves: `"odd_even"`, `"first_last"` or `"random"`.
#' @param seed Random seed used when `split = "random"`.
#' @return A list with the half-test correlation `r_half` and stepped-up `rel`.
#' @family reliability
#' @export
split_half <- function(data, items, split = c("odd_even", "first_last", "random"),
                       seed = NULL) {
  split <- match.arg(split)
  k <- length(items)
  half_a <- switch(split,
    odd_even = items[seq(1, k, by = 2)],
    first_last = items[seq_len(k %/% 2)],
    random = {
      if (!is.null(seed)) set.seed(seed)
      sample(items, k %/% 2)
    }
  )
  half_b <- setdiff(items, half_a)
  x <- item_matrix(data, items)
  r_half <- stats::cor(rowSums(x[, half_a, drop = FALSE]), rowSums(x[, half_b, drop = FALSE]))
  list(r_half = r_half, rel = spearman_brown(r_half, factor = 2), split = split)
}

#' Reliability report for a scale
#'
#' Collects alpha, alpha-if-deleted and corrected item-total correlations into
#' a single object with a print method.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items A character vector of item names making up the scale.
#' @param name A label for the scale used when printing.
#' @return An object of class `itemwise_reliability`.
#' @export
reliability_report <- function(data, items, name = "scale") {
  x <- item_matrix(data, items)
  items_tab <- data.frame(
    item = items,
    mean = colMeans(x),
    sd = apply(x, 2, stats::sd),
    r_drop = item_total_cor(data, items, corrected = TRUE),
    alpha_drop = alpha_if_deleted(data, items),
    row.names = NULL
  )
  structure(
    list(
      name = name,
      n = nrow(x),
      alpha = cronbach_alpha(data, items),
      items = items_tab
    ),
    class = "itemwise_reliability"
  )
}

#' Print a reliability report
#'
#' Shows the overall alpha followed by a rounded item statistics table.
#'
#' @param x An object of class `itemwise_reliability`.
#' @param digits Number of decimal places to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.itemwise_reliability <- function(x, digits = 3, ...) {
  cat("Reliability of '", x$name, "' (n = ", x$n, ")\n", sep = "")
  cat("Cronbach's alpha: ", format(round(x$alpha, digits), nsmall = digits), "\n\n", sep = "")
  tab <- x$items
  num <- vapply(tab, is.numeric, logical(1))
  tab[num] <- lapply(tab[num], round, digits = digits)
  print(tab, row.names = FALSE)
  invisible(x)
}
