#' @title Create a rubric definition
#' @description A rubric is a set of criteria, each scored on an ordered scale
#'   of levels and carrying a weight in the total.
#' @param criteria Character vector of criterion names.
#' @param levels Integer number of performance levels per criterion, or a
#'   vector with one value per criterion.
#' @param weights Numeric vector of criterion weights; defaults to equal.
#' @return A list of class `rubric`.
#' @export
make.rubric <- function(criteria, levels = 4L, weights = NULL) {
  k <- length(criteria)
  if (anyDuplicated(criteria)) {
    stop("Criterion names must be unique.", call. = FALSE)
  }
  levels <- rep_len(as.integer(levels), k)
  if (anyNA(levels) || any(levels < 2L)) {
    stop("Every criterion needs at least two levels.", call. = FALSE)
  }

  if (is.null(weights)) {
    weights <- rep(1, k)
  }
  if (length(weights) != k) {
    stop("`weights` must have one value per criterion.", call. = FALSE)
  }
  # Weights are normalised below, so they only need to be usable as shares.
  if (anyNA(weights) || any(weights < 0) || sum(weights) == 0) {
    stop("`weights` must be non-negative and not all zero.", call. = FALSE)
  }

  structure(
    list(
      criteria = criteria,
      levels = stats::setNames(levels, criteria),
      weights = stats::setNames(weights / sum(weights), criteria)
    ),
    class = "rubric"
  )
}

#' @title Check rubric scores against a rubric
#' @description Ensures the score columns match the criteria and that every
#'   score is a whole number within the criterion's range.
#' @param scores A data frame of rubric scores, one row per submission and
#'   one column per criterion.
#' @param rubric A `rubric` object from [make.rubric()].
#' @return `scores`, invisibly, if all checks pass.
#' @keywords internal
check.rubric.scores <- function(scores, rubric) {
  absent <- setdiff(rubric$criteria, names(scores))
  if (length(absent) > 0) {
    stop("Missing criterion columns: ", paste(absent, collapse = ", "), call. = FALSE)
  }

  for (crit in rubric$criteria) {
    x <- scores[[crit]]
    # Unscored criteria (NA) are allowed; they simply propagate.
    bad <- !is.na(x) & (x < 1 | x > rubric$levels[[crit]] | x != round(x))
    if (any(bad)) {
      stop("Criterion `", crit, "` has out-of-range scores in rows ",
        paste(utils::head(which(bad), 5), collapse = ", "),
        call. = FALSE
      )
    }
  }
  invisible(scores)
}

#' @title Convert rubric scores to a percentage
#' @description Rescales each criterion to 0-1 from its lowest to its highest
#'   level and combines them with the rubric weights.
#' @param scores A data frame of rubric scores, one row per submission and
#'   one column per criterion.
#' @param rubric A `rubric` object from [make.rubric()].
#' @return A numeric vector of percentages, one per row of `scores`.
#' @export
rubric.to.percent <- function(scores, rubric) {
  check.rubric.scores(scores, rubric)

  total <- numeric(nrow(scores))
  for (crit in rubric$criteria) {
    scaled <- (scores[[crit]] - 1) / (rubric$levels[[crit]] - 1)
    total <- total + rubric$weights[[crit]] * scaled
  }
  100 * total
}

#' @title Convert rubric scores to a performance band
#' @description Uses the weighted mean level, rather than a percentage, and
#'   cuts it into named bands.
#' @param scores A data frame of rubric scores, one row per submission and
#'   one column per criterion.
#' @param rubric A `rubric` object from [make.rubric()].
#' @param bands Character vector of band labels from lowest to highest.
#' @return A factor of bands, one per row of `scores`.
#' @export
rubric.to.band <- function(scores, rubric,
                           bands = c("beginning", "developing", "proficient", "exemplary")) {
  check.rubric.scores(scores, rubric)

  total <- numeric(nrow(scores))
  for (crit in rubric$criteria) {
    scaled <- (scores[[crit]] - 1) / (rubric$levels[[crit]] - 1)
    total <- total + rubric$weights[[crit]] * scaled
  }

  # Equal-width bands on the 0-1 scale; the top band includes 1 itself.
  breaks <- seq(0, 1, length.out = length(bands) + 1)
  cut(total, breaks = breaks, labels = bands, include.lowest = TRUE, ordered_result = TRUE)
}

#' @title Agreement between two raters on one criterion
#' @description Computes exact agreement, adjacent agreement (within one
#'   level), or Cohen's kappa for two vectors of ratings.
#' @param a,b Integer vectors of ratings from two raters for the same
#'   submissions.
#' @param type Agreement statistic to compute.
#' @return A single number.
#' @export
rater.agreement <- function(a, b, type = c("exact", "adjacent", "kappa")) {
  type <- match.arg(type)
  if (length(a) != length(b)) {
    stop("`a` and `b` must rate the same submissions.", call. = FALSE)
  }

  # Only submissions scored by both raters count.
  keep <- !is.na(a) & !is.na(b)
  a <- a[keep]
  b <- b[keep]
  if (length(a) == 0) {
    return(NA_real_)
  }

  switch(type,
    exact = mean(a == b),
    adjacent = mean(abs(a - b) <= 1),
    kappa = {
      lv <- sort(unique(c(a, b)))
      tab <- table(factor(a, levels = lv), factor(b, levels = lv))
      p.obs <- sum(diag(tab)) / sum(tab)
      p.exp <- sum(rowSums(tab) * colSums(tab)) / sum(tab)^2
      # Both raters used a single level: agreement is perfect by definition.
      if (p.exp == 1) 1 else (p.obs - p.exp) / (1 - p.exp)
    }
  )
}

#' @title Agreement table for a double-marked sample
#' @description Runs [rater.agreement()] for every criterion of a rubric and
#'   returns one row per criterion.
#' @param first,second Data frames of rubric scores from two raters, rows in
#'   the same submission order.
#' @param rubric A `rubric` object from [make.rubric()].
#' @return A data frame with columns `criterion`, `exact`, `adjacent` and
#'   `kappa`.
#' @export
agreement.table <- function(first, second, rubric) {
  if (nrow(first) != nrow(second)) {
    stop("Both raters must score the same submissions.", call. = FALSE)
  }

  k <- length(rubric$criteria)
  exact <- numeric(k)
  adjacent <- numeric(k)
  kappa <- numeric(k)
  for (i in seq_len(k)) {
    crit <- rubric$criteria[i]
    exact[i] <- rater.agreement(first[[crit]], second[[crit]], "exact")
    adjacent[i] <- rater.agreement(first[[crit]], second[[crit]], "adjacent")
    kappa[i] <- rater.agreement(first[[crit]], second[[crit]], "kappa")
  }

  data.frame(
    criterion = rubric$criteria,
    exact = exact,
    adjacent = adjacent,
    kappa = kappa,
    stringsAsFactors = FALSE
  )
}

#' @title Print a rubric
#' @description Shows each criterion with its number of levels and its share
#'   of the total.
#' @param x A `rubric` object from [make.rubric()].
#' @param ... Unused, for compatibility with [print()].
#' @return `x`, invisibly.
#' @export
print.rubric <- function(x, ...) {
  cat("<rubric: ", length(x$criteria), " criteria>\n", sep = "")
  for (crit in x$criteria) {
    cat(sprintf("  %-20s %d levels  %5.1f%%\n", crit, x$levels[[crit]],
      100 * x$weights[[crit]]))
  }
  invisible(x)
}
