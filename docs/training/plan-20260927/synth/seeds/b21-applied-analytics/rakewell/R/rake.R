#' Validate a set of raking margins
#'
#' Checks that every margin is a named numeric vector of non-negative targets
#' and that each margin refers to a column present in the sample.
#'
#' @param margins A named list of target margins, one numeric vector per
#'   raking variable, named by category.
#' @param data A data frame of survey respondents, one row per respondent.
#' @return `margins`, invisibly, if all checks pass.
#' @keywords internal
check_margins <- function(margins, data) {
  if (!is.list(margins) || is.null(names(margins)) || any(names(margins) == "")) {
    stop("`margins` must be a fully named list.", call. = FALSE)
  }
  missing_vars <- setdiff(names(margins), names(data))
  if (length(missing_vars) > 0) {
    stop("Margin variables not found in `data`: ",
      paste(missing_vars, collapse = ", "),
      call. = FALSE
    )
  }
  for (var in names(margins)) {
    target <- margins[[var]]
    if (!is.numeric(target) || is.null(names(target))) {
      stop("Margin `", var, "` must be a named numeric vector.", call. = FALSE)
    }
    if (any(target < 0, na.rm = TRUE) || anyNA(target)) {
      stop("Margin `", var, "` contains negative or missing targets.", call. = FALSE)
    }
    observed <- unique(as.character(data[[var]]))
    unmatched <- setdiff(observed, names(target))
    if (length(unmatched) > 0) {
      stop("Categories of `", var, "` without a target: ",
        paste(unmatched, collapse = ", "),
        call. = FALSE
      )
    }
  }
  return(invisible(margins))
}

#' Convert margin targets to proportions
#'
#' Rescales each margin so that its targets sum to one. Margins given as
#' counts and margins given as shares are thereby treated alike.
#'
#' @param margins A named list of target margins, one numeric vector per
#'   raking variable, named by category.
#' @return A list shaped like `margins` whose elements each sum to one.
#' @keywords internal
margins_as_shares <- function(margins) {
  out <- vector("list", length(margins))
  names(out) <- names(margins)
  for (var in names(margins)) {
    target <- margins[[var]]
    out[[var]] <- target / sum(target)
  }
  return(out)
}

#' Adjust weights to match one margin
#'
#' Performs a single post-stratification step for one raking variable: weights
#' within each category are multiplied so that the weighted share hits the
#' target.
#'
#' @param weights A numeric vector of current respondent weights.
#' @param x A vector of category labels for one raking variable.
#' @param target A named numeric vector of target shares for `x`.
#' @return A numeric vector of adjusted weights, the same length as `weights`.
#' @keywords internal
rake_one_margin <- function(weights, x, target) {
  x <- as.character(x)
  current <- tapply(weights, x, sum)
  current <- current / sum(current)
  factors <- target[names(current)] / current
  factors[!is.finite(factors)] <- 0
  return(weights * unname(factors[x]))
}

#' Measure the distance between weighted and target margins
#'
#' Returns the largest absolute difference between a weighted share and its
#' target, across every category of every margin.
#'
#' @param weights A numeric vector of current respondent weights.
#' @param data A data frame of survey respondents, one row per respondent.
#' @param shares A named list of target shares as returned by
#'   [margins_as_shares()].
#' @return A single non-negative number.
#' @keywords internal
margin_gap <- function(weights, data, shares) {
  gaps <- numeric(length(shares))
  for (i in seq_along(shares)) {
    var <- names(shares)[i]
    current <- tapply(weights, as.character(data[[var]]), sum)
    current <- current / sum(current)
    target <- shares[[var]][names(current)]
    gaps[i] <- max(abs(current - target))
  }
  return(max(gaps))
}

#' Rake survey weights to known margins
#'
#' Iterative proportional fitting: cycles through the margins, adjusting the
#' weights to each in turn, until the largest gap falls below `tol` or
#' `max_iter` sweeps have been made.
#'
#' @param data A data frame of survey respondents, one row per respondent.
#' @param margins A named list of target margins, one numeric vector per
#'   raking variable, named by category.
#' @param base_weights Optional numeric vector of design weights; defaults to
#'   ones.
#' @param total Population total the final weights should sum to. If `NULL`,
#'   the sum of `base_weights` is kept.
#' @param tol Convergence tolerance on the largest margin gap.
#' @param max_iter Maximum number of full sweeps over the margins.
#' @return A numeric vector of raked weights with attributes `iterations`,
#'   `converged` and `gap`.
#' @export
#' @examples
#' df <- data.frame(sex = c("f", "m", "m"), age = c("young", "old", "young"))
#' rake_weights(df, list(sex = c(f = 0.5, m = 0.5), age = c(young = 0.4, old = 0.6)))
rake_weights <- function(data, margins, base_weights = NULL, total = NULL,
                         tol = 1e-6, max_iter = 100L) {
  check_margins(margins, data)
  n <- nrow(data)
  if (is.null(base_weights)) {
    base_weights <- rep(1, n)
  }
  if (length(base_weights) != n) {
    stop("`base_weights` must have one value per row of `data`.", call. = FALSE)
  }
  if (is.null(total)) {
    total <- sum(base_weights)
  }
  shares <- margins_as_shares(margins)
  weights <- base_weights
  converged <- FALSE
  gap <- Inf
  iter <- 0L
  while (iter < max_iter) {
    iter <- iter + 1L
    for (var in names(shares)) {
      weights <- rake_one_margin(weights, data[[var]], shares[[var]])
    }
    gap <- margin_gap(weights, data, shares)
    if (gap < tol) {
      converged <- TRUE
      break
    }
  }
  if (!converged) {
    warning("Raking did not converge after ", max_iter, " iterations (gap = ",
      signif(gap, 3), ").",
      call. = FALSE
    )
  }
  weights <- weights * total / sum(weights)
  attr(weights, "iterations") <- iter
  attr(weights, "converged") <- converged
  attr(weights, "gap") <- gap
  return(weights)
}

#' Tabulate weighted margins against their targets
#'
#' Builds a long data frame comparing the unweighted share, the weighted share
#' and the target share for every category of every raking variable.
#'
#' @param data A data frame of survey respondents, one row per respondent.
#' @param weights A numeric vector of current respondent weights.
#' @param margins A named list of target margins, one numeric vector per
#'   raking variable, named by category.
#' @return A data frame with columns `variable`, `category`, `unweighted`,
#'   `weighted`, `target` and `difference`.
#' @export
compare_margins <- function(data, weights, margins) {
  shares <- margins_as_shares(margins)
  pieces <- vector("list", length(shares))
  for (i in seq_along(shares)) {
    var <- names(shares)[i]
    x <- as.character(data[[var]])
    categories <- names(shares[[var]])
    unweighted <- as.numeric(table(factor(x, levels = categories))) / length(x)
    weighted <- tapply(weights, factor(x, levels = categories), sum)
    weighted[is.na(weighted)] <- 0
    weighted <- as.numeric(weighted) / sum(weights)
    pieces[[i]] <- data.frame(
      variable = var,
      category = categories,
      unweighted = unweighted,
      weighted = weighted,
      target = unname(shares[[var]]),
      stringsAsFactors = FALSE
    )
  }
  out <- do.call(rbind, pieces)
  out$difference <- out$weighted - out$target
  rownames(out) <- NULL
  return(out)
}
