#' @title Construct a permutation test result
#' @description Low-level constructor for objects of class `fw_perm`.
#' @param statistic Observed value of the test statistic.
#' @param null_dist Numeric vector of the statistic under permutation.
#' @param alternative Direction of the alternative hypothesis.
#' @param method Short description of the test.
#' @return An object of class `fw_perm`.
#' @keywords internal
new_fw_perm <- function(statistic, null_dist, alternative, method) {
  structure(
    list(
      statistic = statistic,
      null_dist = null_dist,
      p_value = perm_p_value(statistic, null_dist, alternative),
      alternative = alternative,
      method = method,
      n_perm = length(null_dist)
    ),
    class = "fw_perm"
  )
}

#' @title Monte Carlo p-value from a permutation distribution
#' @description Uses the `(b + 1) / (B + 1)` correction so the p-value is never zero.
#' @param observed Observed value of the test statistic.
#' @param null_dist Numeric vector of the statistic under permutation.
#' @param alternative Direction of the alternative hypothesis.
#' @return A single p-value.
#' @keywords internal
perm_p_value <- function(observed, null_dist, alternative = c("two.sided", "less", "greater")) {
  alternative <- match.arg(alternative)
  B <- length(null_dist)

  hits <- switch(alternative,
    two.sided = sum(abs(null_dist) >= abs(observed)),
    less = sum(null_dist <= observed),
    greater = sum(null_dist >= observed)
  )
  (hits + 1) / (B + 1)
}

#' @title Permutation test for a difference in means
#' @param x,y Numeric vectors of observations from the two groups.
#' @param n_perm Number of random permutations.
#' @param alternative Direction of the alternative hypothesis.
#' @return An object of class `fw_perm`.
#' @export
perm_test_mean <- function(x, y, n_perm = 9999L,
                           alternative = c("two.sided", "less", "greater")) {
  alternative <- match.arg(alternative)
  # Missing values carry no group information, so drop them up front.
  x <- x[!is.na(x)]
  y <- y[!is.na(y)]
  pooled <- c(x, y)
  nx <- length(x)
  observed <- mean(x) - mean(y)

  null_dist <- vapply(seq_len(n_perm), function(i) {
    idx <- sample.int(length(pooled), nx)
    mean(pooled[idx]) - mean(pooled[-idx])
  }, numeric(1))
  new_fw_perm(observed, null_dist, alternative, "Permutation test for difference in means")
}

#' @title Permutation test for a difference in medians
#' @param x,y Numeric vectors of observations from the two groups.
#' @param n_perm Number of random permutations.
#' @param alternative Direction of the alternative hypothesis.
#' @return An object of class `fw_perm`.
#' @export
perm_test_median <- function(x, y, n_perm = 9999L,
                             alternative = c("two.sided", "less", "greater")) {
  alternative <- match.arg(alternative)
  x <- x[!is.na(x)]
  y <- y[!is.na(y)]
  pooled <- c(x, y)
  nx <- length(x)
  observed <- stats::median(x) - stats::median(y)

  null_dist <- vapply(seq_len(n_perm), function(i) {
    idx <- sample.int(length(pooled), nx)
    stats::median(pooled[idx]) - stats::median(pooled[-idx])
  }, numeric(1))
  new_fw_perm(observed, null_dist, alternative, "Permutation test for difference in medians")
}

#' @title Permutation test for correlation
#' @description Shuffles `y` relative to `x` to build the null distribution of the
#'   correlation coefficient.
#' @param x,y Numeric vectors of paired observations.
#' @param n_perm Number of random permutations.
#' @param alternative Direction of the alternative hypothesis.
#' @param method Correlation coefficient passed to [stats::cor()].
#' @return An object of class `fw_perm`.
#' @export
perm_test_cor <- function(x, y, n_perm = 9999L,
                          alternative = c("two.sided", "less", "greater"),
                          method = c("pearson", "spearman", "kendall")) {
  alternative <- match.arg(alternative)
  method <- match.arg(method)
  ok <- stats::complete.cases(x, y)
  x <- x[ok]
  y <- y[ok]

  observed <- stats::cor(x, y, method = method)
  null_dist <- vapply(seq_len(n_perm), function(i) {
    stats::cor(x, sample(y), method = method)
  }, numeric(1))

  label <- paste("Permutation test for", method, "correlation")
  new_fw_perm(observed, null_dist, alternative, label)
}

#' @title Sign-flip permutation test for paired data
#' @description Randomly flips the signs of within-pair differences to test whether the
#'   mean difference is zero.
#' @param x,y Numeric vectors of paired observations.
#' @param n_perm Number of random permutations.
#' @param alternative Direction of the alternative hypothesis.
#' @return An object of class `fw_perm`.
#' @export
perm_test_paired <- function(x, y, n_perm = 9999L,
                             alternative = c("two.sided", "less", "greater")) {
  alternative <- match.arg(alternative)
  if (length(x) != length(y)) {
    stop("`x` and `y` must have the same length.", call. = FALSE)
  }
  d <- x - y
  d <- d[!is.na(d)]
  observed <- mean(d)

  # One row of random signs per permutation, applied in a single product.
  signs <- matrix(sample(c(-1, 1), length(d) * n_perm, replace = TRUE), nrow = n_perm)
  null_dist <- as.vector(signs %*% d) / length(d)
  new_fw_perm(observed, null_dist, alternative, "Sign-flip test for paired mean difference")
}

#' @title Permutation F test for a one-way layout
#' @param y Numeric response vector.
#' @param group Factor or vector of group labels, one per observation.
#' @param n_perm Number of random permutations.
#' @return An object of class `fw_perm`.
#' @export
perm_test_oneway <- function(y, group, n_perm = 9999L) {
  if (length(y) != length(group)) {
    stop("`y` and `group` must have the same length.", call. = FALSE)
  }
  ok <- !is.na(y) & !is.na(group)
  y <- y[ok]
  group <- droplevels(as.factor(group)[ok])

  f_stat <- function(g) {
    means <- tapply(y, g, mean)
    sizes <- tabulate(g, nbins = nlevels(g))
    between <- sum(sizes * (means - mean(y))^2) / (nlevels(g) - 1)
    within <- sum((y - means[g])^2) / (length(y) - nlevels(g))
    between / within
  }

  observed <- f_stat(group)
  null_dist <- vapply(seq_len(n_perm), function(i) f_stat(sample(group)), numeric(1))
  new_fw_perm(observed, null_dist, "greater", "Permutation one-way F test")
}

#' @title Freedman-Lane permutation test for one regression term
#' @description Permutes residuals of the reduced model (without `term`) and refits the
#'   full model to build the null distribution of the term's t statistic.
#' @param formula A model formula for the full model.
#' @param data A data frame containing the model variables.
#' @param term Name of the coefficient to test.
#' @param n_perm Number of random permutations.
#' @return An object of class `fw_perm`.
#' @export
perm_test_lm_term <- function(formula, data, term, n_perm = 999L) {
  full <- stats::lm(formula, data = data)
  if (!term %in% names(stats::coef(full))) {
    stop("`term` is not a coefficient of the full model.", call. = FALSE)
  }

  X <- stats::model.matrix(full)
  y <- stats::model.response(stats::model.frame(full))
  X_red <- X[, colnames(X) != term, drop = FALSE]

  # Both decompositions are reused for every permutation.
  qr_full <- qr(X)
  qr_red <- qr(X_red)
  j <- match(term, colnames(X))
  df_res <- nrow(X) - ncol(X)
  xtx_inv <- chol2inv(qr.R(qr_full))

  t_stat <- function(response) {
    beta <- qr.coef(qr_full, response)
    resid <- qr.resid(qr_full, response)
    se <- sqrt(sum(resid^2) / df_res * xtx_inv[j, j])
    beta[j] / se
  }
  observed <- t_stat(y)

  # Under the null the reduced-model residuals are exchangeable.
  fitted_red <- qr.fitted(qr_red, y)
  resid_red <- qr.resid(qr_red, y)
  null_dist <- vapply(seq_len(n_perm), function(i) {
    t_stat(fitted_red + sample(resid_red))
  }, numeric(1))

  label <- paste0("Freedman-Lane permutation test for `", term, "`")
  new_fw_perm(unname(observed), null_dist, "two.sided", label)
}
