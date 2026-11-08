#' Internal-consistency reliability
#'
#' @description Computes one reliability coefficient for a set of items.
#' @param items A numeric matrix or data frame of item scores, one column per
#'   item.
#' @param method One of `"alpha"`, `"standardized"`, `"splitHalf"`,
#'   `"lambda2"` or `"kr20"`.
#' @return A single number with attribute `method`.
#' @export
reliabilityCoef = function(items, method = c("alpha", "standardized", "splitHalf",
                                             "lambda2", "kr20")) {
  method = match.arg(method)
  items = as.matrix(items)
  items = items[stats::complete.cases(items), , drop = FALSE]
  k = ncol(items)
  if (k < 2) {
    stop("`items` must have at least two columns.", call. = FALSE)
  }
  covMat = stats::cov(items)
  totalVar = sum(covMat)

  value = switch(method,
    alpha = {
      (k / (k - 1)) * (1 - sum(diag(covMat)) / totalVar)
    },
    standardized = {
      corMat = stats::cor(items)
      meanR = mean(corMat[lower.tri(corMat)])
      k * meanR / (1 + (k - 1) * meanR)
    },
    splitHalf = {
      odd = rowSums(items[, seq(1, k, by = 2), drop = FALSE])
      even = rowSums(items[, seq(2, k, by = 2), drop = FALSE])
      r = stats::cor(odd, even)
      2 * r / (1 + r)
    },
    lambda2 = {
      offDiag = covMat
      diag(offDiag) = 0
      lambda1 = 1 - sum(diag(covMat)) / totalVar
      lambda1 + sqrt(k / (k - 1) * sum(offDiag^2)) / totalVar
    },
    kr20 = {
      if (!all(items %in% c(0, 1))) {
        stop("`kr20` needs items scored 0/1.", call. = FALSE)
      }
      p = colMeans(items)
      (k / (k - 1)) * (1 - sum(p * (1 - p)) / stats::var(rowSums(items)))
    }
  )
  attr(value, "method") = method
  value
}

#' Confidence interval for coefficient alpha
#'
#' @description Interval estimate for alpha by one of several methods.
#' @param items A numeric matrix or data frame of item scores, one column per
#'   item.
#' @param level The confidence level.
#' @param method One of `"feldt"`, `"fisher"` or `"bootstrap"`.
#' @param nBoot Number of bootstrap resamples for `method = "bootstrap"`.
#' @return A named numeric vector with elements `alpha`, `lower` and `upper`.
#' @export
alphaInterval = function(items, level = 0.95, method = c("feldt", "fisher", "bootstrap"),
                         nBoot = 500) {
  method = match.arg(method)
  items = as.matrix(items)
  items = items[stats::complete.cases(items), , drop = FALSE]
  n = nrow(items)
  k = ncol(items)
  alpha = c(reliabilityCoef(items, "alpha"))
  tail = (1 - level) / 2

  switch(method,
    feldt = {
      # Feldt: (1 - alpha) / (1 - true alpha) follows an F distribution
      df1 = n - 1
      df2 = (n - 1) * (k - 1)
      lower = 1 - (1 - alpha) * stats::qf(1 - tail, df1, df2)
      upper = 1 - (1 - alpha) * stats::qf(tail, df1, df2)
    },
    fisher = {
      # Fisher: normal interval on atanh(alpha), transformed back
      se = 1 / sqrt(n - 3)
      lower = tanh(atanh(alpha) - stats::qnorm(1 - tail) * se)
      upper = tanh(atanh(alpha) + stats::qnorm(1 - tail) * se)
    },
    bootstrap = {
      # Bootstrap: percentile interval over resampled respondents
      draws = vapply(seq_len(nBoot), function(b) {
        c(reliabilityCoef(items[sample.int(n, n, replace = TRUE), , drop = FALSE], "alpha"))
      }, numeric(1))
      lower = unname(stats::quantile(draws, tail))
      upper = unname(stats::quantile(draws, 1 - tail))
    }
  )
  c(alpha = alpha, lower = lower, upper = upper)
}

#' Correct a correlation for attenuation
#'
#' @description Disattenuates an observed correlation by the reliabilities.
#' @param rxy The observed correlation between two scores.
#' @param rxx,ryy The reliabilities of the two scores.
#' @param correct One of `"both"`, `"x"`, `"y"` or `"none"`.
#' @return The corrected correlation, capped at 1 in absolute value.
#' @export
correctAttenuation = function(rxy, rxx = 1, ryy = 1, correct = c("both", "x", "y", "none")) {
  correct = match.arg(correct)
  if (any(c(rxx, ryy) <= 0 | c(rxx, ryy) > 1)) {
    stop("Reliabilities must lie in (0, 1].", call. = FALSE)
  }

  corrected = switch(correct,
    both = {
      rxy / sqrt(rxx * ryy)
    },
    x = {
      rxy / sqrt(rxx)
    },
    y = {
      rxy / sqrt(ryy)
    },
    none = {
      rxy
    }
  )
  if (any(abs(corrected) > 1)) {
    warning("Corrected correlation exceeds 1 and was capped.", call. = FALSE)
  }
  pmax(pmin(corrected, 1), -1)
}

#' Test-retest agreement
#'
#' @description Stability statistics for scores taken on two occasions.
#' @param time1,time2 Numeric vectors of scores from the first and second
#'   occasion.
#' @return A list with elements `n`, `pearson`, `spearman`, `icc`,
#'   `meanDiff`, `sdDiff` and `limits`.
#' @export
testRetest = function(time1, time2) {
  if (length(time1) != length(time2)) {
    stop("`time1` and `time2` must have the same length.", call. = FALSE)
  }
  keep = !is.na(time1) & !is.na(time2)
  x = time1[keep]
  y = time2[keep]
  n = length(x)
  if (n < 3) {
    stop("At least three complete pairs are needed.", call. = FALSE)
  }

  # Correlations
  pearson = stats::cor(x, y)
  spearman = stats::cor(x, y, method = "spearman")

  # Absolute-agreement ICC from a two-way layout
  scores = cbind(x, y)
  grand = mean(scores)
  msRows = 2 * sum((rowMeans(scores) - grand)^2) / (n - 1)
  msCols = n * sum((colMeans(scores) - grand)^2)
  sse = sum((scores - outer(rowMeans(scores), colMeans(scores), "+") + grand)^2)
  msError = sse / (n - 1)
  icc = (msRows - msError) / (msRows + msError + 2 * (msCols - msError) / n)

  # Differences and limits of agreement
  diffs = y - x
  meanDiff = mean(diffs)
  sdDiff = stats::sd(diffs)
  limits = meanDiff + c(lower = -1.96, upper = 1.96) * sdDiff

  list(n = n, pearson = pearson, spearman = spearman, icc = icc,
    meanDiff = meanDiff, sdDiff = sdDiff, limits = limits)
}

#' Reliability table for several scales
#'
#' @description One row of reliability statistics per scale.
#' @param items A numeric matrix or data frame of item scores, one column per
#'   item.
#' @param scales A named list of character vectors naming the items of each
#'   scale.
#' @return A data frame with columns `scale`, `nItems`, `meanR`, `alpha`,
#'   `sd` and `sem`.
#' @export
reliabilityTable = function(items, scales) {
  if (is.null(names(scales)) || any(names(scales) == "")) {
    stop("`scales` must be a named list.", call. = FALSE)
  }
  stats = c("nItems", "meanR", "alpha", "sd", "sem")
  out = matrix(NA_real_, length(scales), length(stats), dimnames = list(names(scales), stats))

  for (s in names(scales)) {
    cols = scales[[s]]
    if (!all(cols %in% colnames(items))) {
      stop(sprintf("Scale `%s` names unknown items.", s), call. = FALSE)
    }
    block = as.matrix(items[, cols, drop = FALSE])
    block = block[stats::complete.cases(block), , drop = FALSE]
    out[s, "nItems"] = length(cols)
    if (length(cols) < 2) {
      next
    }
    corMat = stats::cor(block)
    out[s, "meanR"] = mean(corMat[lower.tri(corMat)])
    out[s, "alpha"] = reliabilityCoef(block, "alpha")
    out[s, "sd"] = stats::sd(rowSums(block))
    out[s, "sem"] = out[s, "sd"] * sqrt(1 - out[s, "alpha"])
  }

  data.frame(scale = names(scales), out, row.names = NULL)
}

#' Split a scale into two halves
#'
#' @description Assigns each item to half `a` or `b` by a splitting rule.
#' @param items A numeric matrix or data frame of item scores, one column per
#'   item.
#' @param split One of `"oddEven"`, `"firstSecond"`, `"balanced"` or
#'   `"random"`.
#' @return A list with elements `a`, `b` (column indices), `r` and
#'   `spearmanBrown`.
#' @export
splitHalves = function(items, split = c("oddEven", "firstSecond", "balanced", "random")) {
  split = match.arg(split)
  items = as.matrix(items)
  k = ncol(items)
  if (k < 2) {
    stop("`items` must have at least two columns.", call. = FALSE)
  }

  a = switch(split,
    oddEven = {
      seq(1, k, by = 2)
    },
    firstSecond = {
      seq_len(ceiling(k / 2))
    },
    balanced = {
      # Alternate down the items ordered by mean, ABBA style
      ord = order(colMeans(items, na.rm = TRUE))
      ord[rep(c(TRUE, FALSE, FALSE, TRUE), length.out = k)]
    },
    random = {
      sort(sample.int(k, ceiling(k / 2)))
    }
  )
  b = setdiff(seq_len(k), a)

  halfA = rowSums(items[, a, drop = FALSE])
  halfB = rowSums(items[, b, drop = FALSE])
  r = stats::cor(halfA, halfB, use = "complete.obs")
  list(a = sort(a), b = b, r = r, spearmanBrown = 2 * r / (1 + r))
}
