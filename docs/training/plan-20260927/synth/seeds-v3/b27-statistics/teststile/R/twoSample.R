#' Split a variable into two samples by a grouping vector
#'
#' @param x A numeric vector.
#' @param group A vector with exactly two distinct non-missing values.
#' @keywords internal
#' @return A list with elements `x` and `y`, the values of the first and
#'   second group without missing values.
#' @noRd
splitTwo = function(x, group) {
  group = as.factor(group)
  if (nlevels(droplevels(group)) != 2L) {
    stop("`group` must have exactly two levels.", call. = FALSE)
  }
  parts = split(x, droplevels(group))
  list(x = parts[[1]][!is.na(parts[[1]])], y = parts[[2]][!is.na(parts[[2]])])
}

#' Welch two-sample t statistic
#'
#' @param x,y Numeric vectors without missing values.
#' @family two-sample statistics
#' @return A list with elements `statistic`, `parameter`, `pValue` and
#'   `method`.
#' @keywords internal
#' @importFrom stats var pt
welchStat = function(x, y) {
  vx = stats::var(x) / length(x)
  vy = stats::var(y) / length(y)
  statistic = (mean(x) - mean(y)) / sqrt(vx + vy)
  df = (vx + vy)^2 / (vx^2 / (length(x) - 1) + vy^2 / (length(y) - 1))

  list(
    statistic = statistic,
    parameter = df,
    pValue = 2 * stats::pt(-abs(statistic), df),
    method = "welch"
  )
}

#' Pooled-variance two-sample t statistic
#'
#' @param x,y Numeric vectors without missing values.
#' @family two-sample statistics
#' @return A list with elements `statistic`, `parameter`, `pValue` and
#'   `method`.
#' @keywords internal
#' @importFrom stats var pt
pooledStat = function(x, y) {
  nx = length(x)
  ny = length(y)
  df = nx + ny - 2
  sp2 = ((nx - 1) * stats::var(x) + (ny - 1) * stats::var(y)) / df
  statistic = (mean(x) - mean(y)) / sqrt(sp2 * (1 / nx + 1 / ny))

  list(
    statistic = statistic,
    parameter = df,
    pValue = 2 * stats::pt(-abs(statistic), df),
    method = "pooled"
  )
}

#' Wilcoxon rank-sum statistic with a normal approximation
#'
#' @param x,y Numeric vectors without missing values.
#' @family two-sample statistics
#' @return A list with elements `statistic`, `parameter`, `pValue` and
#'   `method`; `parameter` is the continuity-corrected z score.
#' @keywords internal
#' @importFrom stats pnorm
rankSumStat = function(x, y) {
  nx = length(x)
  ny = length(y)
  r = rank(c(x, y))
  statistic = sum(r[seq_len(nx)]) - nx * (nx + 1) / 2
  ties = table(r)
  n = nx + ny
  sigma = sqrt(nx * ny / 12 * ((n + 1) - sum(ties^3 - ties) / (n * (n - 1))))
  delta = statistic - nx * ny / 2
  z = (delta - sign(delta) * 0.5) / sigma

  list(
    statistic = statistic,
    parameter = z,
    pValue = 2 * stats::pnorm(-abs(z)),
    method = "rankSum"
  )
}

#' Two-sample Kolmogorov-Smirnov statistic
#'
#' @param x,y Numeric vectors without missing values.
#' @family two-sample statistics
#' @return A list with elements `statistic`, `parameter`, `pValue` and
#'   `method`; `parameter` is the scaled statistic `sqrt(m * n / (m + n)) * D`.
#' @keywords internal
#' @importFrom stats ecdf
ksStat = function(x, y) {
  grid = sort(unique(c(x, y)))
  statistic = max(abs(stats::ecdf(x)(grid) - stats::ecdf(y)(grid)))
  lambda = sqrt(length(x) * length(y) / (length(x) + length(y))) * statistic
  # The alternating series converges badly near zero, where p is 1 anyway
  k = seq_len(100)
  p = if (lambda < 0.3) 1 else 2 * sum((-1)^(k - 1) * exp(-2 * k^2 * lambda^2))

  list(
    statistic = statistic,
    parameter = lambda,
    pValue = min(1, max(0, p)),
    method = "ks"
  )
}

#' Adjust p-values for multiple testing
#'
#' @param p A numeric vector of p-values.
#' @param method One of `"none"`, `"bonferroni"`, `"holm"`, `"hochberg"`,
#'   `"bh"` (Benjamini-Hochberg) or `"by"` (Benjamini-Yekutieli).
#' @export
adjustPValues = function(p, method = c("none", "bonferroni", "holm", "hochberg", "bh", "by")) {
  method = match.arg(method)
  m = sum(!is.na(p))
  out = p
  ok = !is.na(p)
  pp = p[ok]

  out[ok] = switch(method,
    none = {
      pp
    },
    bonferroni = {
      pmin(1, m * pp)
    },
    holm = {
      o = order(pp)
      adj = cummax((m - seq_len(m) + 1) * pp[o])
      pmin(1, adj)[order(o)]
    },
    hochberg = {
      o = order(pp, decreasing = TRUE)
      adj = cummin(seq_len(m) * pp[o])
      pmin(1, adj)[order(o)]
    },
    bh = {
      o = order(pp, decreasing = TRUE)
      adj = cummin(m / rev(seq_len(m)) * pp[o])
      pmin(1, adj)[order(o)]
    },
    by = {
      o = order(pp, decreasing = TRUE)
      q = sum(1 / seq_len(m))
      adj = cummin(q * m / rev(seq_len(m)) * pp[o])
      pmin(1, adj)[order(o)]
    }
  )
  out
}

#' Two-sample tests for several variables
#'
#' @param data A data frame.
#' @param vars Character vector of numeric column names to test.
#' @param group Name of the two-level grouping column.
#' @param test One of `"welch"`, `"pooled"`, `"rankSum"` or `"ks"`.
#' @param adjust Multiple-testing adjustment passed to [adjustPValues()].
#' @export
#' @return A data frame with columns `variable`, `nX`, `nY`, `statistic`,
#'   `parameter`, `pValue` and `pAdjusted`.
twoSampleTable = function(data, vars, group, test = c("welch", "pooled", "rankSum", "ks"),
                          adjust = "holm") {
  test = match.arg(test)
  absent = setdiff(c(vars, group), names(data))
  if (length(absent) > 0L) {
    stop("Unknown columns: ", paste(absent, collapse = ", "), call. = FALSE)
  }
  notNumeric = vars[!vapply(data[vars], is.numeric, logical(1))]
  if (length(notNumeric) > 0L) {
    stop("Non-numeric columns: ", paste(notNumeric, collapse = ", "), call. = FALSE)
  }

  statFun = switch(test,
    welch = welchStat,
    pooled = pooledStat,
    rankSum = rankSumStat,
    ks = ksStat
  )

  rows = vector("list", length(vars))
  for (i in seq_along(vars)) {
    s = splitTwo(data[[vars[i]]], data[[group]])
    res = statFun(s$x, s$y)
    rows[[i]] = data.frame(
      variable = vars[i],
      nX = length(s$x),
      nY = length(s$y),
      statistic = res$statistic,
      parameter = res$parameter,
      pValue = res$pValue,
      stringsAsFactors = FALSE
    )
  }

  out = do.call(rbind, rows)
  out$pAdjusted = adjustPValues(out$pValue, method = adjust)
  rownames(out) = NULL
  out
}
