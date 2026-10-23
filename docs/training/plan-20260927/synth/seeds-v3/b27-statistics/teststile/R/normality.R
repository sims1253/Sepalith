#' Sample skewness
#'
#' @param x A numeric vector.
#' @param type Estimator: `"g1"` (moment), `"G1"` (adjusted Fisher-Pearson)
#'   or `"b1"` (based on the sample standard deviation).
#' @export
sampleSkewness = function(x, type = c("g1", "G1", "b1")) {
  type = match.arg(type)
  x = x[!is.na(x)]
  n = length(x)
  dev = x - mean(x)
  g1 = mean(dev^3) / mean(dev^2)^1.5

  switch(type,
    g1 = g1,
    G1 = g1 * sqrt(n * (n - 1)) / (n - 2),
    b1 = g1 * ((n - 1) / n)^1.5
  )
}

#' Sample kurtosis
#'
#' @param x A numeric vector.
#' @param type Estimator: `"g2"` (moment), `"G2"` (adjusted) or `"b2"`
#'   (based on the sample standard deviation); all return excess kurtosis.
#' @export
sampleKurtosis = function(x, type = c("g2", "G2", "b2")) {
  type = match.arg(type)
  x = x[!is.na(x)]
  n = length(x)
  dev = x - mean(x)
  g2 = mean(dev^4) / mean(dev^2)^2 - 3

  switch(type,
    g2 = g2,
    G2 = ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3)),
    b2 = (g2 + 3) * ((n - 1) / n)^2 - 3
  )
}

#' Jarque-Bera normality statistic
#'
#' @param x A numeric vector.
#' @return A list with elements `statistic`, `parameter`, `pValue` and
#'   `method`.
#' @export
jarqueBera = function(x) {
  x = x[!is.na(x)]
  n = length(x)
  s = sampleSkewness(x, "g1")
  k = sampleKurtosis(x, "g2")
  statistic = n / 6 * (s^2 + k^2 / 4)

  list(
    statistic = statistic,
    parameter = 2,
    pValue = stats::pchisq(statistic, df = 2, lower.tail = FALSE),
    method = "jb"
  )
}

#' Anderson-Darling normality statistic
#'
#' @description
#' Uses estimated mean and standard deviation, the small-sample correction
#' `A * (1 + 0.75 / n + 2.25 / n^2)` and the piecewise p-value
#' approximation of D'Agostino and Stephens.
#'
#' @param x A numeric vector with at least eight non-missing values.
#' @return A list with elements `statistic`, `parameter`, `pValue` and
#'   `method`; `parameter` is the corrected statistic.
#' @export
andersonDarling = function(x) {
  x = sort(x[!is.na(x)])
  n = length(x)
  if (n < 8L) {
    stop("`x` must have at least 8 non-missing values.", call. = FALSE)
  }
  z = stats::pnorm((x - mean(x)) / stats::sd(x))
  i = seq_len(n)
  statistic = -n - mean((2 * i - 1) * (log(z) + log(1 - rev(z))))
  aa = statistic * (1 + 0.75 / n + 2.25 / n^2)

  # Piecewise p-value approximation on the corrected statistic
  pValue = if (aa < 0.2) {
    1 - exp(-13.436 + 101.14 * aa - 223.73 * aa^2)
  } else if (aa < 0.34) {
    1 - exp(-8.318 + 42.796 * aa - 59.938 * aa^2)
  } else if (aa < 0.6) {
    exp(0.9177 - 4.279 * aa - 1.38 * aa^2)
  } else {
    exp(1.2937 - 5.709 * aa + 0.0186 * aa^2)
  }
  list(statistic = statistic, parameter = aa, pValue = pValue, method = "ad")
}

#' D'Agostino skewness, kurtosis and omnibus tests
#'
#' @param x A numeric vector with at least 20 non-missing values.
#' @return A list with elements `skewZ`, `skewP`, `kurtZ`, `kurtP`,
#'   `omnibus` and `pValue`, where `omnibus` is `skewZ^2 + kurtZ^2`.
#' @export
dagostinoTests = function(x) {
  x = x[!is.na(x)]
  n = length(x)
  if (n < 20L) {
    stop("`x` must have at least 20 non-missing values.", call. = FALSE)
  }

  # Skewness test
  y = sampleSkewness(x, "g1") * sqrt((n + 1) * (n + 3) / (6 * (n - 2)))
  beta2 = 3 * (n^2 + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
  w2 = -1 + sqrt(2 * (beta2 - 1))
  delta = 1 / sqrt(log(sqrt(w2)))
  alpha = sqrt(2 / (w2 - 1))
  skewZ = delta * log(y / alpha + sqrt((y / alpha)^2 + 1))
  skewP = 2 * stats::pnorm(-abs(skewZ))

  # Kurtosis test
  b2 = sampleKurtosis(x, "g2") + 3
  e = 3 * (n - 1) / (n + 1)
  v = 24 * n * (n - 2) * (n - 3) / ((n + 1)^2 * (n + 3) * (n + 5))
  sb1 = 6 * (n^2 - 5 * n + 2) / ((n + 7) * (n + 9)) *
    sqrt(6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
  a = 6 + 8 / sb1 * (2 / sb1 + sqrt(1 + 4 / sb1^2))
  r = (1 - 2 / a) / (1 + (b2 - e) / sqrt(v) * sqrt(2 / (a - 4)))
  kurtZ = (1 - 2 / (9 * a) - sign(r) * abs(r)^(1 / 3)) / sqrt(2 / (9 * a))
  kurtP = 2 * stats::pnorm(-abs(kurtZ))

  omnibus = skewZ^2 + kurtZ^2
  list(
    skewZ = skewZ,
    skewP = skewP,
    kurtZ = kurtZ,
    kurtP = kurtP,
    omnibus = omnibus,
    pValue = stats::pchisq(omnibus, df = 2, lower.tail = FALSE)
  )
}

#' Normality tests for several variables
#'
#' Runs the Jarque-Bera (`jb`), Shapiro-Wilk (`sw`) and Anderson-Darling
#' (`ad`) tests on every variable.
#'
#' @param data A data frame.
#' @param vars Character vector of numeric column names.
#' @return A data frame with columns `variable`, `n`, `jbStat`, `jbP`,
#'   `swStat`, `swP`, `adStat` and `adP`.
#' @export
normalityTable = function(data, vars = names(data)) {
  absent = setdiff(vars, names(data))
  if (length(absent) > 0L) {
    stop("Unknown columns: ", paste(absent, collapse = ", "), call. = FALSE)
  }

  k = length(vars)
  n = integer(k)
  jbStat = jbP = swStat = swP = adStat = adP = rep(NA_real_, k)
  for (i in seq_len(k)) {
    x = data[[vars[i]]]
    x = x[!is.na(x)]
    n[i] = length(x)
    if (n[i] < 8L) {
      next
    }
    # Jarque-Bera
    jb = jarqueBera(x)
    jbStat[i] = jb$statistic
    jbP[i] = jb$pValue
    # Shapiro-Wilk
    sw = stats::shapiro.test(x)
    swStat[i] = unname(sw$statistic)
    swP[i] = sw$p.value
    # Anderson-Darling
    ad = andersonDarling(x)
    adStat[i] = ad$statistic
    adP[i] = ad$pValue
  }

  data.frame(
    variable = vars, n = n,
    jbStat = jbStat, jbP = jbP,
    swStat = swStat, swP = swP,
    adStat = adStat, adP = adP,
    stringsAsFactors = FALSE
  )
}

#' Flag outlying values
#'
#' @param x A numeric vector.
#' @param method One of `"z"` (standard score), `"modified"` (median and
#'   MAD based score), `"iqr"` (Tukey fences) or `"percentile"` (tails).
#' @param threshold Cut-off on the score; for `"iqr"` the fence multiplier
#'   and for `"percentile"` the probability in each tail.
#' @export
outlierFlags = function(x, method = c("z", "modified", "iqr", "percentile"), threshold = NULL) {
  method = match.arg(method)
  ok = x[!is.na(x)]

  switch(method,
    z = {
      if (is.null(threshold)) threshold = 3
      abs(x - mean(ok)) / stats::sd(ok) > threshold
    },
    modified = {
      if (is.null(threshold)) threshold = 3.5
      abs(0.6745 * (x - stats::median(ok)) / stats::mad(ok, constant = 1)) > threshold
    },
    iqr = {
      if (is.null(threshold)) threshold = 1.5
      q = stats::quantile(ok, c(0.25, 0.75), names = FALSE)
      x < q[1] - threshold * diff(q) | x > q[2] + threshold * diff(q)
    },
    percentile = {
      if (is.null(threshold)) threshold = 0.01
      q = stats::quantile(ok, c(threshold, 1 - threshold), names = FALSE)
      x < q[1] | x > q[2]
    }
  )
}
