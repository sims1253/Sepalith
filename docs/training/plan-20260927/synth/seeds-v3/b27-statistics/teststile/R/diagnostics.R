#' Conventional cut-offs for influence measures
#'
#' @param n Number of observations.
#' @param p Number of model coefficients.
#' @keywords internal
#' @noRd
diagnosticCutoffs = function(n, p) {
  c(
    leverage = 2 * p / n,
    rstandard = 2,
    rstudent = 2,
    cooks = 4 / n,
    dffits = 2 * sqrt(p / n)
  )
}

#' Influence measures of a linear model
#'
#' @param fit A fitted `lm` object.
#' @param measures Character vector naming the measures to compute, any of
#'   `"leverage"`, `"rstandard"`, `"rstudent"`, `"cooks"` and `"dffits"`.
#' @return A data frame with one column per entry of `measures`, one row
#'   per observation.
#' @export
influenceMeasures = function(fit, measures = c("leverage", "rstandard", "rstudent",
                                               "cooks", "dffits")) {
  measures = match.arg(measures, several.ok = TRUE)
  if (!inherits(fit, "lm")) {
    stop("`fit` must be an `lm` object.", call. = FALSE)
  }
  h = stats::hatvalues(fit)
  e = stats::residuals(fit)
  n = length(e)
  p = length(stats::coef(fit))
  if (n - p < 2) {
    stop("`fit` needs at least two residual degrees of freedom.", call. = FALSE)
  }

  # Residual scale, and the leave-one-out scale for every observation
  s = sqrt(sum(e^2) / (n - p))
  sLoo = sqrt(((n - p) * s^2 - e^2 / (1 - h)) / (n - p - 1))

  out = list()
  if ("leverage" %in% measures) {
    out$leverage = unname(h)
  }
  if ("rstandard" %in% measures) {
    out$rstandard = unname(e / (s * sqrt(1 - h)))
  }
  if ("rstudent" %in% measures) {
    out$rstudent = unname(e / (sLoo * sqrt(1 - h)))
  }
  if ("cooks" %in% measures) {
    r = e / (s * sqrt(1 - h))
    out$cooks = unname(r^2 * h / (p * (1 - h)))
  }
  if ("dffits" %in% measures) {
    r = e / (sLoo * sqrt(1 - h))
    out$dffits = unname(r * sqrt(h / (1 - h)))
  }
  as.data.frame(out)[measures]
}

#' Flag influential observations
#'
#' @param fit A fitted `lm` object.
#' @param measures Character vector of measures accepted by
#'   [influenceMeasures()].
#' @return A logical matrix with one row per observation and one column per
#'   measure, plus an attribute `"cutoffs"`.
#' @export
flagInfluential = function(fit, measures = c("leverage", "rstudent", "cooks", "dffits")) {
  values = influenceMeasures(fit, measures)
  cutoffs = diagnosticCutoffs(nrow(values), length(stats::coef(fit)))[measures]
  flags = matrix(FALSE, nrow = nrow(values), ncol = length(measures),
                 dimnames = list(NULL, measures))

  for (m in measures) {
    flags[, m] = abs(values[[m]]) > cutoffs[[m]]
  }
  attr(flags, "cutoffs") = cutoffs
  flags
}

#' Residuals of a linear model on several scales
#'
#' @param fit A fitted `lm` object.
#' @param type One of `"raw"`, `"scaled"` (divided by the residual standard
#'   error), `"standardized"` (also divided by `sqrt(1 - h)`) or
#'   `"studentized"` (leave-one-out scale).
#' @keywords internal
residualsByType = function(fit, type = c("raw", "scaled", "standardized", "studentized")) {
  type = match.arg(type)
  e = unname(stats::residuals(fit))
  s = stats::sigma(fit)
  switch(type,
    raw = e,
    scaled = e / s,
    standardized = influenceMeasures(fit, "rstandard")$rstandard,
    studentized = influenceMeasures(fit, "rstudent")$rstudent
  )
}

#' Variance inflation factors
#'
#' @param fit A fitted `lm` object with an intercept and at least two
#'   predictors.
#' @export
vifValues = function(fit) {
  X = stats::model.matrix(fit)
  X = X[, colnames(X) != "(Intercept)", drop = FALSE]
  if (ncol(X) < 2L) {
    stop("`fit` must have at least two predictors.", call. = FALSE)
  }

  # Regress every predictor on the others
  out = numeric(ncol(X))
  names(out) = colnames(X)
  for (j in seq_len(ncol(X))) {
    aux = stats::lm.fit(cbind(1, X[, -j, drop = FALSE]), X[, j])
    r2 = 1 - sum(aux$residuals^2) / sum((X[, j] - mean(X[, j]))^2)
    out[j] = 1 / (1 - r2)
  }
  out
}

#' Breusch-Pagan statistic for heteroscedasticity
#'
#' @description
#' The studentized (Koenker) version: `n` times the R-squared of the
#' squared residuals regressed on the model matrix.
#'
#' @param fit A fitted `lm` object.
#' @return A list with elements `statistic`, `parameter`, `pValue` and
#'   `method`.
#' @export
breuschPagan = function(fit) {
  X = stats::model.matrix(fit)
  e2 = stats::residuals(fit)^2

  # Auxiliary regression of the squared residuals on the model matrix
  aux = stats::lm.fit(X, e2)
  r2 = 1 - sum(aux$residuals^2) / sum((e2 - mean(e2))^2)
  statistic = length(e2) * r2
  df = ncol(X) - 1

  list(
    statistic = statistic,
    parameter = df,
    pValue = stats::pchisq(statistic, df, lower.tail = FALSE),
    method = "breuschPagan"
  )
}

#' Durbin-Watson statistic
#'
#' @param fit A fitted `lm` object, with observations in time order.
#' @export
durbinWatson = function(fit) {
  e = stats::residuals(fit)
  if (length(e) < 3L) {
    stop("`fit` must have at least 3 observations.", call. = FALSE)
  }
  sum(diff(e)^2) / sum(e^2)
}

#' Diagnostic report for a linear model
#'
#' @description
#' Runs one check per assumption and reports, for each, the statistic, the
#' threshold it is compared with and whether the check raises a flag:
#' 1. `leverage`: the largest hat value against `2 * p / n`;
#' 2. `outliers`: the largest absolute studentized residual against 2;
#' 3. `influence`: the largest Cook's distance against `4 / n`;
#' 4. `heteroscedasticity`: the Breusch-Pagan p-value against `alpha`;
#' 5. `autocorrelation`: the distance of the Durbin-Watson statistic from 2
#'    against 1;
#' 6. `collinearity`: the largest variance inflation factor against 10.
#'
#' @param fit A fitted `lm` object.
#' @param alpha Significance level for the Breusch-Pagan check.
#' @return A data frame with columns `check`, `value`, `threshold` and
#'   `flagged`, one row per check in the order above.
#' @export
diagnosticReport = function(fit, alpha = 0.05) {
  n = length(stats::residuals(fit))
  p = length(stats::coef(fit))
  cutoffs = diagnosticCutoffs(n, p)
  im = influenceMeasures(fit, c("leverage", "rstudent", "cooks"))
  checks = c("leverage", "outliers", "influence", "heteroscedasticity",
             "autocorrelation", "collinearity")
  value = threshold = rep(NA_real_, length(checks))
  names(value) = names(threshold) = checks

  # 1. Leverage
  value[["leverage"]] = max(im$leverage)
  threshold[["leverage"]] = cutoffs[["leverage"]]
  # 2. Outliers
  value[["outliers"]] = max(abs(im$rstudent))
  threshold[["outliers"]] = cutoffs[["rstudent"]]
  # 3. Influence
  value[["influence"]] = max(im$cooks)
  threshold[["influence"]] = cutoffs[["cooks"]]
  # 4. Heteroscedasticity
  value[["heteroscedasticity"]] = breuschPagan(fit)$pValue
  threshold[["heteroscedasticity"]] = alpha
  # 5. Autocorrelation
  value[["autocorrelation"]] = abs(durbinWatson(fit) - 2)
  threshold[["autocorrelation"]] = 1
  # 6. Collinearity
  if (sum(colnames(stats::model.matrix(fit)) != "(Intercept)") >= 2L) {
    value[["collinearity"]] = max(vifValues(fit))
  }
  threshold[["collinearity"]] = 10

  flagged = value > threshold
  flagged[["heteroscedasticity"]] = value[["heteroscedasticity"]] < alpha
  data.frame(check = checks, value = unname(value), threshold = unname(threshold),
             flagged = unname(flagged), stringsAsFactors = FALSE)
}
