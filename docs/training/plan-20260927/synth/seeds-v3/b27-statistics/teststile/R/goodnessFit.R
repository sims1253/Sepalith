#' Prediction accuracy measures
#'
#' @param observed A numeric vector of observed values.
#' @param predicted A numeric vector of predictions, the same length as
#'   `observed`.
#' @param measures Character vector naming the measures to compute, any of
#'   `"rmse"`, `"mae"`, `"mape"`, `"smape"`, `"bias"` and `"r2"`.
#' @export
#' @return A named numeric vector in the order of `measures`.
fitMeasures = function(observed, predicted,
                       measures = c("rmse", "mae", "mape", "smape", "bias", "r2")) {
  measures = match.arg(measures, several.ok = TRUE)
  if (!is.numeric(observed)) {
    stop("`observed` must be numeric.", call. = FALSE)
  }
  if (!is.numeric(predicted)) {
    stop("`predicted` must be numeric.", call. = FALSE)
  }
  if (length(observed) != length(predicted)) {
    stop("`observed` and `predicted` must have the same length.", call. = FALSE)
  }

  ok = !is.na(observed) & !is.na(predicted)
  err = predicted[ok] - observed[ok]
  obs = observed[ok]

  out = numeric(0)
  if ("rmse" %in% measures) {
    out["rmse"] = sqrt(mean(err^2))
  }
  if ("mae" %in% measures) {
    out["mae"] = mean(abs(err))
  }
  if ("mape" %in% measures) {
    out["mape"] = if (any(obs == 0)) NA_real_ else 100 * mean(abs(err / obs))
  }
  if ("smape" %in% measures) {
    denom = (abs(obs) + abs(predicted[ok])) / 2
    out["smape"] = if (any(denom == 0)) NA_real_ else 100 * mean(abs(err) / denom)
  }
  if ("bias" %in% measures) {
    out["bias"] = mean(err)
  }
  if ("r2" %in% measures) {
    out["r2"] = 1 - sum(err^2) / sum((obs - mean(obs))^2)
  }
  out[measures]
}

#' Information criteria of a fitted model
#'
#' @param fit A fitted model with a [logLik()] method.
#' @keywords internal
#' @return A list with elements `logLik`, `df`, `nobs`, `aic`, `bic` and
#'   `aicc`.
infoCriteria = function(fit) {
  ll = stats::logLik(fit)
  df = attr(ll, "df")
  n = attr(ll, "nobs")
  ll = as.numeric(ll)

  list(
    logLik = ll,
    df = df,
    nobs = n,
    aic = -2 * ll + 2 * df,
    bic = -2 * ll + log(n) * df,
    aicc = -2 * ll + 2 * df + 2 * df * (df + 1) / (n - df - 1)
  )
}

#' Compare fitted models by information criteria
#'
#' @param fits A named list of fitted models on the same data.
#' @param criterion Criterion used for ordering and weights: `"aic"`,
#'   `"bic"` or `"aicc"`.
#' @export
#' @return A data frame with columns `model`, `df`, `logLik`, `aic`, `bic`,
#'   `aicc`, `delta` and `weight`, ordered by `criterion`.
compareModels = function(fits, criterion = c("aic", "bic", "aicc")) {
  criterion = match.arg(criterion)
  if (is.null(names(fits)) || any(names(fits) == "")) {
    names(fits) = paste0("model", seq_along(fits))
  }
  rows = lapply(names(fits), function(nm) {
    ic = infoCriteria(fits[[nm]])
    data.frame(model = nm, df = ic$df, logLik = ic$logLik, aic = ic$aic,
               bic = ic$bic, aicc = ic$aicc, stringsAsFactors = FALSE)
  })

  out = do.call(rbind, rows)
  out = out[order(out[[criterion]]), , drop = FALSE]
  # Akaike-type weights from the criterion differences
  out$delta = out[[criterion]] - out[[criterion]][1]
  rel = exp(-out$delta / 2)
  out$weight = rel / sum(rel)
  rownames(out) = NULL
  out
}

#' K-fold cross-validation of a linear model
#'
#' @param formula A model formula.
#' @param data A data frame.
#' @param k Number of folds.
#' @param measures Character vector of measures accepted by [fitMeasures()].
#' @param seed Optional random seed for the fold assignment.
#' @export
#' @return A list with elements `folds` (a matrix with one row per fold and
#'   one column per measure), `mean` and `sd` (column summaries of
#'   `folds`), `predictions` (the out-of-fold prediction for every row),
#'   `fold` (the fold of every row) and `overall` (the measures computed on
#'   all out-of-fold predictions at once).
crossValidateLm = function(formula, data, k = 5L, measures = c("rmse", "mae", "r2"),
                           seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  n = nrow(data)
  if (k < 2L || k > n) {
    stop("`k` must be between 2 and the number of rows.", call. = FALSE)
  }
  response = all.vars(formula)[1]
  fold = sample(rep_len(seq_len(k), n))
  predictions = rep(NA_real_, n)
  folds = matrix(NA_real_, nrow = k, ncol = length(measures),
                 dimnames = list(paste0("fold", seq_len(k)), measures))

  for (f in seq_len(k)) {
    inFold = fold == f
    train = data[!inFold, , drop = FALSE]
    test = data[inFold, , drop = FALSE]
    fit = stats::lm(formula, data = train)
    pred = stats::predict(fit, newdata = test)
    predictions[inFold] = pred
    folds[f, ] = fitMeasures(test[[response]], pred, measures = measures)
  }

  list(
    folds = folds,
    mean = colMeans(folds),
    sd = apply(folds, 2, stats::sd),
    predictions = predictions,
    fold = fold,
    overall = fitMeasures(data[[response]], predictions, measures = measures)
  )
}

#' Chi-squared goodness-of-fit statistic
#'
#' @param observed A vector of observed counts.
#' @param p A vector of hypothesised probabilities, rescaled to sum to one.
#' @param minExpected Smallest expected count before a warning is raised.
#' @export
#' @return A list with elements `statistic`, `parameter`, `pValue` and
#'   `method`.
chisqGoodness = function(observed, p = rep(1, length(observed)), minExpected = 5) {
  if (length(p) != length(observed)) {
    stop("`p` must have the same length as `observed`.", call. = FALSE)
  }
  expected = sum(observed) * p / sum(p)
  if (any(expected < minExpected)) {
    warning("Some expected counts are below `minExpected`.", call. = FALSE)
  }

  statistic = sum((observed - expected)^2 / expected)
  df = length(observed) - 1

  list(
    statistic = statistic,
    parameter = df,
    pValue = stats::pchisq(statistic, df, lower.tail = FALSE),
    method = "chisqGoodness"
  )
}

#' Calibration table for predicted probabilities
#'
#' @description
#' Groups observations into bins of predicted probability and compares the
#' mean prediction with the observed event rate in each bin.
#'
#' @param observed A 0/1 or logical vector of outcomes.
#' @param prob A numeric vector of predicted probabilities.
#' @param bins Number of quantile bins.
#' @export
#' @return A data frame with columns `bin`, `n`, `meanPred`, `obsRate` and
#'   `contribution`, with attribute `"hosmerLemeshow"` holding the sum of
#'   the contributions.
calibrationTable = function(observed, prob, bins = 10L) {
  observed = as.numeric(observed)
  breaks = unique(stats::quantile(prob, seq(0, 1, length.out = bins + 1L), names = FALSE))
  bin = cut(prob, breaks = breaks, include.lowest = TRUE, labels = FALSE)
  n = tabulate(bin, nbins = length(breaks) - 1L)
  meanPred = as.numeric(tapply(prob, factor(bin, seq_along(n)), mean))
  obsRate = as.numeric(tapply(observed, factor(bin, seq_along(n)), mean))

  # Hosmer-Lemeshow contribution of every bin
  expected = n * meanPred
  contribution = (n * obsRate - expected)^2 / (expected * (1 - meanPred))
  out = data.frame(bin = seq_along(n), n = n, meanPred = meanPred,
                   obsRate = obsRate, contribution = contribution)
  out = out[n > 0, , drop = FALSE]
  attr(out, "hosmerLemeshow") = sum(out$contribution)
  out
}
