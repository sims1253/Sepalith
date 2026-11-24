#' Anomalies relative to a baseline
#'
#' @param data A data frame of observations, one row per reading.
#' @param value Name of the measurement column.
#' @param site Name of the site identifier column.
#' @param baseline What to subtract: `"site.mean"` or `"site.median"`
#'   (per site), `"overall.mean"` (all sites pooled) or `"reference"`
#'   (a named vector of per-site values in `reference`).
#' @param reference Named numeric vector of baselines by site, used when
#'   `baseline = "reference"`.
#' @export
#' @return `data` with an added column `anomaly` and an attribute
#'   `baseline` naming the baseline used.
site.anomaly <- function(data, value, site = "site",
                         baseline = c("site.mean", "site.median", "overall.mean", "reference"),
                         reference = NULL) {
  baseline <- match.arg(baseline)
  x <- data[[value]]
  g <- as.character(data[[site]])
  base <- switch(baseline,
    site.mean = {
      centre <- tapply(x, g, mean, na.rm = TRUE)
      as.numeric(centre[g])
    },
    site.median = {
      centre <- tapply(x, g, stats::median, na.rm = TRUE)
      as.numeric(centre[g])
    },
    overall.mean = {
      centre <- mean(x, na.rm = TRUE)
      rep(centre, length(x))
    },
    reference = {
      if (is.null(reference) || !all(unique(g) %in% names(reference))) {
        stop("`reference` must have a value for every site.", call. = FALSE)
      }
      as.numeric(reference[g])
    }
  )
  data$anomaly <- x - base
  attr(data, "baseline") <- baseline
  data
}

#' Compare two sites over their common times
#'
#' @inheritParams site.anomaly
#' @param time Name of the time column.
#' @param a,b Identifiers of the two sites to compare.
#' @export
#' @return A list with elements `n`, `mean_a`, `mean_b`, `bias` (mean of
#'   `a - b`), `rmse` and `r`.
site.pair.stats <- function(data, value, a, b, site = "site", time = "time") {
  # Series of site `a`
  sa <- data[data[[site]] == a, c(time, value), drop = FALSE]
  if (nrow(sa) == 0L) {
    stop(sprintf("Site `%s` has no readings.", a), call. = FALSE)
  }
  names(sa) <- c("time", "a")

  # Series of site `b`
  sb <- data[data[[site]] == b, c(time, value), drop = FALSE]
  if (nrow(sb) == 0L) {
    stop(sprintf("Site `%s` has no readings.", b), call. = FALSE)
  }
  names(sb) <- c("time", "b")

  both <- merge(sa, sb, by = "time")
  both <- both[stats::complete.cases(both), , drop = FALSE]
  diff <- both$a - both$b
  list(
    n = nrow(both),
    mean_a = mean(both$a),
    mean_b = mean(both$b),
    bias = mean(diff),
    rmse = sqrt(mean(diff^2)),
    r = if (nrow(both) > 2L) stats::cor(both$a, both$b) else NA_real_
  )
}

#' Linear trend per site
#'
#' @inheritParams site.pair.stats
#' @param per Time unit of the slope: `"day"`, `"month"` (30 days),
#'   `"year"` (365.25 days) or `"decade"`.
#' @param min.n Minimum number of readings for a trend to be fitted.
#' @export
#' @return A data frame with columns `site`, `n`, `intercept`, `slope`,
#'   `p.value` and `r.squared`.
site.trend <- function(data, value, site = "site", time = "time",
                       per = c("year", "day", "month", "decade"), min.n = 3L) {
  per <- match.arg(per)
  days <- c(day = 1, month = 30, year = 365.25, decade = 10 * 365.25)
  sites <- sort(unique(data[[site]]))
  out <- data.frame(site = sites, n = 0L, intercept = NA_real_, slope = NA_real_,
                    p.value = NA_real_, r.squared = NA_real_)
  for (i in seq_along(sites)) {
    sub <- data[data[[site]] == sites[i], , drop = FALSE]
    t <- as.numeric(sub[[time]]) / (86400 * days[[per]])
    y <- sub[[value]]
    ok <- is.finite(t) & is.finite(y)
    out$n[i] <- sum(ok)
    if (sum(ok) < min.n) {
      next
    }

    # Fit and store the coefficients of `y ~ t`
    fit <- summary(stats::lm(y[ok] ~ t[ok]))
    out$intercept[i] <- fit$coefficients[1, "Estimate"]
    out$slope[i] <- fit$coefficients[2, "Estimate"]
    out$p.value[i] <- fit$coefficients[2, "Pr(>|t|)"]
    out$r.squared[i] <- fit$r.squared
  }
  out
}

#' Network value at each time
#'
#' Combines the readings of all sites at each time into one value.
#'
#' @inheritParams site.pair.stats
#' @param method `"mean"`, `"median"`, `"trimmed"` (mean without the
#'   lowest and highest 10 percent) or `"weighted"` (mean weighted by
#'   `weights`).
#' @param weights Named numeric vector of weights by site, used when
#'   `method = "weighted"`.
#' @export
#' @return A data frame with columns `time`, `n_sites` and `value`.
site.network <- function(data, value, site = "site", time = "time",
                         method = c("mean", "median", "trimmed", "weighted"),
                         weights = NULL) {
  method <- match.arg(method)
  if (method == "weighted" && is.null(weights)) {
    stop("`weights` must be supplied when `method = \"weighted\"`.", call. = FALSE)
  }
  ok <- !is.na(data[[value]])
  x <- data[[value]][ok]
  g <- data[[site]][ok]
  key <- data[[time]][ok]
  times <- sort(unique(key))
  out <- data.frame(time = times, n_sites = 0L, value = NA_real_)
  for (i in seq_along(times)) {
    at <- key == times[i]
    out$n_sites[i] <- sum(at)
    out$value[i] <- switch(method,
      mean = mean(x[at]),
      median = stats::median(x[at]),
      trimmed = mean(x[at], trim = 0.1),
      weighted = stats::weighted.mean(x[at], weights[as.character(g[at])])
    )
  }
  out
}

#' Rank sites by a summary of their readings
#'
#' @inheritParams site.anomaly
#' @param by Statistic to rank on: `"mean"`, `"median"`, `"max"` or
#'   `"p95"` (95th percentile).
#' @param decreasing Rank the largest value first.
#' @export
#' @return A data frame with columns `rank`, `site` and the statistic
#'   named by `by`.
site.rank <- function(data, value, site = "site", by = c("mean", "median", "max", "p95"),
                      decreasing = TRUE) {
  by <- match.arg(by)
  fun <- switch(by,
    mean = function(v) mean(v, na.rm = TRUE),
    median = function(v) stats::median(v, na.rm = TRUE),
    max = function(v) max(v, na.rm = TRUE),
    p95 = function(v) unname(stats::quantile(v, 0.95, na.rm = TRUE))
  )
  stat <- tapply(data[[value]], data[[site]], fun)
  ord <- order(stat, decreasing = decreasing)
  out <- data.frame(rank = seq_along(ord), site = names(stat)[ord])
  out[[by]] <- as.numeric(stat[ord])
  out
}
