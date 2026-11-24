#' Per-replicate outcome metrics
#'
#' Each metric maps a list of trajectories to one value per trajectory;
#' trajectories with missing values in `column` give `NA`.
#'
#' @param runs A list of trajectories, each a data frame with a `time`
#'   column, as returned by [toy.epidemic()].
#' @param column Column the metric is computed from.
#' @return A numeric vector with one value per element of `runs`.
#' @rdname outcome-metrics
#' @export
metric.peak.size <- function(runs, column = "I") {
  if (!is.list(runs) || length(runs) == 0L) {
    stop("`runs` must be a non-empty list of trajectories.", call. = FALSE)
  }
  values <- vapply(runs, function(run) {
    if (!column %in% names(run)) {
      stop(sprintf("Column `%s` not found in a run.", column), call. = FALSE)
    }
    if (anyNA(run[[column]])) {
      return(NA_real_)
    }
    max(run[[column]])
  }, numeric(1))
  names(values) <- names(runs)
  values
}

#' @rdname outcome-metrics
#' @export
metric.peak.time <- function(runs, column = "I") {
  if (!is.list(runs) || length(runs) == 0L) {
    stop("`runs` must be a non-empty list of trajectories.", call. = FALSE)
  }
  values <- vapply(runs, function(run) {
    if (!column %in% names(run)) {
      stop(sprintf("Column `%s` not found in a run.", column), call. = FALSE)
    }
    if (anyNA(run[[column]])) {
      return(NA_real_)
    }
    run$time[[which.max(run[[column]])]]
  }, numeric(1))
  names(values) <- names(runs)
  values
}

#' @rdname outcome-metrics
#' @export
metric.attack.rate <- function(runs, column = "incidence") {
  if (!is.list(runs) || length(runs) == 0L) {
    stop("`runs` must be a non-empty list of trajectories.", call. = FALSE)
  }
  values <- vapply(runs, function(run) {
    if (!column %in% names(run)) {
      stop(sprintf("Column `%s` not found in a run.", column), call. = FALSE)
    }
    if (anyNA(run[[column]])) {
      return(NA_real_)
    }
    sum(run[[column]]) / (run$S[[1L]] + run$I[[1L]] + run$R[[1L]])
  }, numeric(1))
  names(values) <- names(runs)
  values
}

#' @rdname outcome-metrics
#' @export
metric.duration <- function(runs, column = "I") {
  if (!is.list(runs) || length(runs) == 0L) {
    stop("`runs` must be a non-empty list of trajectories.", call. = FALSE)
  }
  values <- vapply(runs, function(run) {
    if (!column %in% names(run)) {
      stop(sprintf("Column `%s` not found in a run.", column), call. = FALSE)
    }
    if (anyNA(run[[column]])) {
      return(NA_real_)
    }
    max(run$time[run[[column]] > 0], 0)
  }, numeric(1))
  names(values) <- names(runs)
  values
}

#' Summarise replicate values of one metric
#'
#' @param values Numeric vector of per-replicate values.
#' @param stats Statistics to compute, any of `"mean"`, `"sd"`, `"median"`,
#'   `"lower"` and `"upper"` (the central `level` interval).
#' @param level Coverage of the interval given by `lower` and `upper`.
#' @return A named numeric vector with one element per requested statistic.
#' @export
summarise.metric <- function(values, stats = c("mean", "sd", "median", "lower", "upper"),
                             level = 0.9) {
  stats <- match.arg(stats, several.ok = TRUE)
  values <- values[!is.na(values)]
  tail.prob <- (1 - level) / 2
  out <- numeric(0)

  # Statistic `mean`
  if ("mean" %in% stats) {
    out[["mean"]] <- mean(values)
  }
  # Statistic `sd`
  if ("sd" %in% stats) {
    out[["sd"]] <- if (length(values) > 1L) stats::sd(values) else NA_real_
  }
  # Statistic `median`
  if ("median" %in% stats) {
    out[["median"]] <- stats::median(values)
  }
  # Statistic `lower`
  if ("lower" %in% stats) {
    out[["lower"]] <- unname(stats::quantile(values, tail.prob))
  }
  # Statistic `upper`
  if ("upper" %in% stats) {
    out[["upper"]] <- unname(stats::quantile(values, 1 - tail.prob))
  }
  out
}

#' Summarise all outcome metrics of a replicate set
#'
#' @param runs A list of trajectories, as returned by [toy.epidemic()].
#' @param metrics Metrics to summarise, any of `"peak.size"`, `"peak.time"`,
#'   `"attack.rate"` and `"duration"`.
#' @param level Coverage of the interval given by `lower` and `upper`.
#' @return A data frame with column `metric` followed by the columns of
#'   [summarise.metric()], one row per requested metric.
#' @export
summarise.replicates <- function(runs, metrics = c("peak.size", "peak.time",
                                                   "attack.rate", "duration"),
                                 level = 0.9) {
  metrics <- match.arg(metrics, several.ok = TRUE)
  rows <- list()

  # Metric `peak.size`
  if ("peak.size" %in% metrics) {
    values <- metric.peak.size(runs)
    rows[["peak.size"]] <- summarise.metric(values, level = level)
  }
  # Metric `peak.time`
  if ("peak.time" %in% metrics) {
    values <- metric.peak.time(runs)
    rows[["peak.time"]] <- summarise.metric(values, level = level)
  }
  # Metric `attack.rate`
  if ("attack.rate" %in% metrics) {
    values <- metric.attack.rate(runs)
    rows[["attack.rate"]] <- summarise.metric(values, level = level)
  }
  # Metric `duration`
  if ("duration" %in% metrics) {
    values <- metric.duration(runs)
    rows[["duration"]] <- summarise.metric(values, level = level)
  }

  out <- data.frame(metric = names(rows), do.call(rbind, rows))
  rownames(out) <- NULL
  out
}

#' Compare scenarios on all outcome metrics
#'
#' @param results A named list of replicate lists, one per scenario.
#' @param metrics Metrics to summarise, see [summarise.replicates()].
#' @param level Coverage of the interval given by `lower` and `upper`.
#' @return A data frame with column `scenario` followed by the columns of
#'   [summarise.replicates()].
#' @export
compare.scenarios <- function(results, metrics = c("peak.size", "peak.time",
                                                   "attack.rate", "duration"),
                              level = 0.9) {
  if (is.null(names(results)) || any(names(results) == "")) {
    stop("`results` must be a named list with one element per scenario.", call. = FALSE)
  }
  tables <- lapply(names(results), function(name) {
    table <- summarise.replicates(results[[name]], metrics, level)
    data.frame(scenario = name, table)
  })
  out <- do.call(rbind, tables)
  rownames(out) <- NULL
  out
}
