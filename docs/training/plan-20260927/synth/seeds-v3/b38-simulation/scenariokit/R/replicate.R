#' Derive per-replicate seeds
#'
#' @param n Number of replicates.
#' @param seed Master seed.
#' @return An integer vector of `n` distinct seeds.
#' @keywords internal
seed.streams <- function(n, seed = 1L) {
  set.seed(seed)
  sample.int(.Machine$integer.max, n)
}

#' Run replicates of a model
#'
#' @param model A function `(params, days)` returning a trajectory.
#' @param params A list of model parameters.
#' @param n.reps Number of replicates.
#' @param seed Master seed; replicate `i` uses the `i`-th seed of
#'   [seed.streams()].
#' @param days Number of days to simulate.
#' @return A list of `n.reps` trajectories with attribute `seeds`.
#' @export
replicate.runs <- function(model, params, n.reps = 10L, seed = 1L, days = 100L) {
  if (!is.function(model)) {
    stop("`model` must be a function.", call. = FALSE)
  }
  if (!is.numeric(n.reps) || length(n.reps) != 1L || n.reps < 1) {
    stop(sprintf("`%s` must be a single positive number.", "n.reps"), call. = FALSE)
  }
  if (!is.numeric(seed) || length(seed) != 1L || seed < 1) {
    stop(sprintf("`%s` must be a single positive number.", "seed"), call. = FALSE)
  }
  if (!is.numeric(days) || length(days) != 1L || days < 1) {
    stop(sprintf("`%s` must be a single positive number.", "days"), call. = FALSE)
  }

  seeds <- seed.streams(n.reps, seed)
  runs <- vector("list", n.reps)
  for (i in seq_len(n.reps)) {
    set.seed(seeds[[i]])
    runs[[i]] <- model(params, days)
  }
  attr(runs, "seeds") <- seeds
  runs
}

#' Run replicates on several cores
#'
#' Uses the same per-replicate seeds as [replicate.runs()], so both
#' functions return identical results.
#'
#' @inheritParams replicate.runs
#' @param cores Number of worker processes for [parallel::mclapply()].
#' @return A list of `n.reps` trajectories with attribute `seeds`.
#' @export
run.parallel <- function(model, params, n.reps = 10L, seed = 1L, days = 100L,
                         cores = 1L) {
  if (!is.function(model)) {
    stop("`model` must be a function.", call. = FALSE)
  }
  if (!is.numeric(n.reps) || length(n.reps) != 1L || n.reps < 1) {
    stop(sprintf("`%s` must be a single positive number.", "n.reps"), call. = FALSE)
  }
  if (!is.numeric(seed) || length(seed) != 1L || seed < 1) {
    stop(sprintf("`%s` must be a single positive number.", "seed"), call. = FALSE)
  }
  if (!is.numeric(days) || length(days) != 1L || days < 1) {
    stop(sprintf("`%s` must be a single positive number.", "days"), call. = FALSE)
  }

  seeds <- seed.streams(n.reps, seed)
  runs <- parallel::mclapply(seq_len(n.reps), function(i) {
    set.seed(seeds[[i]])
    model(params, days)
  }, mc.cores = cores)
  attr(runs, "seeds") <- seeds
  runs
}

#' Run replicates with the future framework
#'
#' @inheritParams replicate.runs
#' @return A list of `n.reps` trajectories with attribute `seeds`.
#' @importFrom future.apply future_lapply
#' @export
run.future <- function(model, params, n.reps = 10L, seed = 1L, days = 100L) {
  if (!is.function(model)) {
    stop("`model` must be a function.", call. = FALSE)
  }
  seeds <- seed.streams(n.reps, seed)
  runs <- future_lapply(seq_len(n.reps), function(i) {
    set.seed(seeds[[i]])
    model(params, days)
  }, future.seed = NULL)
  attr(runs, "seeds") <- seeds
  runs
}

#' Bootstrap confidence interval for a replicate statistic
#'
#' @param x Numeric vector of replicate values.
#' @param stat Statistic: `"mean"`, `"median"` or `"sd"`.
#' @param method Interval type: `"percentile"`, `"basic"` or `"normal"`.
#' @param n.boot Number of bootstrap resamples.
#' @param level Confidence level.
#' @return A named numeric vector with elements `estimate`, `lower` and
#'   `upper`.
#' @export
bootstrap.ci <- function(x, stat = c("mean", "median", "sd"),
                         method = c("percentile", "basic", "normal"),
                         n.boot = 999L, level = 0.95) {
  stat <- match.arg(stat)
  method <- match.arg(method)
  x <- x[!is.na(x)]
  alpha <- (1 - level) / 2

  # 1. Pick the statistic
  fn <- switch(stat,
    mean = mean,
    median = stats::median,
    sd = stats::sd
  )
  # 2. Resample
  estimate <- fn(x)
  boot <- vapply(seq_len(n.boot), function(b) fn(sample(x, replace = TRUE)), numeric(1))
  # 3. Build the interval
  bounds <- switch(method,
    percentile = {
      stats::quantile(boot, c(alpha, 1 - alpha), names = FALSE)
    },
    basic = {
      q <- stats::quantile(boot, c(1 - alpha, alpha), names = FALSE)
      2 * estimate - q
    },
    normal = {
      z <- stats::qnorm(1 - alpha)
      estimate + c(-1, 1) * z * stats::sd(boot)
    }
  )
  c(estimate = estimate, lower = bounds[[1L]], upper = bounds[[2L]])
}

#' Running statistics over replicates
#'
#' Shows how the summary of a metric settles as replicates are added, to
#' judge whether enough replicates were run.
#'
#' @param values Numeric vector of per-replicate values, in run order.
#' @param stats Statistics to track, any of `"mean"`, `"sd"`, `"lower"` and
#'   `"upper"` (normal-theory interval of the mean).
#' @param level Confidence level of the interval.
#' @return A data frame with column `n` followed by one column per element
#'   of `stats`.
#' @export
running.stats <- function(values, stats = c("mean", "sd", "lower", "upper"), level = 0.95) {
  stats <- match.arg(stats, several.ok = TRUE)
  n <- seq_along(values)
  run.mean <- cumsum(values) / n
  run.sd <- c(NA_real_, vapply(n[-1L], function(k) stats::sd(values[seq_len(k)]), numeric(1)))
  half <- stats::qnorm(1 - (1 - level) / 2) * run.sd / sqrt(n)
  out <- data.frame(n = n)

  # Track `mean`
  if ("mean" %in% stats) {
    out[["mean"]] <- run.mean
  }
  # Track `sd`
  if ("sd" %in% stats) {
    out[["sd"]] <- run.sd
  }
  # Track `lower`
  if ("lower" %in% stats) {
    out[["lower"]] <- run.mean - half
  }
  # Track `upper`
  if ("upper" %in% stats) {
    out[["upper"]] <- run.mean + half
  }
  out
}

#' Monte Carlo standard error
#'
#' @param x Numeric vector of replicate values.
#' @param method `"iid"` treats the values as independent; `"batch"` uses
#'   the spread of `n.batches` batch means.
#' @param n.batches Number of batches for the `"batch"` method.
#' @return A single number.
#' @export
mcse <- function(x, method = c("iid", "batch"), n.batches = 10L) {
  method <- match.arg(method)
  x <- x[!is.na(x)]
  if (method == "iid") {
    se <- stats::sd(x) / sqrt(length(x))
  } else {
    size <- length(x) %/% n.batches
    if (size < 1L) {
      stop("`x` has fewer values than `n.batches`.", call. = FALSE)
    }
    batch <- rep(seq_len(n.batches), each = size)
    means <- tapply(x[seq_along(batch)], batch, mean)
    se <- stats::sd(means) / sqrt(n.batches)
  }
  se
}
