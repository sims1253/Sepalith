#' Convert raw scores to a norm scale
#'
#' @param raw A numeric vector of raw scale scores.
#' @param mean,sd Mean and standard deviation of the norm sample.
#' @param type One of `"z"`, `"t"`, `"iq"`, `"stanine"`, `"sten"` or
#'   `"percentile"`.
#' @return A numeric vector of converted scores with attribute `type`.
#' @export
convert_norm <- function(raw, mean, sd,
                         type = c("z", "t", "iq", "stanine", "sten", "percentile")) {
  type <- match.arg(type)
  if (!is.numeric(sd) || length(sd) != 1L || sd <= 0) {
    stop("`sd` must be a single positive number.", call. = FALSE)
  }
  z <- (raw - mean) / sd

  out <- switch(type,
    z = {
      z
    },
    t = {
      # T scores have mean 50 and sd 10
      round(50 + 10 * z)
    },
    iq = {
      # IQ scores have mean 100 and sd 15
      round(100 + 15 * z)
    },
    stanine = {
      # Stanines have mean 5 and sd 2, limited to 1-9
      pmin(pmax(round(5 + 2 * z), 1), 9)
    },
    sten = {
      # Stens have mean 5.5 and sd 2, limited to 1-10
      pmin(pmax(round(5.5 + 2 * z), 1), 10)
    },
    percentile = {
      round(100 * stats::pnorm(z), 1)
    }
  )
  attr(out, "type") <- type
  return(out)
}

#' Convert between standard score scales
#'
#' @param x A numeric vector of standard scores.
#' @param from,to Score scales, each one of `"z"`, `"t"`, `"iq"`, `"scaled"`
#'   or `"percentile"`.
#' @return A numeric vector of scores on the `to` scale.
#' @export
rescale_standard <- function(x, from = c("z", "t", "iq", "scaled", "percentile"),
                             to = c("z", "t", "iq", "scaled", "percentile")) {
  from <- match.arg(from)
  to <- match.arg(to)
  # Scale means and standard deviations
  centre <- c(z = 0, t = 50, iq = 100, scaled = 10)
  spread <- c(z = 1, t = 10, iq = 15, scaled = 3)

  # From the source scale to z
  z <- switch(from,
    z = x,
    t = (x - centre[["t"]]) / spread[["t"]],
    iq = (x - centre[["iq"]]) / spread[["iq"]],
    scaled = (x - centre[["scaled"]]) / spread[["scaled"]],
    percentile = {
      if (any(x <= 0 | x >= 100, na.rm = TRUE)) {
        stop("Percentiles must lie strictly between 0 and 100.", call. = FALSE)
      }
      stats::qnorm(x / 100)
    }
  )

  # From z to the target scale
  out <- switch(to,
    z = z,
    t = centre[["t"]] + spread[["t"]] * z,
    iq = centre[["iq"]] + spread[["iq"]] * z,
    scaled = centre[["scaled"]] + spread[["scaled"]] * z,
    percentile = 100 * stats::pnorm(z)
  )
  return(out)
}

#' Norm raw scores within age groups
#'
#' @param raw A numeric vector of raw scale scores.
#' @param group A character vector, one of `"young"`, `"middle"` or `"older"`
#'   per score.
#' @param young,middle,older Numeric vectors `c(mean = , sd = )` giving the
#'   norms of each age group.
#' @return A data frame with columns `raw`, `group`, `z` and `t`.
#' @export
norm_by_group <- function(raw, group, young = c(mean = 20, sd = 5),
                          middle = c(mean = 22, sd = 6), older = c(mean = 25, sd = 7)) {
  if (length(raw) != length(group)) {
    stop("`raw` and `group` must have the same length.", call. = FALSE)
  }
  z <- rep(NA_real_, length(raw))

  # Group `young`
  if (!all(c("mean", "sd") %in% names(young)) || young[["sd"]] <= 0) {
    stop(sprintf("`%s` must be c(mean = , sd = ) with sd > 0.", "young"), call. = FALSE)
  }
  in_group <- group %in% "young"
  z[in_group] <- (raw[in_group] - young[["mean"]]) / young[["sd"]]

  # Group `middle`
  if (!all(c("mean", "sd") %in% names(middle)) || middle[["sd"]] <= 0) {
    stop(sprintf("`%s` must be c(mean = , sd = ) with sd > 0.", "middle"), call. = FALSE)
  }
  in_group <- group %in% "middle"
  z[in_group] <- (raw[in_group] - middle[["mean"]]) / middle[["sd"]]

  # Group `older`
  if (!all(c("mean", "sd") %in% names(older)) || older[["sd"]] <= 0) {
    stop(sprintf("`%s` must be c(mean = , sd = ) with sd > 0.", "older"), call. = FALSE)
  }
  in_group <- group %in% "older"
  z[in_group] <- (raw[in_group] - older[["mean"]]) / older[["sd"]]

  unmatched <- setdiff(unique(group), c("young", "middle", "older"))
  if (length(unmatched) > 0) {
    warning(sprintf("No norms for groups: %s", toString(unmatched)), call. = FALSE)
  }
  return(data.frame(raw = raw, group = group, z = z, t = 50 + 10 * z))
}

#' Percentile ranks against a reference sample
#'
#' @param x A numeric vector of scores to rank.
#' @param reference A numeric vector of scores from the norm sample.
#' @param method One of `"below"`, `"at_or_below"`, `"midpoint"` or
#'   `"normal"`.
#' @return A numeric vector of percentile ranks between 0 and 100.
#' @export
percentile_rank <- function(x, reference,
                            method = c("below", "at_or_below", "midpoint", "normal")) {
  method <- match.arg(method)
  reference <- reference[!is.na(reference)]
  n_ref <- length(reference)
  if (n_ref < 2) {
    stop("`reference` needs at least two non-missing scores.", call. = FALSE)
  }

  ranks <- switch(method,
    below = {
      n_below <- vapply(x, function(v) sum(reference < v), numeric(1))
      100 * n_below / n_ref
    },
    at_or_below = {
      n_at_or_below <- vapply(x, function(v) sum(reference <= v), numeric(1))
      100 * n_at_or_below / n_ref
    },
    midpoint = {
      n_below <- vapply(x, function(v) sum(reference < v), numeric(1))
      n_at <- vapply(x, function(v) sum(reference == v), numeric(1))
      100 * (n_below + n_at / 2) / n_ref
    },
    normal = {
      z <- (x - mean(reference)) / stats::sd(reference)
      100 * stats::pnorm(z)
    }
  )
  ranks[is.na(x)] <- NA_real_
  return(ranks)
}

#' Describe one standard score on several norm scales
#'
#' @param z A single z score.
#' @return A list with elements `z`, `t`, `iq`, `percentile`, `stanine`,
#'   `sten` and `band`.
#' @export
describe_norm_score <- function(z) {
  if (!is.numeric(z) || length(z) != 1L || is.na(z)) {
    stop("`z` must be a single non-missing number.", call. = FALSE)
  }
  # 1. T score
  t <- round(50 + 10 * z)
  # 2. IQ-metric score
  iq <- round(100 + 15 * z)
  # 3. Percentile
  percentile <- round(100 * stats::pnorm(z), 1)
  # 4. Stanine
  stanine <- min(max(round(5 + 2 * z), 1), 9)
  # 5. Sten
  sten <- min(max(round(5.5 + 2 * z), 1), 10)
  # 6. Descriptive band from the T score
  band <- if (t < 30) {
    "very low"
  } else if (t < 40) {
    "low"
  } else if (t <= 60) {
    "average"
  } else if (t <= 70) {
    "high"
  } else {
    "very high"
  }

  return(list(
    z = z, t = t, iq = iq, percentile = percentile,
    stanine = stanine, sten = sten, band = band
  ))
}

#' Confidence band around a test score
#'
#' @param score A numeric vector of observed scores.
#' @param reliability The reliability of the test.
#' @param sd The standard deviation of the norm sample.
#' @param mean The mean of the norm sample.
#' @param level The confidence level.
#' @param type One of `"observed"`, `"true"` or `"difference"`.
#' @return A data frame with columns `score`, `centre`, `se`, `lower` and
#'   `upper`.
#' @export
confidence_band <- function(score, reliability, sd, mean = 0, level = 0.95,
                            type = c("observed", "true", "difference")) {
  type <- match.arg(type)
  if (reliability <= 0 || reliability >= 1) {
    stop("`reliability` must lie strictly between 0 and 1.", call. = FALSE)
  }
  q <- stats::qnorm(1 - (1 - level) / 2)

  # Standard error and centre per band type
  switch(type,
    observed = {
      se <- sd * sqrt(1 - reliability)
      centre <- score
    },
    true = {
      se <- sd * sqrt(reliability * (1 - reliability))
      centre <- mean + reliability * (score - mean)
    },
    difference = {
      se <- sd * sqrt(2 * (1 - reliability))
      centre <- score
    }
  )

  lower <- centre - q * se
  upper <- centre + q * se
  return(data.frame(score = score, centre = centre, se = se, lower = lower, upper = upper))
}
