#' @title Compute exceedance probabilities for a discharge series
#'
#' @description Ranks daily discharge values from largest to smallest and assigns each a
#'   plotting-position exceedance probability.
#'
#' @param flow A numeric vector of daily discharge values in cubic metres per second.
#' @param method Plotting-position formula, one of `"weibull"`, `"gringorten"`
#'   or `"cunnane"`.
#' @param na_rm Logical; drop missing discharge values before ranking.
#' @return A data frame with columns `flow`, `rank` and `exceedance`, sorted
#'   by decreasing flow.
#' @export
fdc_exceedance <- function(flow, method = c("weibull", "gringorten", "cunnane"),
                           na_rm = TRUE) {
  method <- match.arg(method)
  if (!is.numeric(flow)) {
    stop("`flow` must be numeric.", call. = FALSE)
  }
  if (na_rm) {
    flow <- flow[!is.na(flow)]
  } else if (anyNA(flow)) {
    stop("`flow` contains missing values; set `na_rm = TRUE`.", call. = FALSE)
  }

  n <- length(flow)
  sorted <- sort(flow, decreasing = TRUE)
  ranks <- seq_len(n)
  # Rank 1 is the largest flow, so exceedance grows down the table
  exceedance <- switch(method,
    weibull = ranks / (n + 1),
    gringorten = (ranks - 0.44) / (n + 0.12),
    cunnane = (ranks - 0.4) / (n + 0.2)
  )

  data.frame(flow = sorted, rank = ranks, exceedance = exceedance)
}

#' @title Interpolate a flow quantile from a flow duration curve
#'
#' @description Returns the discharge equalled or exceeded for a given percentage of time,
#'   interpolating linearly in exceedance space.
#'
#' @param flow A numeric vector of daily discharge values in cubic metres per second.
#' @param percent Numeric vector of exceedance percentages between 0 and 100.
#' @param method Plotting-position formula passed to [fdc_exceedance()].
#' @return A named numeric vector of discharges, one per element of `percent`.
#' @export
fdc_quantile <- function(flow, percent, method = "weibull") {
  if (anyNA(percent) || any(percent <= 0 | percent >= 100)) {
    stop("`percent` must lie strictly between 0 and 100.", call. = FALSE)
  }

  curve <- fdc_exceedance(flow, method = method)
  # rule = 2 clamps requests beyond the observed tails to the extreme flows
  values <- stats::approx(
    x = curve$exceedance,
    y = curve$flow,
    xout = percent / 100,
    rule = 2,
    ties = mean
  )$y

  names(values) <- paste0("Q", formatC(percent, format = "fg"))
  values
}

#' @title Standard flow duration indices
#'
#' @description Convenience wrappers for the high-flow index Q5, the median flow Q50 and
#'   the low-flow index Q95.
#'
#' @param flow A numeric vector of daily discharge values in cubic metres per second.
#' @return A single numeric discharge value.
#' @rdname fdc_indices
#' @export
fdc_q5 <- function(flow) {
  unname(fdc_quantile(flow, percent = 5))
}

#' @rdname fdc_indices
#' @export
fdc_q50 <- function(flow) {
  unname(fdc_quantile(flow, percent = 50))
}

#' @rdname fdc_indices
#' @export
fdc_q95 <- function(flow) {
  unname(fdc_quantile(flow, percent = 95))
}

#' @title Slope of the mid-segment of a flow duration curve
#'
#' @description Computes the log-space slope between two exceedance levels, a common
#'   signature of catchment flashiness.
#'
#' @param flow A numeric vector of daily discharge values in cubic metres per second.
#' @param lower,upper Exceedance percentages bounding the mid-segment.
#' @param zero_flow Value substituted for zero discharge before taking logs.
#' @return A single numeric slope; more negative values indicate a flashier
#'   regime.
#' @export
fdc_slope <- function(flow, lower = 33, upper = 66, zero_flow = 1e-3) {
  if (lower >= upper) {
    stop("`lower` must be smaller than `upper`.", call. = FALSE)
  }

  q <- fdc_quantile(flow, percent = c(lower, upper))
  # Ephemeral streams have zero flows; avoid log(0)
  q[q <= 0] <- zero_flow
  (log(q[[1]]) - log(q[[2]])) / ((lower - upper) / 100)
}

#' @title Summarise flow duration curves by group
#'
#' @description Splits a discharge table by a grouping column and computes standard flow
#'   duration indices for each group.
#'
#' @param data A data frame of gauged discharge with one row per day.
#' @param flow_col Name of the discharge column in `data`.
#' @param group_col Name of the grouping column in `data`, for example a
#'   station identifier or water year.
#' @param min_days Minimum number of non-missing days required per group.
#' @return A data frame with one row per group and columns `group`, `n_days`,
#'   `q5`, `q50`, `q95` and `slope`.
#' @export
fdc_summary <- function(data, flow_col = "flow", group_col = "station",
                        min_days = 300) {
  missing_cols <- setdiff(c(flow_col, group_col), names(data))
  if (length(missing_cols) > 0) {
    stop("Missing columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  groups <- split(data[[flow_col]], data[[group_col]])
  out <- vector("list", length(groups))
  for (i in seq_along(groups)) {
    flow <- groups[[i]][!is.na(groups[[i]])]
    # Short records still get a row, just without indices
    if (length(flow) < min_days) {
      out[[i]] <- data.frame(
        group = names(groups)[i], n_days = length(flow),
        q5 = NA_real_, q50 = NA_real_, q95 = NA_real_, slope = NA_real_
      )
      next
    }

    out[[i]] <- data.frame(
      group = names(groups)[i],
      n_days = length(flow),
      q5 = fdc_q5(flow),
      q50 = fdc_q50(flow),
      q95 = fdc_q95(flow),
      slope = fdc_slope(flow)
    )
  }

  result <- do.call(rbind, out)
  rownames(result) <- NULL
  result
}

#' @title Compare flow duration curves between two periods
#'
#' @description Evaluates both curves at a common set of exceedance levels and reports
#'   absolute and relative differences, for example before and after a land
#'   use change.
#'
#' @param before,after Numeric vectors of daily discharge for the reference
#'   and comparison periods.
#' @param percent Numeric vector of exceedance percentages between 0 and 100.
#' @param segments Named list of exceedance ranges used to summarise the
#'   high, mid and low flow segments.
#' @return A list with elements `curve` (a data frame of paired quantiles) and
#'   `segments` (mean relative change per segment).
#' @export
fdc_compare <- function(before, after, percent = seq(1, 99, by = 1),
                        segments = list(high = c(0, 20), mid = c(20, 70),
                                        low = c(70, 100))) {
  q_before <- unname(fdc_quantile(before, percent = percent))
  q_after <- unname(fdc_quantile(after, percent = percent))

  # Relative change is undefined where the reference flow is zero
  rel <- ifelse(q_before > 0, (q_after - q_before) / q_before, NA_real_)
  curve <- data.frame(
    percent = percent,
    before = q_before,
    after = q_after,
    difference = q_after - q_before,
    relative = rel
  )

  seg_change <- numeric(length(segments))
  names(seg_change) <- names(segments)
  for (s in names(segments)) {
    # Segments are half-open, (lower, upper], so neighbours do not overlap
    bounds <- segments[[s]]
    in_seg <- percent > bounds[1] & percent <= bounds[2]
    if (!any(in_seg)) {
      seg_change[[s]] <- NA_real_
    } else {
      seg_change[[s]] <- mean(rel[in_seg], na.rm = TRUE)
    }
  }

  list(curve = curve, segments = seg_change)
}
