#' @title Check a discharge vector before baseflow filtering
#'
#' @description Stops with an informative error when the series is too short, contains
#'   negative values or is not numeric.
#'
#' @param flow A numeric vector of daily streamflow, without gaps.
#' @param min_length Minimum number of observations required.
#' @keywords internal
check_baseflow_input <- function(flow, min_length = 30) {
  if (!is.numeric(flow)) {
    stop("`flow` must be a numeric vector.", call. = FALSE)
  }
  if (anyNA(flow)) {
    stop("`flow` must not contain missing values; fill gaps first.", call. = FALSE)
  }
  if (any(flow < 0)) {
    stop("`flow` must be non-negative.", call. = FALSE)
  }
  if (length(flow) < min_length) {
    stop(sprintf("`flow` needs at least %d values.", min_length), call. = FALSE)
  }
  return(invisible(flow))
}

#' @title One-parameter digital filter for baseflow separation
#'
#' @description Applies alternating forward and backward passes of a recursive high-pass
#'   filter to remove quickflow, constraining baseflow to lie between zero and
#'   total streamflow.
#'
#' @param flow A numeric vector of daily streamflow, without gaps.
#' @param alpha Filter parameter between 0 and 1.
#' @param passes Number of alternating forward and backward passes.
#' @return A numeric vector of baseflow the same length as `flow`.
#' @export
baseflow_one_param <- function(flow, alpha = 0.925, passes = 3) {
  check_baseflow_input(flow)
  if (alpha <= 0 || alpha >= 1) {
    stop("`alpha` must be between 0 and 1.", call. = FALSE)
  }

  base <- flow
  n <- length(flow)
  for (p in seq_len(passes)) {
    # Odd passes run forward in time, even passes backward
    idx <- if (p %% 2 == 1) seq_len(n) else rev(seq_len(n))
    input <- base[idx]
    quick <- numeric(n)
    for (t in 2:n) {
      quick[t] <- alpha * quick[t - 1] + (1 + alpha) / 2 * (input[t] - input[t - 1])
      # Keep baseflow between zero and the streamflow itself
      if (quick[t] < 0) {
        quick[t] <- 0
      } else if (quick[t] > input[t]) {
        quick[t] <- input[t]
      }
    }

    base[idx] <- input - quick
  }
  return(base)
}

#' @title Two-parameter recursive filter for baseflow separation
#'
#' @description Separates baseflow using a recession constant and a maximum baseflow
#'   index, which suits catchments with well characterised aquifers.
#'
#' @param flow A numeric vector of daily streamflow, without gaps.
#' @param alpha Recession constant between 0 and 1.
#' @param bfi_max Maximum attainable baseflow index between 0 and 1.
#' @return A numeric vector of baseflow the same length as `flow`.
#' @export
baseflow_two_param <- function(flow, alpha = 0.98, bfi_max = 0.8) {
  check_baseflow_input(flow)
  if (alpha <= 0 || alpha >= 1) {
    stop("`alpha` must be between 0 and 1.", call. = FALSE)
  }
  if (bfi_max <= 0 || bfi_max >= 1) {
    stop("`bfi_max` must be between 0 and 1.", call. = FALSE)
  }

  n <- length(flow)
  base <- numeric(n)
  base[1] <- flow[1] * bfi_max
  denom <- 1 - alpha * bfi_max

  for (t in 2:n) {
    b <- ((1 - bfi_max) * alpha * base[t - 1] + (1 - alpha) * bfi_max * flow[t]) / denom
    base[t] <- min(b, flow[t])
  }
  return(base)
}

#' @title Fixed-interval minimum method for baseflow separation
#'
#' @description Splits the series into consecutive blocks and assigns each day the minimum
#'   discharge of its block.
#'
#' @param flow A numeric vector of daily streamflow, without gaps.
#' @param interval Block length in days; odd values are recommended.
#' @return A numeric vector of baseflow the same length as `flow`.
#' @export
baseflow_fixed_interval <- function(flow, interval = 5) {
  check_baseflow_input(flow)
  n <- length(flow)

  block <- ceiling(seq_len(n) / interval)
  block_min <- tapply(flow, block, min)
  # Look minima up by block label so every day gets its own block's value
  base <- unname(block_min[as.character(block)])
  return(as.numeric(base))
}

#' @title Separate baseflow with a named method
#'
#' @description Dispatches to one of the baseflow filters and returns both the baseflow and
#'   quickflow components alongside the input series.
#'
#' @param flow A numeric vector of daily streamflow, without gaps.
#' @param method Separation method, one of `"one_param"`, `"two_param"` or
#'   `"fixed_interval"`.
#' @param ... Passed to the selected filter function.
#' @return A data frame with columns `flow`, `baseflow` and `quickflow`.
#' @export
separate_baseflow <- function(flow, method = c("one_param", "two_param", "fixed_interval"),
                              ...) {
  method <- match.arg(method)

  base <- switch(method,
    one_param = baseflow_one_param(flow, ...),
    two_param = baseflow_two_param(flow, ...),
    fixed_interval = baseflow_fixed_interval(flow, ...)
  )

  return(data.frame(
    flow = flow,
    baseflow = base,
    quickflow = flow - base
  ))
}

#' @title Baseflow index
#'
#' @description Ratio of total baseflow volume to total streamflow volume.
#'
#' @param flow A numeric vector of daily streamflow, without gaps.
#' @param baseflow A numeric vector of separated baseflow matching `flow`.
#' @return A single numeric value between 0 and 1.
#' @export
baseflow_index <- function(flow, baseflow) {
  if (length(flow) != length(baseflow)) {
    stop("`flow` and `baseflow` must have the same length.", call. = FALSE)
  }

  total <- sum(flow)
  if (total == 0) {
    return(NA_real_)
  }
  return(sum(baseflow) / total)
}

#' @title Estimate a master recession constant
#'
#' @description Identifies recession days, where discharge falls for at least `min_run`
#'   consecutive days, and fits the median ratio of successive flows.
#'
#' @param flow A numeric vector of daily streamflow, without gaps.
#' @param min_run Minimum length of a declining run, in days.
#' @param quantile_cut Upper flow quantile above which days are ignored to
#'   exclude storm response.
#' @return A list with elements `k` (recession constant), `n_pairs` and
#'   `half_life` in days.
#' @export
recession_constant <- function(flow, min_run = 5, quantile_cut = 0.7) {
  check_baseflow_input(flow)

  falling <- c(FALSE, diff(flow) < 0)
  runs <- rle(falling)
  ends <- cumsum(runs$lengths)
  starts <- ends - runs$lengths + 1

  # Days just after a storm peak recede faster; leave them out
  cap <- stats::quantile(flow, quantile_cut, names = FALSE)
  ratios <- numeric(0)
  for (i in seq_along(runs$values)) {
    if (!runs$values[i] || runs$lengths[i] < min_run) {
      next
    }
    days <- starts[i]:ends[i]
    days <- days[days > 1 & flow[days - 1] <= cap & flow[days - 1] > 0]
    if (length(days) > 0) {
      ratios <- c(ratios, flow[days] / flow[days - 1])
    }
  }

  if (length(ratios) == 0) {
    warning("No recession periods found.", call. = FALSE)
    return(list(k = NA_real_, n_pairs = 0L, half_life = NA_real_))
  }

  k <- stats::median(ratios)
  return(list(
    k = k,
    n_pairs = length(ratios),
    half_life = log(0.5) / log(k)
  ))
}

#' @title Identify storm events from quickflow
#'
#' @description Delineates events as runs where quickflow exceeds a threshold, merging
#'   runs separated by short dips, and summarises each event.
#'
#' @param flow A numeric vector of daily streamflow, without gaps.
#' @param baseflow A numeric vector of separated baseflow matching `flow`.
#' @param threshold Minimum quickflow, in the units of `flow`, that counts as
#'   event flow.
#' @param min_gap Shortest interval, in days, that separates two events.
#' @return A data frame with one row per event and columns `start`, `end`,
#'   `peak`, `peak_day` and `quick_volume`.
#' @export
baseflow_events <- function(flow, baseflow, threshold = 0.1, min_gap = 2L) {
  if (length(flow) != length(baseflow)) {
    stop("`flow` and `baseflow` must have the same length.", call. = FALSE)
  }

  quick <- flow - baseflow
  active <- quick > threshold
  # Missing days cannot be part of an event; NA would also break rle()
  active[is.na(active)] <- FALSE
  runs <- rle(active)
  ends <- cumsum(runs$lengths)
  starts <- ends - runs$lengths + 1L
  ev_start <- starts[runs$values]
  ev_end <- ends[runs$values]
  if (length(ev_start) == 0) {
    return(data.frame(
      start = integer(0), end = integer(0), peak = numeric(0),
      peak_day = integer(0), quick_volume = numeric(0)
    ))
  }

  # Merge runs whose separating dip is shorter than `min_gap` days
  merged_start <- ev_start[1]
  merged_end <- ev_end[1]
  for (i in seq_along(ev_start)[-1]) {
    if (ev_start[i] - merged_end[length(merged_end)] - 1L < min_gap) {
      merged_end[length(merged_end)] <- ev_end[i]
    } else {
      merged_start <- c(merged_start, ev_start[i])
      merged_end <- c(merged_end, ev_end[i])
    }
  }

  n_ev <- length(merged_start)
  peak <- numeric(n_ev)
  peak_day <- integer(n_ev)
  volume <- numeric(n_ev)

  for (k in seq_len(n_ev)) {
    days <- merged_start[k]:merged_end[k]
    peak[k] <- max(flow[days], na.rm = TRUE)
    peak_day[k] <- days[which.max(flow[days])]
    # Daily means times seconds per day: a volume when flow is in m3/s
    volume[k] <- sum(quick[days], na.rm = TRUE) * 86400
  }

  return(data.frame(
    start = merged_start,
    end = merged_end,
    peak = peak,
    peak_day = peak_day,
    quick_volume = volume
  ))
}
