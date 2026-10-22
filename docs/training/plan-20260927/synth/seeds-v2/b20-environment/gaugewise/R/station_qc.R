#' @title Quality-control flag codes
#'
#' @description Named integer codes used by all `qc_*()` functions. Higher codes are more
#'   severe.
#'
#' @keywords internal
qc_codes <- function() {
  c(good = 0L, suspect = 1L, bad = 2L, missing = 9L)
}

#' @title Flag values outside a plausible range
#'
#' @description Marks observations below `lower` or above `upper` as bad and observations
#'   within `margin` of either limit as suspect.
#'
#' @param x A numeric vector of station observations in time order.
#' @param lower,upper Physical limits for the variable.
#' @param margin Width of the suspect band inside each limit.
#' @return An integer vector of QC flags the same length as `x`.
#' @export
qc_range <- function(x, lower, upper, margin = 0) {
  codes <- qc_codes()
  flag <- rep(codes[["good"]], length(x))

  # Apply milder flags first so the more severe ones overwrite them
  near <- x < lower + margin | x > upper - margin
  flag[!is.na(near) & near] <- codes[["suspect"]]
  outside <- x < lower | x > upper
  flag[!is.na(outside) & outside] <- codes[["bad"]]
  flag[is.na(x)] <- codes[["missing"]]
  flag
}

#' @title Flag sudden jumps between consecutive observations
#'
#' @description Marks observations whose absolute change from the previous value exceeds
#'   `max_step` as suspect, and twice that as bad.
#'
#' @param x A numeric vector of station observations in time order.
#' @param max_step Largest plausible change between consecutive observations.
#' @return An integer vector of QC flags the same length as `x`.
#' @export
qc_step <- function(x, max_step) {
  codes <- qc_codes()
  flag <- rep(codes[["good"]], length(x))
  # The first value has no predecessor and cannot be judged
  jump <- abs(c(NA_real_, diff(x)))
  flag[!is.na(jump) & jump > max_step] <- codes[["suspect"]]
  flag[!is.na(jump) & jump > 2 * max_step] <- codes[["bad"]]
  flag[is.na(x)] <- codes[["missing"]]
  flag
}

#' @title Flag stuck sensors
#'
#' @description Marks runs of identical consecutive values that last at least `min_run`
#'   observations as suspect, which often indicates a frozen or failed sensor.
#'
#' @param x A numeric vector of station observations in time order.
#' @param min_run Minimum run length that is considered implausible.
#' @param tolerance Values closer than this are treated as identical.
#' @return An integer vector of QC flags the same length as `x`.
#' @export
qc_persistence <- function(x, min_run = 6L, tolerance = 1e-8) {
  codes <- qc_codes()
  flag <- rep(codes[["good"]], length(x))

  same <- c(FALSE, abs(diff(x)) <= tolerance)
  same[is.na(same)] <- FALSE
  # A new run starts wherever the value changes
  run_id <- cumsum(!same)
  run_len <- stats::ave(run_id, run_id, FUN = length)
  flag[run_len >= min_run] <- codes[["suspect"]]
  flag[is.na(x)] <- codes[["missing"]]
  flag
}

#' @title Flag inconsistent daily minimum and maximum temperatures
#'
#' @description Marks days where the minimum exceeds the maximum as bad on both series,
#'   and days with an implausibly small diurnal range as suspect.
#'
#' @param tmin,tmax Numeric vectors of daily minimum and maximum temperature.
#' @param min_range Smallest plausible diurnal range in degrees.
#' @return A list with integer flag vectors `tmin` and `tmax`.
#' @export
qc_temperature_pair <- function(tmin, tmax, min_range = 0.2) {
  if (length(tmin) != length(tmax)) {
    stop("`tmin` and `tmax` must have the same length.", call. = FALSE)
  }

  codes <- qc_codes()
  flag <- rep(codes[["good"]], length(tmin))
  spread <- tmax - tmin
  flag[!is.na(spread) & spread < min_range] <- codes[["suspect"]]
  flag[!is.na(spread) & spread < 0] <- codes[["bad"]]

  # Missingness is specific to each series
  flag_min <- flag
  flag_max <- flag
  flag_min[is.na(tmin)] <- codes[["missing"]]
  flag_max[is.na(tmax)] <- codes[["missing"]]
  list(tmin = flag_min, tmax = flag_max)
}

#' @title Combine several QC flag vectors
#'
#' @description Takes the most severe flag at each position, so that a value flagged bad
#'   by any check stays bad.
#'
#' @param ... Integer flag vectors of equal length.
#' @return A single integer flag vector.
#' @export
qc_combine <- function(...) {
  flags <- list(...)
  if (length(flags) == 0) {
    stop("Supply at least one flag vector.", call. = FALSE)
  }
  lens <- vapply(flags, length, integer(1))
  if (length(unique(lens)) != 1) {
    stop("All flag vectors must have the same length.", call. = FALSE)
  }

  Reduce(pmax, flags)
}

#' @title Run the standard QC suite on a station table
#'
#' @description Applies range, step and persistence checks to each configured variable and
#'   adds a `<variable>_qc` flag column for each.
#'
#' @param data A data frame of station observations, sorted by time.
#' @param limits A named list; each element is a list with `lower`, `upper`
#'   and `max_step` for the variable of the same name.
#' @param min_run Minimum run length passed to [qc_persistence()].
#' @return `data` with one additional integer flag column per variable.
#' @export
qc_station <- function(data, limits, min_run = 6L) {
  vars <- intersect(names(limits), names(data))
  if (length(vars) == 0) {
    warning("No variables in `limits` match columns of `data`.", call. = FALSE)
    return(data)
  }

  for (v in vars) {
    lim <- limits[[v]]
    x <- data[[v]]
    flag <- qc_combine(
      qc_range(x, lower = lim$lower, upper = lim$upper),
      qc_step(x, max_step = lim$max_step),
      qc_persistence(x, min_run = min_run)
    )
    data[[paste0(v, "_qc")]] <- flag
  }

  # Cross-variable check, only when both temperature series were configured
  if (all(c("tmin", "tmax") %in% vars)) {
    pair <- qc_temperature_pair(data$tmin, data$tmax)
    data$tmin_qc <- pmax(data$tmin_qc, pair$tmin)
    data$tmax_qc <- pmax(data$tmax_qc, pair$tmax)
  }
  data
}

#' @title Tabulate QC flags per variable
#'
#' @description Counts how many observations of each flagged variable fall in each QC
#'   category.
#'
#' @param data A data frame of station observations, sorted by time.
#' @return A data frame with one row per variable and one count column per
#'   flag category.
#' @export
qc_report <- function(data) {
  qc_cols <- grep("_qc$", names(data), value = TRUE)
  codes <- qc_codes()
  out <- matrix(0L, nrow = length(qc_cols), ncol = length(codes),
                dimnames = list(NULL, names(codes)))

  for (i in seq_along(qc_cols)) {
    # Fix the levels so absent categories still get a zero column
    counts <- table(factor(data[[qc_cols[i]]], levels = codes))
    out[i, ] <- as.integer(counts)
  }

  data.frame(variable = sub("_qc$", "", qc_cols), out, row.names = NULL)
}

#' @title Flag values that disagree with neighbouring stations
#'
#' @description Compares each observation with the median of the same time step at
#'   neighbouring stations, scaled by the robust spread of past differences.
#'
#' @param x A numeric vector of station observations in time order.
#' @param neighbours A numeric matrix with one column per neighbouring station
#'   and one row per element of `x`.
#' @param threshold Robust z-score above which a value is suspect; values
#'   beyond twice this are bad.
#' @param min_neighbours Minimum number of reporting neighbours required.
#' @return An integer vector of QC flags the same length as `x`.
#' @export
qc_spatial <- function(x, neighbours, threshold = 4, min_neighbours = 2L) {
  neighbours <- as.matrix(neighbours)
  if (nrow(neighbours) != length(x)) {
    stop("`neighbours` must have one row per element of `x`.", call. = FALSE)
  }

  codes <- qc_codes()
  flag <- rep(codes[["good"]], length(x))

  # Reference series: median of the neighbours where enough of them report
  n_report <- rowSums(!is.na(neighbours))
  ref <- rep(NA_real_, length(x))
  usable <- n_report >= min_neighbours
  ref[usable] <- apply(neighbours[usable, , drop = FALSE], 1, stats::median, na.rm = TRUE)

  # Robust z-scores of the station-minus-reference differences
  diffs <- x - ref
  centre <- stats::median(diffs, na.rm = TRUE)
  spread <- stats::mad(diffs, na.rm = TRUE)
  if (is.na(spread) || spread == 0) {
    warning("Spatial differences have zero spread; no values flagged.", call. = FALSE)
    flag[is.na(x)] <- codes[["missing"]]
    return(flag)
  }

  z <- abs(diffs - centre) / spread
  for (i in seq_along(x)) {
    if (is.na(x[i])) {
      flag[i] <- codes[["missing"]]
    } else if (is.na(z[i])) {
      next
    } else if (z[i] > 2 * threshold) {
      flag[i] <- codes[["bad"]]
    } else if (z[i] > threshold) {
      flag[i] <- codes[["suspect"]]
    }
  }
  flag
}
