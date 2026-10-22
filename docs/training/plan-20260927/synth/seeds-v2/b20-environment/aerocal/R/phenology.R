#' Assign phenological stages from accumulated degree-days
#'
#' Maps cumulative degree-days onto an ordered set of stage thresholds.
#'
#' @param cumDd A numeric vector of cumulative degree-days.
#' @param thresholds A named numeric vector of degree-day thresholds at which
#'   each stage begins, in increasing order.
#' @param before Label used for values below the first threshold.
#' @return An ordered factor of stages the same length as `cumDd`.
#' @export
stageFromDd <- function(cumDd, thresholds, before = "dormant") {
  if (is.unsorted(thresholds)) {
    stop("`thresholds` must be in increasing order.", call. = FALSE)
  }

  labels <- c(before, names(thresholds))
  # findInterval() is 0 below the first threshold, hence the shift by one
  pos <- findInterval(cumDd, thresholds) + 1L
  factor(labels[pos], levels = labels, ordered = TRUE)
}

#' Date on which a degree-day threshold is first reached
#'
#' Linearly interpolates between the last observed day below and the first
#' day at or above the threshold.
#'
#' @param date A vector of class `Date`, sorted increasingly.
#' @param cumDd A numeric vector of cumulative degree-days matching `date`.
#' @param threshold Degree-day total that defines the event.
#' @return A single `Date`, or `NA` if the threshold is never reached.
#' @export
onsetDate <- function(date, cumDd, threshold) {
  hit <- which(cumDd >= threshold)
  if (length(hit) == 0) {
    return(as.Date(NA))
  }

  j <- hit[1]
  # Interpolate from the last non-missing day before the hit
  prev <- which(!is.na(cumDd[seq_len(j - 1L)]))
  if (length(prev) == 0) {
    return(date[j])
  }
  i <- prev[length(prev)]

  frac <- (threshold - cumDd[i]) / (cumDd[j] - cumDd[i])
  date[i] + round(frac * as.numeric(date[j] - date[i]))
}

#' Smooth a vegetation index series
#'
#' Applies a centred moving median followed by a moving mean to suppress
#' cloud-contaminated dips before season detection.
#'
#' @param vi A numeric vector of vegetation index values at a regular step.
#' @param width Odd window width in observations.
#' @return A numeric vector of smoothed values the same length as `vi`.
#' @export
smoothVi <- function(vi, width = 5L) {
  if (width %% 2 == 0) {
    stop("`width` must be odd.", call. = FALSE)
  }

  # Median first so single cloudy dips do not drag the mean down
  med <- stats::runmed(vi, k = width, endrule = "median", na.action = "na.omit")
  stats::filter(med, rep(1 / width, width), sides = 2) |>
    as.numeric() |>
    (\(x) ifelse(is.na(x), med, x))()
}

#' Start of season by amplitude threshold
#'
#' Finds the first time the smoothed index rises above a fraction of the
#' seasonal amplitude, searching only before the seasonal peak.
#'
#' @param doy An integer vector of day-of-year values.
#' @param vi A numeric vector of vegetation index values at a regular step.
#' @param fraction Fraction of the seasonal amplitude that defines the event.
#' @return A single numeric day of year, or `NA`.
#' @export
seasonStart <- function(doy, vi, fraction = 0.5) {
  if (all(is.na(vi))) {
    return(NA_real_)
  }

  peak <- which.max(vi)
  base <- min(vi[seq_len(peak)], na.rm = TRUE)
  level <- base + fraction * (vi[peak] - base)

  for (i in seq_len(peak)[-1]) {
    if (!is.na(vi[i]) && !is.na(vi[i - 1]) && vi[i - 1] < level && vi[i] >= level) {
      # Interpolate the crossing between the two observations
      w <- (level - vi[i - 1]) / (vi[i] - vi[i - 1])
      return(doy[i - 1] + w * (doy[i] - doy[i - 1]))
    }
  }
  NA_real_
}

#' End of season by amplitude threshold
#'
#' Finds the first time the smoothed index falls below a fraction of the
#' seasonal amplitude, searching only after the seasonal peak.
#'
#' @param doy An integer vector of day-of-year values.
#' @param vi A numeric vector of vegetation index values at a regular step.
#' @param fraction Fraction of the seasonal amplitude that defines the event.
#' @return A single numeric day of year, or `NA`.
#' @export
seasonEnd <- function(doy, vi, fraction = 0.5) {
  if (all(is.na(vi))) {
    return(NA_real_)
  }

  peak <- which.max(vi)
  n <- length(vi)
  base <- min(vi[peak:n], na.rm = TRUE)
  level <- base + fraction * (vi[peak] - base)

  for (i in seq.int(peak + 1L, length.out = max(0L, n - peak))) {
    if (!is.na(vi[i]) && !is.na(vi[i - 1]) && vi[i - 1] >= level && vi[i] < level) {
      w <- (vi[i - 1] - level) / (vi[i - 1] - vi[i])
      return(doy[i - 1] + w * (doy[i] - doy[i - 1]))
    }
  }
  NA_real_
}

#' Season metrics for each year
#'
#' Smooths each year's vegetation index series and extracts start, peak and
#' end of season together with the season length.
#'
#' @param data A data frame with columns `year`, `doy` and `vi`.
#' @param fraction Fraction of the seasonal amplitude that defines the event.
#' @param width Odd window width in observations.
#' @return A data frame with one row per year and columns `year`, `sos`,
#'   `pos`, `eos`, `length` and `amplitude`.
#' @export
seasonMetrics <- function(data, fraction = 0.5, width = 5L) {
  # Years with fewer valid points than the window cannot be smoothed
  data |>
    dplyr::arrange(.data$year, .data$doy) |>
    dplyr::group_by(.data$year) |>
    dplyr::filter(sum(!is.na(.data$vi)) >= width) |>
    dplyr::mutate(viSmooth = smoothVi(.data$vi, width = width)) |>
    dplyr::summarise(
      sos = seasonStart(.data$doy, .data$viSmooth, fraction),
      pos = .data$doy[which.max(.data$viSmooth)],
      eos = seasonEnd(.data$doy, .data$viSmooth, fraction),
      amplitude = max(.data$viSmooth, na.rm = TRUE) - min(.data$viSmooth, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(length = .data$eos - .data$sos) |>
    dplyr::relocate("length", .after = "eos")
}

#' Predict stage onset dates for each site and season
#'
#' Combines degree-day accumulation with stage thresholds to give the date on
#' which each stage is first reached.
#'
#' @param data A data frame with columns `site`, `season`, `date` and `cumDd`.
#' @param thresholds A named numeric vector of degree-day thresholds at which
#'   each stage begins, in increasing order.
#' @return A long data frame with columns `site`, `season`, `stage` and
#'   `onset`.
#' @export
stageOnsets <- function(data, thresholds) {
  data |>
    dplyr::arrange(.data$site, .data$date) |>
    dplyr::group_by(.data$site, .data$season) |>
    dplyr::reframe(
      stage = names(thresholds),
      onset = do.call(c, lapply(thresholds, \(t) onsetDate(.data$date, .data$cumDd, t)))
    ) |>
    dplyr::mutate(stage = factor(.data$stage, levels = names(thresholds)))
}

#' Thermal growing season from daily mean temperature
#'
#' The season starts on the first day of the first run of `runLength` days
#' above `threshold`, and ends on the day before the first such run below the
#' threshold after `midYear`.
#'
#' @param date A vector of class `Date`, sorted increasingly.
#' @param tmean A numeric vector of daily mean air temperature in degrees
#'   Celsius.
#' @param threshold Temperature threshold in degrees Celsius.
#' @param runLength Number of consecutive days that defines a run.
#' @param midYear Day of year after which an end of season is sought.
#' @return A data frame with one row per year and columns `year`, `start`,
#'   `end` and `length`.
#' @export
growingSeason <- function(date, tmean, threshold = 5, runLength = 6L, midYear = 182L) {
  yr <- as.integer(format(date, "%Y"))
  doy <- as.integer(format(date, "%j"))
  years <- sort(unique(yr))
  start <- rep(as.Date(NA), length(years))
  end <- rep(as.Date(NA), length(years))

  for (k in seq_along(years)) {
    idx <- which(yr == years[k])
    # Missing days break a warm run rather than extending it
    warm <- tmean[idx] > threshold
    warm[is.na(warm)] <- FALSE

    above <- 0L
    below <- 0L
    for (j in seq_along(idx)) {
      if (is.na(start[k])) {
        above <- if (warm[j]) above + 1L else 0L
        if (above == runLength) {
          start[k] <- date[idx[j - runLength + 1L]]
        }
      } else if (doy[idx[j]] > midYear) {
        below <- if (!warm[j]) below + 1L else 0L
        if (below == runLength) {
          end[k] <- date[idx[j - runLength + 1L]] - 1
          break
        }
      }
    }

    # No cold run after midsummer: the season lasts to the end of the record
    if (!is.na(start[k]) && is.na(end[k])) {
      end[k] <- date[idx[length(idx)]]
    }
  }

  data.frame(
    year = years,
    start = start,
    end = end,
    length = as.integer(end - start) + 1L
  )
}
