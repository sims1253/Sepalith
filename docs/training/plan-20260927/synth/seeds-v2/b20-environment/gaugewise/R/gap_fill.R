#' @title Locate runs of missing values
#'
#' @description Describes each contiguous block of `NA` values in a series by its start
#'   position, end position and length.
#'
#' @param x A numeric vector of observations at a regular time step.
#' @return A data frame with integer columns `start`, `end` and `length`; zero
#'   rows when `x` has no gaps.
#' @export
find_gaps <- function(x) {
  miss <- is.na(x)
  if (!any(miss)) {
    return(data.frame(start = integer(0), end = integer(0), length = integer(0)))
  }

  runs <- rle(miss)
  ends <- cumsum(runs$lengths)
  starts <- ends - runs$lengths + 1L
  # rle() alternates observed and missing runs; keep the missing ones
  keep <- runs$values
  return(data.frame(
    start = as.integer(starts[keep]),
    end = as.integer(ends[keep]),
    length = as.integer(runs$lengths[keep])
  ))
}

#' @title Fill short gaps by linear interpolation
#'
#' @description Interpolates linearly across gaps no longer than `max_gap` that are bounded
#'   by observed values on both sides.
#'
#' @param x A numeric vector of observations at a regular time step.
#' @param max_gap Longest gap, in time steps, that will be filled.
#' @return A list with elements `value` (the filled series) and `filled` (a
#'   logical vector marking imputed positions).
#' @export
fill_linear <- function(x, max_gap = 3L) {
  gaps <- find_gaps(x)
  value <- x
  filled <- rep(FALSE, length(x))

  for (i in seq_len(nrow(gaps))) {
    g <- gaps[i, ]
    # Leading and trailing gaps have no anchor on one side
    if (g$length > max_gap || g$start == 1L || g$end == length(x)) {
      next
    }

    left <- g$start - 1L
    right <- g$end + 1L
    idx <- g$start:g$end
    value[idx] <- x[left] + (x[right] - x[left]) * (idx - left) / (right - left)
    filled[idx] <- TRUE
  }
  return(list(value = value, filled = filled))
}

#' @title Fill gaps from a day-of-year climatology
#'
#' @description Replaces missing values with the mean of all observed values sharing the
#'   same day of year, smoothed over a centred window.
#'
#' @param x A numeric vector of observations at a regular time step.
#' @param date A vector of class `Date` matching `x`.
#' @param window Half-width of the smoothing window in days.
#' @return A list with elements `value` (the filled series) and `filled` (a
#'   logical vector marking imputed positions).
#' @export
fill_climatology <- function(x, date, window = 7L) {
  if (length(x) != length(date)) {
    stop("`x` and `date` must have the same length.", call. = FALSE)
  }

  # Fold leap days onto 31 December so every year has 365 slots
  doy <- as.integer(format(date, "%j"))
  doy[doy == 366L] <- 365L

  clim <- rep(NA_real_, 365L)
  for (d in seq_len(365L)) {
    # Circular day distance, so late December pools with early January
    near <- ((doy - d + 182L) %% 365L) - 182L
    pool <- x[abs(near) <= window]
    clim[d] <- if (all(is.na(pool))) NA_real_ else mean(pool, na.rm = TRUE)
  }

  value <- x
  miss <- is.na(x)
  value[miss] <- clim[doy[miss]]
  filled <- miss & !is.na(value)
  return(list(value = value, filled = filled))
}

#' @title Fill gaps by regression on a neighbouring station
#'
#' @description Fits a linear regression of the target on a neighbour over overlapping
#'   observations and predicts the target where it is missing.
#'
#' @param x A numeric vector of observations at a regular time step.
#' @param neighbour A numeric vector from a nearby station, aligned with `x`.
#' @param min_overlap Minimum number of jointly observed values required.
#' @param min_r2 Minimum coefficient of determination for the fit to be used.
#' @return A list with elements `value` (the filled series) and `filled` (a
#'   logical vector marking imputed positions).
#' @export
fill_neighbour <- function(x, neighbour, min_overlap = 60L, min_r2 = 0.7) {
  if (length(x) != length(neighbour)) {
    stop("`x` and `neighbour` must have the same length.", call. = FALSE)
  }

  both <- !is.na(x) & !is.na(neighbour)
  value <- x
  filled <- rep(FALSE, length(x))
  if (sum(both) < min_overlap) {
    warning("Not enough overlap with neighbour; no gaps filled.", call. = FALSE)
    return(list(value = value, filled = filled))
  }

  fit <- stats::lm(x[both] ~ neighbour[both])
  r2 <- summary(fit)$r.squared
  if (r2 < min_r2) {
    warning(sprintf("Neighbour fit too weak (R2 = %.2f).", r2), call. = FALSE)
    return(list(value = value, filled = filled))
  }

  coefs <- stats::coef(fit)
  target <- is.na(x) & !is.na(neighbour)
  value[target] <- coefs[[1]] + coefs[[2]] * neighbour[target]
  filled[target] <- TRUE
  return(list(value = value, filled = filled))
}

#' @title Fill gaps with a cascade of methods
#'
#' @description Applies the requested fillers in order, each one only touching values that
#'   are still missing after the previous step, and records which method filled
#'   each value.
#'
#' @param x A numeric vector of observations at a regular time step.
#' @param date A vector of class `Date` matching `x`.
#' @param neighbour Optional numeric vector from a nearby station, aligned
#'   with `x`.
#' @param methods Character vector of fillers to apply, in order; any of
#'   `"linear"`, `"neighbour"` and `"climatology"`.
#' @param max_gap Longest gap, in time steps, that will be filled.
#' @return A data frame with columns `date`, `value`, `original` and `source`.
#' @export
fill_cascade <- function(x, date, neighbour = NULL,
                         methods = c("linear", "neighbour", "climatology"),
                         max_gap = 3L) {
  methods <- match.arg(methods, several.ok = TRUE)
  value <- x
  source <- ifelse(is.na(x), NA_character_, "observed")

  for (m in methods) {
    if (!anyNA(value)) {
      break
    }
    step <- switch(m,
      linear = fill_linear(value, max_gap = max_gap),
      neighbour = if (is.null(neighbour)) NULL else fill_neighbour(value, neighbour),
      climatology = fill_climatology(value, date)
    )
    if (is.null(step)) {
      next
    }

    # Each filler only sees what is still missing, so sources never overlap
    value <- step$value
    source[step$filled] <- m
  }

  return(data.frame(
    date = date,
    value = value,
    original = x,
    source = source
  ))
}

#' @title Summarise gap filling by source
#'
#' @description Counts observed, filled and still-missing values from the output of
#'   [fill_cascade()].
#'
#' @param filled A data frame returned by [fill_cascade()].
#' @return A named integer vector of counts per source, including `"missing"`.
#' @export
fill_summary <- function(filled) {
  src <- filled$source
  src[is.na(src)] <- "missing"
  counts <- table(src)
  return(stats::setNames(as.integer(counts), names(counts)))
}

#' @title Evaluate gap fillers on artificially removed data
#'
#' @description Removes random blocks of observed values, fills them with each method and
#'   compares the filled values with the withheld observations.
#'
#' @param x A numeric vector of observations at a regular time step.
#' @param date A vector of class `Date` matching `x`.
#' @param neighbour Optional numeric vector from a nearby station, aligned
#'   with `x`.
#' @param gap_length Length of each artificial gap, in time steps.
#' @param n_gaps Number of artificial gaps to insert.
#' @param seed Optional random seed for reproducibility.
#' @return A data frame with one row per method and columns `method`, `n`,
#'   `bias` and `rmse`.
#' @export
evaluate_fillers <- function(x, date, neighbour = NULL, gap_length = 3L,
                             n_gaps = 50L, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Gaps must start and end inside the record so linear filling can work
  candidates <- which(!is.na(x))
  candidates <- candidates[candidates > 1L & candidates < length(x) - gap_length]
  # Index via sample.int(): sample(v) on a length-one v would draw from 1:v
  starts <- candidates[sample.int(length(candidates), min(n_gaps, length(candidates)))]
  held <- unique(unlist(lapply(starts, function(s) s:(s + gap_length - 1L))))
  held <- held[!is.na(x[held])]
  masked <- x
  masked[held] <- NA_real_

  methods <- c("linear", "climatology")
  if (!is.null(neighbour)) {
    methods <- c(methods, "neighbour")
  }

  rows <- vector("list", length(methods))
  for (k in seq_along(methods)) {
    filled <- switch(methods[k],
      linear = fill_linear(masked, max_gap = gap_length),
      climatology = fill_climatology(masked, date),
      neighbour = fill_neighbour(masked, neighbour)
    )

    err <- filled$value[held] - x[held]
    err <- err[!is.na(err)]
    rows[[k]] <- data.frame(
      method = methods[k],
      n = length(err),
      bias = if (length(err) > 0) mean(err) else NA_real_,
      rmse = if (length(err) > 0) sqrt(mean(err^2)) else NA_real_
    )
  }
  return(do.call(rbind, rows))
}
