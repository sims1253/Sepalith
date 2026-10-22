#' Construct a sensor calibration object
#'
#' @param type Calibration model type.
#' @param coef A named numeric vector of model coefficients.
#' @param nObs Number of collocated observations used in the fit.
#' @return An object of class `sensor_cal`.
#' @keywords internal
newSensorCal <- function(type, coef, nObs) {
  return(structure(
    list(type = type, coef = coef, nObs = nObs, created = Sys.time()),
    class = "sensor_cal"
  ))
}

#' Linear calibration against a reference instrument
#'
#' Fits `reference = intercept + slope * raw` by ordinary least squares over
#' collocated observations.
#'
#' @param raw A numeric vector of raw sensor readings.
#' @param reference A numeric vector of collocated reference readings.
#' @return An object of class `sensor_cal`.
#' @family calibration fits
#' @export
fitLinearCal <- function(raw, reference) {
  ok <- !is.na(raw) & !is.na(reference)
  if (sum(ok) < 10) {
    stop("At least 10 collocated observations are required.", call. = FALSE)
  }
  fit <- stats::lm(reference[ok] ~ raw[ok])
  cf <- unname(stats::coef(fit))
  return(newSensorCal("linear", c(intercept = cf[1], slope = cf[2]), sum(ok)))
}

#' Two-point calibration from zero and span checks
#'
#' Derives offset and gain from readings taken in a zero gas (or clean air)
#' and a known span standard.
#'
#' @param rawZero,rawSpan Raw sensor readings at the zero and span points.
#' @param refZero,refSpan Reference values at the zero and span points.
#' @return An object of class `sensor_cal`.
#' @family calibration fits
#' @export
fitTwoPointCal <- function(rawZero, rawSpan, refZero = 0, refSpan) {
  rawZero <- mean(rawZero, na.rm = TRUE)
  rawSpan <- mean(rawSpan, na.rm = TRUE)
  if (isTRUE(all.equal(rawZero, rawSpan))) {
    stop("Zero and span readings are identical; cannot compute gain.", call. = FALSE)
  }
  slope <- (refSpan - refZero) / (rawSpan - rawZero)
  intercept <- refZero - slope * rawZero
  return(newSensorCal("two_point", c(intercept = intercept, slope = slope), 2L))
}

#' Humidity-corrected calibration for optical particle sensors
#'
#' Fits a hygroscopic growth correction in which raw readings are divided by
#' a humidity-dependent factor before a linear rescaling.
#'
#' @param raw A numeric vector of raw sensor readings.
#' @param reference A numeric vector of collocated reference readings.
#' @param rh A numeric vector of relative humidity in percent.
#' @param kappa Hygroscopicity parameter of the aerosol.
#' @return An object of class `sensor_cal`.
#' @family calibration fits
#' @export
fitHumidityCal <- function(raw, reference, rh, kappa = 0.4) {
  ok <- !is.na(raw) & !is.na(reference) & !is.na(rh) & rh < 99
  if (sum(ok) < 10) {
    stop("At least 10 collocated observations are required.", call. = FALSE)
  }
  growth <- 1 + (kappa / 1.65) / (100 / rh[ok] - 1)
  dry <- raw[ok] / growth
  fit <- stats::lm(reference[ok] ~ dry)
  cf <- unname(stats::coef(fit))
  return(newSensorCal(
    "humidity",
    c(intercept = cf[1], slope = cf[2], kappa = kappa),
    sum(ok)
  ))
}

#' Apply a calibration to raw readings
#'
#' @param cal An object of class `sensor_cal`.
#' @param raw A numeric vector of raw sensor readings.
#' @param rh A numeric vector of relative humidity in percent; required for
#'   humidity calibrations.
#' @return A numeric vector of calibrated readings.
#' @export
applyCal <- function(cal, raw, rh = NULL) {
  if (!inherits(cal, "sensor_cal")) {
    stop("`cal` must be a `sensor_cal` object.", call. = FALSE)
  }
  cf <- cal$coef
  out <- switch(cal$type,
    linear = ,
    two_point = cf[["intercept"]] + cf[["slope"]] * raw,
    humidity = {
      if (is.null(rh)) {
        stop("`rh` is required for humidity calibrations.", call. = FALSE)
      }
      rh <- pmin(rh, 98)
      growth <- 1 + (cf[["kappa"]] / 1.65) / (100 / rh - 1)
      cf[["intercept"]] + cf[["slope"]] * raw / growth
    },
    stop("Unknown calibration type: ", cal$type, call. = FALSE)
  )
  return(pmax(out, 0))
}

#' Linear drift correction between two calibration checks
#'
#' Interpolates the offset measured at two check times and subtracts it from
#' every reading in between.
#'
#' @param time A `POSIXct` vector of reading times.
#' @param value A numeric vector of readings matching `time`.
#' @param checkTimes A `POSIXct` vector of length two giving check times.
#' @param checkOffsets A numeric vector of length two giving the measured
#'   offset (sensor minus reference) at each check.
#' @return A numeric vector of drift-corrected readings.
#' @export
correctDrift <- function(time, value, checkTimes, checkOffsets) {
  if (length(checkTimes) != 2 || length(checkOffsets) != 2) {
    stop("Exactly two check times and offsets are required.", call. = FALSE)
  }
  t0 <- as.numeric(checkTimes[1])
  t1 <- as.numeric(checkTimes[2])
  frac <- (as.numeric(time) - t0) / (t1 - t0)
  frac <- pmin(pmax(frac, 0), 1)
  offset <- checkOffsets[1] + frac * (checkOffsets[2] - checkOffsets[1])
  return(value - offset)
}

#' Agreement metrics between sensor and reference
#'
#' @param estimate A numeric vector of calibrated sensor readings.
#' @param reference A numeric vector of collocated reference readings.
#' @return A named numeric vector with `n`, `bias`, `mae`, `rmse`, `nrmse`
#'   and `r2`.
#' @export
calMetrics <- function(estimate, reference) {
  ok <- !is.na(estimate) & !is.na(reference)
  e <- estimate[ok]
  r <- reference[ok]
  if (length(e) < 2) {
    return(c(n = length(e), bias = NA, mae = NA, rmse = NA, nrmse = NA, r2 = NA))
  }
  err <- e - r
  rmse <- sqrt(mean(err^2))
  return(c(
    n = length(e),
    bias = mean(err),
    mae = mean(abs(err)),
    rmse = rmse,
    nrmse = rmse / mean(r),
    r2 = stats::cor(e, r)^2
  ))
}

#' Print a sensor calibration
#'
#' @param x An object of class `sensor_cal`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.sensor_cal <- function(x, ...) {
  cat("<sensor_cal> type:", x$type, "| n =", x$nObs, "\n")
  for (nm in names(x$coef)) {
    cat(sprintf("  %-10s %10.4f\n", nm, x$coef[[nm]]))
  }
  return(invisible(x))
}

#' Cross-validate calibration models
#'
#' Splits collocated data into contiguous folds, fits each calibration type
#' on the remaining folds and scores it on the held-out fold.
#'
#' @param raw A numeric vector of raw sensor readings.
#' @param reference A numeric vector of collocated reference readings.
#' @param rh A numeric vector of relative humidity in percent.
#' @param folds Number of contiguous folds.
#' @param types Calibration types to compare.
#' @return A data frame with one row per type and fold and columns `type`,
#'   `fold`, `n`, `bias`, `mae`, `rmse`, `nrmse` and `r2`.
#' @export
crossValidateCal <- function(raw, reference, rh = NULL, folds = 5L,
                             types = c("linear", "humidity")) {
  types <- match.arg(types, several.ok = TRUE)
  if ("humidity" %in% types && is.null(rh)) {
    stop("`rh` is required for humidity calibrations.", call. = FALSE)
  }
  n <- length(raw)
  foldId <- cut(seq_len(n), breaks = folds, labels = FALSE)
  rows <- list()
  for (type in types) {
    for (k in seq_len(folds)) {
      train <- foldId != k
      test <- foldId == k
      cal <- tryCatch(
        switch(type,
          linear = fitLinearCal(raw[train], reference[train]),
          humidity = fitHumidityCal(raw[train], reference[train], rh[train])
        ),
        error = function(e) {
          warning(sprintf("Fold %d (%s) failed: %s", k, type, conditionMessage(e)),
                  call. = FALSE)
          NULL
        }
      )
      if (is.null(cal)) {
        next
      }
      est <- applyCal(cal, raw[test], rh = if (is.null(rh)) NULL else rh[test])
      m <- calMetrics(est, reference[test])
      rows[[length(rows) + 1L]] <- data.frame(type = type, fold = k, t(m))
    }
  }
  if (length(rows) == 0) {
    stop("All folds failed to fit.", call. = FALSE)
  }
  return(do.call(rbind, rows))
}
