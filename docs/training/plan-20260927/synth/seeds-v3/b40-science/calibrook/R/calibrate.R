#' Default calibration coefficients
#'
#' @description
#' Identity calibrations for every sensor the package knows about. Each
#' element is a named numeric vector with `offset`, `gain` and `curve`
#' (a quadratic term).
#'
#' @return A named list with elements `temp`, `rh`, `pressure`, `co2` and
#'   `pm25`.
#' @keywords internal
defaultCoefficients = function() {
  identity = c(offset = 0, gain = 1, curve = 0)
  return(list(
    temp = identity,
    rh = identity,
    pressure = identity,
    co2 = identity,
    pm25 = identity
  ))
}

#' Validate a calibration coefficient list
#'
#' @description
#' Every coefficient vector must hold single finite `offset`, `gain` and
#' `curve` values, and `gain` must not be zero. The terms are checked in
#' that order.
#'
#' @param coefs A named list of coefficient vectors, one per sensor.
#' @return `coefs`, invisibly.
#' @keywords internal
checkCoefficients = function(coefs) {
  if (!is.list(coefs) || is.null(names(coefs))) {
    stop("`coefs` must be a named list.", call. = FALSE)
  }
  for (sensor in names(coefs)) {
    cf = coefs[[sensor]]
    missingTerms = setdiff(c("offset", "gain", "curve"), names(cf))
    if (length(missingTerms) > 0L) {
      stop(sprintf("Coefficients for `%s` lack: %s.", sensor,
                   paste(missingTerms, collapse = ", ")), call. = FALSE)
    }

    # Check `offset`
    if (length(cf[["offset"]]) != 1L || !is.finite(cf[["offset"]])) {
      stop(sprintf("`offset` for `%s` must be a single finite number.", sensor), call. = FALSE)
    }

    # Check `gain`
    if (length(cf[["gain"]]) != 1L || !is.finite(cf[["gain"]])) {
      stop(sprintf("`gain` for `%s` must be a single finite number.", sensor), call. = FALSE)
    }
    if (cf[["gain"]] == 0) {
      stop(sprintf("`gain` for `%s` must not be zero.", sensor), call. = FALSE)
    }

    # Check `curve`
    if (length(cf[["curve"]]) != 1L || !is.finite(cf[["curve"]])) {
      stop(sprintf("`curve` for `%s` must be a single finite number.", sensor), call. = FALSE)
    }
  }
  return(invisible(coefs))
}

#' Apply per-sensor calibrations to raw readings
#'
#' @description
#' Each sensor column is corrected as `offset + gain * x + curve * x^2`
#' using its own coefficients. Relative humidity is clipped to 0-100 and
#' concentrations to non-negative values after correction.
#'
#' @param raw A data frame of raw readings with one column per sensor.
#' @param coefs A named list of coefficient vectors, one per sensor;
#'   sensors without an entry use [defaultCoefficients()].
#' @param sensors Sensor columns to calibrate: any of `"temp"`, `"rh"`,
#'   `"pressure"`, `"co2"` and `"pm25"`.
#' @return `raw` with the calibrated columns replaced and an attribute
#'   `calibrated` listing them.
#' @export
calibrateSensors = function(raw, coefs = list(),
                            sensors = c("temp", "rh", "pressure", "co2", "pm25")) {
  sensors = match.arg(sensors, several.ok = TRUE)
  coefs = utils::modifyList(defaultCoefficients(), coefs)
  checkCoefficients(coefs)
  done = character()

  # Calibrate `temp`
  if ("temp" %in% sensors && "temp" %in% names(raw)) {
    cf = coefs$temp
    x = raw$temp
    raw$temp = cf[["offset"]] + cf[["gain"]] * x + cf[["curve"]] * x^2
    done = c(done, "temp")
  }

  # Calibrate `rh`
  if ("rh" %in% sensors && "rh" %in% names(raw)) {
    cf = coefs$rh
    x = raw$rh
    raw$rh = cf[["offset"]] + cf[["gain"]] * x + cf[["curve"]] * x^2
    raw$rh = pmin(pmax(raw$rh, 0), 100)
    done = c(done, "rh")
  }

  # Calibrate `pressure`
  if ("pressure" %in% sensors && "pressure" %in% names(raw)) {
    cf = coefs$pressure
    x = raw$pressure
    raw$pressure = cf[["offset"]] + cf[["gain"]] * x + cf[["curve"]] * x^2
    done = c(done, "pressure")
  }

  # Calibrate `co2`
  if ("co2" %in% sensors && "co2" %in% names(raw)) {
    cf = coefs$co2
    x = raw$co2
    raw$co2 = cf[["offset"]] + cf[["gain"]] * x + cf[["curve"]] * x^2
    raw$co2 = pmax(raw$co2, 0)
    done = c(done, "co2")
  }

  # Calibrate `pm25`
  if ("pm25" %in% sensors && "pm25" %in% names(raw)) {
    cf = coefs$pm25
    x = raw$pm25
    raw$pm25 = cf[["offset"]] + cf[["gain"]] * x + cf[["curve"]] * x^2
    raw$pm25 = pmax(raw$pm25, 0)
    done = c(done, "pm25")
  }

  attr(raw, "calibrated") = done
  return(raw)
}

#' Fit a calibration against reference readings
#'
#' @description
#' Fits `reference ~ observed` with one of four models and returns the
#' coefficients in the `offset`/`gain`/`curve` form used by
#' [calibrateSensors()].
#'
#' @param reference A numeric vector of reference readings.
#' @param observed A numeric vector of sensor readings, same length.
#' @param model `"offset"` (shift only), `"ratio"` (gain only, through the
#'   origin), `"linear"` (offset and gain) or `"quadratic"`.
#' @return A list with elements `model`, `coefs` and `rmse`.
#' @export
fitCalibration = function(reference, observed,
                          model = c("linear", "offset", "ratio", "quadratic")) {
  model = match.arg(model)
  ok = is.finite(reference) & is.finite(observed)
  ref = reference[ok]
  obs = observed[ok]
  if (length(ref) < 3L) {
    stop("At least 3 complete pairs are needed to fit a calibration.", call. = FALSE)
  }

  coefs = switch(model,
    offset = {
      shift = mean(ref - obs)
      c(offset = shift, gain = 1, curve = 0)
    },
    ratio = {
      fit = stats::lm(ref ~ 0 + obs)
      c(offset = 0, gain = unname(stats::coef(fit)[1]), curve = 0)
    },
    linear = {
      fit = stats::lm(ref ~ obs)
      c(offset = unname(stats::coef(fit)[1]), gain = unname(stats::coef(fit)[2]), curve = 0)
    },
    quadratic = {
      fit = stats::lm(ref ~ obs + I(obs^2))
      cf = unname(stats::coef(fit))
      c(offset = cf[1], gain = cf[2], curve = cf[3])
    }
  )

  fitted = coefs[["offset"]] + coefs[["gain"]] * obs + coefs[["curve"]] * obs^2
  return(list(
    model = model,
    coefs = coefs,
    rmse = sqrt(mean((ref - fitted)^2))
  ))
}

#' Compare sensors with their reference instruments
#'
#' @description
#' Computes bias, root mean square error and correlation for each sensor
#' that has a matching `ref_` column, for example `temp` and `ref_temp`.
#'
#' @param data A data frame with sensor columns and `ref_<sensor>` columns.
#' @param sensors Sensor columns to compare.
#' @return A data frame with columns `sensor`, `n`, `bias`, `rmse` and
#'   `r`, one row per sensor that has a reference column.
#' @export
calibrationReport = function(data, sensors = c("temp", "rh", "pressure", "co2", "pm25")) {
  rows = vector("list", length(sensors))
  names(rows) = sensors
  for (sensor in sensors) {
    refCol = paste0("ref_", sensor)
    if (!all(c(sensor, refCol) %in% names(data))) {
      next
    }
    obs = data[[sensor]]
    ref = data[[refCol]]
    ok = is.finite(obs) & is.finite(ref)
    rows[[sensor]] = data.frame(
      sensor = sensor,
      n = sum(ok),
      bias = mean(obs[ok] - ref[ok]),
      rmse = sqrt(mean((obs[ok] - ref[ok])^2)),
      r = if (sum(ok) > 2L) stats::cor(obs[ok], ref[ok]) else NA_real_
    )
  }
  out = do.call(rbind, rows[!vapply(rows, is.null, logical(1))])
  rownames(out) = NULL
  return(out)
}
