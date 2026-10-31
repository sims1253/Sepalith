#' Saturation vapour pressure over water
#'
#' @param temp Air temperature in degrees Celsius.
#' @param method Approximation to use: `"magnus"`, `"tetens"`, `"bolton"`
#'   or `"buck"`. All share the form `a * exp(b * temp / (temp + c))`,
#'   except `"buck"`, which adds a temperature term to `b`.
#' @return Saturation vapour pressure in hPa.
#' @export
saturation_vapour_pressure <- function(temp, method = c("magnus", "tetens", "bolton", "buck")) {
  method <- match.arg(method)
  switch(method,
    magnus = {
      a <- 6.1094
      b <- 17.625
      c <- 243.04
      a * exp(b * temp / (temp + c))
    },
    tetens = {
      a <- 6.1078
      b <- 17.27
      c <- 237.3
      a * exp(b * temp / (temp + c))
    },
    bolton = {
      a <- 6.112
      b <- 17.67
      c <- 243.5
      a * exp(b * temp / (temp + c))
    },
    buck = {
      a <- 6.1121
      b <- 18.678 - temp / 234.5
      c <- 257.14
      a * exp(b * temp / (temp + c))
    }
  )
}

#' Split wind into vector components
#'
#' @param speed Wind speed.
#' @param direction Meteorological direction the wind blows from, degrees.
#' @return A list with elements `u` (towards east) and `v` (towards north).
#' @export
wind_to_components <- function(speed, direction) {
  rad <- direction * pi / 180
  u <- -speed * sin(rad)
  v <- -speed * cos(rad)
  list(u = zapsmall(u), v = zapsmall(v))
}

#' Join vector components into wind speed and direction
#'
#' @param u Eastward component.
#' @param v Northward component.
#' @return A list with elements `speed` and `direction` (degrees the wind
#'   blows from, in `[0, 360)`).
#' @export
components_to_wind <- function(u, v) {
  speed <- sqrt(u^2 + v^2)
  direction <- (atan2(-u, -v) * 180 / pi) %% 360
  list(speed = zapsmall(speed), direction = zapsmall(direction))
}

#' Check the physical range of meteorological columns
#'
#' @param data A data frame of meteorological observations.
#' @return A data frame with one row per checked column and columns
#'   `column`, `n_low`, `n_high` and `n_missing`. Columns that are absent
#'   from `data` are skipped.
#' @keywords internal
met_range_report <- function(data) {
  limits <- list(
    temp = c(-90, 60),
    rh = c(0, 100),
    wind_speed = c(0, 75),
    wind_dir = c(0, 360),
    pressure = c(800, 1100)
  )
  rows <- list()

  # Range of `temp`
  if ("temp" %in% names(data)) {
    x <- data[["temp"]]
    rows$temp <- data.frame(column = "temp",
      n_low = sum(x < limits$temp[1], na.rm = TRUE),
      n_high = sum(x > limits$temp[2], na.rm = TRUE),
      n_missing = sum(is.na(x)))
  }

  # Range of `rh`
  if ("rh" %in% names(data)) {
    x <- data[["rh"]]
    rows$rh <- data.frame(column = "rh",
      n_low = sum(x < limits$rh[1], na.rm = TRUE),
      n_high = sum(x > limits$rh[2], na.rm = TRUE),
      n_missing = sum(is.na(x)))
  }

  # Range of `wind_speed`
  if ("wind_speed" %in% names(data)) {
    x <- data[["wind_speed"]]
    rows$wind_speed <- data.frame(column = "wind_speed",
      n_low = sum(x < limits$wind_speed[1], na.rm = TRUE),
      n_high = sum(x > limits$wind_speed[2], na.rm = TRUE),
      n_missing = sum(is.na(x)))
  }

  # Range of `wind_dir`
  if ("wind_dir" %in% names(data)) {
    x <- data[["wind_dir"]]
    rows$wind_dir <- data.frame(column = "wind_dir",
      n_low = sum(x < limits$wind_dir[1], na.rm = TRUE),
      n_high = sum(x > limits$wind_dir[2], na.rm = TRUE),
      n_missing = sum(is.na(x)))
  }

  # Range of `pressure`
  if ("pressure" %in% names(data)) {
    x <- data[["pressure"]]
    rows$pressure <- data.frame(column = "pressure",
      n_low = sum(x < limits$pressure[1], na.rm = TRUE),
      n_high = sum(x > limits$pressure[2], na.rm = TRUE),
      n_missing = sum(is.na(x)))
  }

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' Add derived meteorological variables
#'
#' Expects columns `temp` (degrees Celsius), `rh` (percent), `wind_speed`
#' (m/s) and `wind_dir` (degrees), depending on which variables are asked
#' for.
#'
#' @param data A data frame of meteorological observations.
#' @param vars Variables to add, any of `"vapour_pressure"`, `"dew_point"`,
#'   `"abs_humidity"`, `"u"` and `"v"`.
#' @param method Passed to [saturation_vapour_pressure()].
#' @return `data` with one new column per element of `vars`.
#' @export
derive_met <- function(data, vars = c("vapour_pressure", "dew_point", "abs_humidity", "u", "v"),
                       method = "magnus") {
  vars <- match.arg(vars, several.ok = TRUE)
  needs <- list(
    vapour_pressure = c("temp", "rh"),
    dew_point = c("temp", "rh"),
    abs_humidity = c("temp", "rh"),
    u = c("wind_speed", "wind_dir"),
    v = c("wind_speed", "wind_dir")
  )
  missing_cols <- setdiff(unlist(needs[vars]), names(data))
  if (length(missing_cols) > 0L) {
    stop(sprintf("`data` lacks columns: %s.", paste(missing_cols, collapse = ", ")),
         call. = FALSE)
  }

  for (var in vars) {
    data[[var]] <- switch(var,
      vapour_pressure = {
        saturation_vapour_pressure(data$temp, method) * data$rh / 100
      },
      dew_point = {
        e <- saturation_vapour_pressure(data$temp, "magnus") * data$rh / 100
        243.04 * log(e / 6.1094) / (17.625 - log(e / 6.1094))
      },
      abs_humidity = {
        e <- saturation_vapour_pressure(data$temp, method) * data$rh / 100
        100 * e / (461.5 * (data$temp + 273.15)) * 1000
      },
      u = wind_to_components(data$wind_speed, data$wind_dir)$u,
      v = wind_to_components(data$wind_speed, data$wind_dir)$v
    )
  }
  data
}
