#' Multiplicative unit factors relative to a base unit
#'
#' Internal lookup table of factors that convert one unit of each quantity
#' into the quantity's base unit.
#'
#' @param quantity Name of the physical quantity.
#' @return A named numeric vector of conversion factors.
#' @noRd
unit_factors <- function(quantity) {
  switch(quantity,
    speed = c(m_s = 1, km_h = 1 / 3.6, knot = 0.514444, mph = 0.44704),
    pressure = c(hPa = 1, kPa = 10, inHg = 33.8639, mmHg = 1.333224),
    precipitation = c(mm = 1, cm = 10, inch = 25.4),
    discharge = c(m3_s = 1, L_s = 1e-3, cfs = 0.0283168, ML_d = 1e3 / 86400),
    stop("Unknown quantity: ", quantity, call. = FALSE)
  )
}

#' Convert between linear units of one quantity
#'
#' Shared implementation for all factor-based conversions.
#'
#' @param x A numeric vector of values to convert.
#' @param from,to Unit names valid for `quantity`.
#' @param quantity Name of the physical quantity.
#' @return A numeric vector of converted values.
#' @noRd
convert_linear <- function(x, from, to, quantity) {
  factors <- unit_factors(quantity)
  bad <- setdiff(c(from, to), names(factors))
  if (length(bad) > 0) {
    stop(
      sprintf("Unknown %s unit(s): %s", quantity, paste(bad, collapse = ", ")),
      call. = FALSE
    )
  }
  x * factors[[from]] / factors[[to]]
}

#' Convert wind speed units
#'
#' @param x A numeric vector of values to convert.
#' @param from,to Units, any of `"m_s"`, `"km_h"`, `"knot"` and `"mph"`.
#' @return A numeric vector of converted values.
#' @family unit converters
#' @export
convert_speed <- function(x, from = "m_s", to = "km_h") {
  convert_linear(x, from = from, to = to, quantity = "speed")
}

#' Convert atmospheric pressure units
#'
#' @param x A numeric vector of values to convert.
#' @param from,to Units, any of `"hPa"`, `"kPa"`, `"inHg"` and `"mmHg"`.
#' @return A numeric vector of converted values.
#' @family unit converters
#' @export
convert_pressure <- function(x, from = "hPa", to = "kPa") {
  convert_linear(x, from = from, to = to, quantity = "pressure")
}

#' Convert precipitation depth units
#'
#' @param x A numeric vector of values to convert.
#' @param from,to Units, any of `"mm"`, `"cm"` and `"inch"`.
#' @return A numeric vector of converted values.
#' @family unit converters
#' @export
convert_precipitation <- function(x, from = "inch", to = "mm") {
  convert_linear(x, from = from, to = to, quantity = "precipitation")
}

#' Convert discharge units
#'
#' @param x A numeric vector of values to convert.
#' @param from,to Units, any of `"m3_s"`, `"L_s"`, `"cfs"` and `"ML_d"`.
#' @return A numeric vector of converted values.
#' @family unit converters
#' @export
convert_discharge <- function(x, from = "cfs", to = "m3_s") {
  convert_linear(x, from = from, to = to, quantity = "discharge")
}

#' Convert temperature units
#'
#' Temperature scales have offsets, so values are routed through kelvin
#' rather than through a single multiplicative factor.
#'
#' @param x A numeric vector of values to convert.
#' @param from,to Units, any of `"C"`, `"F"` and `"K"`.
#' @return A numeric vector of converted values.
#' @family unit converters
#' @export
convert_temperature <- function(x, from = c("F", "C", "K"), to = c("C", "F", "K")) {
  from <- match.arg(from)
  to <- match.arg(to)
  kelvin <- switch(from,
    C = x + 273.15,
    F = (x - 32) * 5 / 9 + 273.15,
    K = x
  )
  if (any(kelvin < 0, na.rm = TRUE)) {
    warning("Values below absolute zero set to NA.", call. = FALSE)
    kelvin[!is.na(kelvin) & kelvin < 0] <- NA_real_
  }
  switch(to,
    C = kelvin - 273.15,
    F = (kelvin - 273.15) * 9 / 5 + 32,
    K = kelvin
  )
}

#' Convert several columns of a station table at once
#'
#' Applies a conversion specification to a data frame, converting each named
#' column and recording the new unit in a `units` attribute.
#'
#' @param data A data frame of station observations.
#' @param spec A named list; each element is a character vector
#'   `c(quantity, from, to)` for the column of the same name.
#' @return `data` with converted columns and an updated `units` attribute.
#' @export
convert_columns <- function(data, spec) {
  units <- attr(data, "units")
  if (is.null(units)) {
    units <- list()
  }
  for (col in names(spec)) {
    if (!col %in% names(data)) {
      warning("Column `", col, "` not found; skipped.", call. = FALSE)
      next
    }
    rule <- spec[[col]]
    converter <- switch(rule[[1]],
      speed = convert_speed,
      pressure = convert_pressure,
      precipitation = convert_precipitation,
      discharge = convert_discharge,
      temperature = convert_temperature,
      stop("Unknown quantity for `", col, "`: ", rule[[1]], call. = FALSE)
    )
    data[[col]] <- converter(data[[col]], from = rule[[2]], to = rule[[3]])
    units[[col]] <- rule[[3]]
  }
  attr(data, "units") <- units
  data
}

#' Harmonise units across stations using metadata
#'
#' Looks up each station's reporting units in a metadata table and converts
#' its rows to a common target unit per variable, so that data from mixed
#' networks can be combined.
#'
#' @param data A data frame of station observations.
#' @param metadata A data frame with columns `station`, `variable`,
#'   `quantity` and `unit`.
#' @param target A named character vector giving the target unit for each
#'   variable.
#' @param station_col Name of the station identifier column in `data`.
#' @return `data` with converted values and a `units` attribute set to
#'   `target`.
#' @export
harmonise_units <- function(data, metadata, target, station_col = "station") {
  needed <- c("station", "variable", "quantity", "unit")
  if (!all(needed %in% names(metadata))) {
    stop("`metadata` must have columns: ", paste(needed, collapse = ", "), call. = FALSE)
  }
  stations <- unique(data[[station_col]])
  for (st in stations) {
    rows <- data[[station_col]] == st
    meta <- metadata[metadata$station == st, , drop = FALSE]
    for (v in names(target)) {
      if (!v %in% names(data)) {
        next
      }
      m <- meta[meta$variable == v, , drop = FALSE]
      if (nrow(m) == 0) {
        warning("No unit metadata for ", st, "/", v, "; values left as-is.", call. = FALSE)
        next
      }
      if (m$unit[1] == target[[v]]) {
        next
      }
      converter <- switch(m$quantity[1],
        speed = convert_speed,
        pressure = convert_pressure,
        precipitation = convert_precipitation,
        discharge = convert_discharge,
        temperature = convert_temperature,
        stop("Unknown quantity: ", m$quantity[1], call. = FALSE)
      )
      data[[v]][rows] <- converter(data[[v]][rows], from = m$unit[1], to = target[[v]])
    }
  }
  attr(data, "units") <- as.list(target)
  data
}
