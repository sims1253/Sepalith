#' Conversion factors for one physical quantity
#'
#' Returns a named vector that converts one unit of each listed unit into
#' the base unit of the quantity: metre, kilogram, litre, pascal, joule
#' and second.
#'
#' @param quantity One of `"length"`, `"mass"`, `"volume"`, `"pressure"`,
#'   `"energy"` or `"time"`.
#' @return A named numeric vector of factors to the base unit.
#' @keywords internal
unit_factors <- function(quantity = c("length", "mass", "volume", "pressure",
                                      "energy", "time")) {
  quantity <- match.arg(quantity)
  switch(quantity,
    length = c(
      mm = 1e-3, cm = 1e-2, m = 1, km = 1e3,
      inch = 0.0254, ft = 0.3048, mile = 1609.344
    ),
    mass = c(
      mg = 1e-6, g = 1e-3, kg = 1, t = 1e3,
      oz = 0.028349523125, lb = 0.45359237
    ),
    volume = c(
      ml = 1e-3, cl = 1e-2, l = 1, m3 = 1e3,
      gal = 3.785411784, qt = 3.785411784 / 4
    ),
    pressure = c(
      Pa = 1, hPa = 1e2, kPa = 1e3, bar = 1e5,
      atm = 101325, mmHg = 101325 / 760, psi = 6894.757
    ),
    energy = c(
      J = 1, kJ = 1e3, MJ = 1e6, Wh = 3600,
      kWh = 3600 * 1e3, cal = 4.184, kcal = 4.184 * 1e3
    ),
    time = c(
      s = 1, min = 60, h = 60 * 60, d = 24 * 60 * 60,
      wk = 7 * 24 * 60 * 60
    )
  )
}

#' Convert values between units of one quantity
#'
#' Looks up both units in the factor table of `quantity`, converts `x` to
#' the base unit and from there into `to`.
#'
#' @param x A numeric vector of values in unit `from`.
#' @param from A single unit name present in [unit_factors()] for `quantity`.
#' @param to A single unit name present in [unit_factors()] for `quantity`.
#' @param quantity The physical quantity, passed to [unit_factors()].
#' @return A numeric vector the same length as `x`, in unit `to`.
#' @export
convert_units <- function(x, from, to, quantity = "length") {
  factors <- unit_factors(quantity)
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }

  # Validate `from`
  if (!is.character(from) || length(from) != 1L) {
    stop("`from` must be a single unit name.", call. = FALSE)
  }
  if (!from %in% names(factors)) {
    stop(sprintf("Unknown %s unit `%s` in `from`.", quantity, from), call. = FALSE)
  }

  # Validate `to`
  if (!is.character(to) || length(to) != 1L) {
    stop("`to` must be a single unit name.", call. = FALSE)
  }
  if (!to %in% names(factors)) {
    stop(sprintf("Unknown %s unit `%s` in `to`.", quantity, to), call. = FALSE)
  }

  base <- x * factors[[from]]
  base / factors[[to]]
}

#' Convert temperatures between scales
#'
#' Temperatures are not proportional, so they cannot use a factor table.
#' Values are first brought to Celsius and then out to the target scale.
#'
#' @param x A numeric vector of temperatures.
#' @param from Scale of `x`: `"C"`, `"F"`, `"K"` or `"R"` (Rankine).
#' @param to Target scale, one of the same choices as `from`.
#' @return A numeric vector of temperatures on scale `to`.
#' @export
convert_temperature <- function(x, from = c("C", "F", "K", "R"),
                                to = c("C", "F", "K", "R")) {
  from <- match.arg(from)
  to <- match.arg(to)
  if (identical(from, to)) {
    return(x)
  }

  # Into Celsius
  celsius <- switch(from,
    C = x,
    F = (x - 32) * 5 / 9,
    K = x - 273.15,
    R = (x - 491.67) * 5 / 9
  )

  # Out of Celsius
  switch(to,
    C = celsius,
    F = celsius * 9 / 5 + 32,
    K = celsius + 273.15,
    R = celsius * 9 / 5 + 491.67
  )
}

#' Convert a rate between unit systems
#'
#' A rate is an amount per time. The amount is converted with the factor
#' table of its quantity and the denominator with the `time` table.
#'
#' @param x A numeric vector of rates.
#' @param from,to Units written as `"amount/time"`, for example `"l/s"`.
#' @param type The kind of rate: `"flow"` (volume per time), `"speed"`
#'   (length per time), `"discharge"` (mass per time) or `"power"`
#'   (energy per time).
#' @return A numeric vector of rates in unit `to`.
#' @export
convert_rate <- function(x, from, to,
                         type = c("flow", "speed", "discharge", "power")) {
  type <- match.arg(type)
  quantity <- switch(type,
    flow = "volume",
    speed = "length",
    discharge = "mass",
    power = "energy"
  )
  from_parts <- strsplit(from, "/", fixed = TRUE)[[1]]
  to_parts <- strsplit(to, "/", fixed = TRUE)[[1]]
  if (length(from_parts) != 2L) {
    stop(sprintf("`%s` is not a rate unit of the form amount/time.", from), call. = FALSE)
  }
  if (length(to_parts) != 2L) {
    stop(sprintf("`%s` is not a rate unit of the form amount/time.", to), call. = FALSE)
  }

  # Numerator, then denominator
  amount <- convert_units(x, from_parts[1], to_parts[1], quantity = quantity)
  per <- convert_units(1, from_parts[2], to_parts[2], quantity = "time")
  amount / per
}

#' Convert several data frame columns at once
#'
#' @param data A data frame of measurements.
#' @param units A named character vector of current units, one per column
#'   to convert, named by column.
#' @param target A named character vector of target units with the same
#'   names as `units`.
#' @param quantity A named character vector giving the quantity of each
#'   column, with the same names as `units`.
#' @return `data` with the listed columns converted and an attribute
#'   `units` recording the new unit of every converted column.
#' @export
convert_columns <- function(data, units, target, quantity) {
  stopifnot(is.data.frame(data))
  cols <- names(units)
  missing_cols <- setdiff(cols, names(data))
  if (length(missing_cols) > 0L) {
    stop(sprintf("Columns not found in `data`: %s.",
                 paste(missing_cols, collapse = ", ")), call. = FALSE)
  }
  missing_target <- setdiff(cols, names(target))
  if (length(missing_target) > 0L) {
    stop(sprintf("Columns without a unit in `target`: %s.",
                 paste(missing_target, collapse = ", ")), call. = FALSE)
  }
  missing_quantity <- setdiff(cols, names(quantity))
  if (length(missing_quantity) > 0L) {
    stop(sprintf("Columns without a unit in `quantity`: %s.",
                 paste(missing_quantity, collapse = ", ")), call. = FALSE)
  }

  for (col in cols) {
    if (identical(quantity[[col]], "temperature")) {
      data[[col]] <- convert_temperature(data[[col]], units[[col]], target[[col]])
    } else {
      data[[col]] <- convert_units(data[[col]], units[[col]], target[[col]],
                                   quantity = quantity[[col]])
    }
  }
  attr(data, "units") <- target[cols]
  data
}

#' Choose a readable unit for a vector of values
#'
#' Picks, among the units of `quantity`, the largest unit in which the
#' median absolute value of `x` is still at least one.
#'
#' @param x A numeric vector in the base unit of `quantity`.
#' @param quantity The physical quantity, passed to [unit_factors()].
#' @param candidates Optional subset of unit names to choose from.
#' @return A list with elements `unit`, `factor` and `value` (the
#'   converted vector).
#' @export
pretty_unit <- function(x, quantity = "length", candidates = NULL) {
  factors <- unit_factors(quantity)
  if (!is.null(candidates)) {
    factors <- factors[intersect(candidates, names(factors))]
  }
  factors <- sort(factors)
  typical <- stats::median(abs(x), na.rm = TRUE)
  fits <- factors[factors <= typical]
  unit <- if (length(fits) == 0L) names(factors)[1] else names(fits)[length(fits)]
  list(
    unit = unit,
    factor = factors[[unit]],
    value = x / factors[[unit]]
  )
}
