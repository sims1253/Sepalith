#' @title Base dimension names
#' @description The seven SI base dimensions, in the order used for every
#'   dimension vector stored in a [Unit-class] object.
#' @keywords internal
baseDimensions <- function() {
  c("length", "mass", "time", "current", "temperature", "amount", "luminosity")
}

#' @title Build a full dimension vector
#' @description Expands a partial named vector of exponents to the full set
#'   of base dimensions, filling unspecified entries with zero.
#' @param dims A named numeric vector of exponents, e.g. `c(length = 1, time = -1)`.
#' @return A named numeric vector of length seven.
#' @keywords internal
makeDims <- function(dims = numeric()) {
  out <- stats::setNames(numeric(length(baseDimensions())), baseDimensions())
  if (length(dims) == 0L) {
    return(out)
  }

  # Unnamed exponents would otherwise be dropped without a word
  if (is.null(names(dims)) || any(names(dims) == "")) {
    stop("Every exponent in `dims` must be named by a base dimension.", call. = FALSE)
  }
  unknown <- setdiff(names(dims), baseDimensions())
  if (length(unknown) > 0L) {
    stop("Unknown dimension(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }

  out[names(dims)] <- dims
  out
}

#' @title A unit of measure
#' @description A `Unit` couples a printable symbol with a dimension vector
#'   and a scale factor relative to the coherent SI unit of the same
#'   dimension.
#' @slot symbol The printed symbol, such as `"km"`.
#' @slot dims Named numeric exponents of the base dimensions.
#' @slot scale Multiplier converting one of this unit to SI.
#' @slot offset Additive offset to SI, used for temperature scales.
#' @export
setClass(
  "Unit",
  representation(
    symbol = "character",
    dims = "numeric",
    scale = "numeric",
    offset = "numeric"
  ),
  prototype(
    symbol = "1",
    dims = makeDims(),
    scale = 1,
    offset = 0
  )
)

setValidity("Unit", function(object) {
  # Collect every problem so the user sees them all at once
  problems <- character()
  if (length(object@symbol) != 1L || is.na(object@symbol)) {
    problems <- c(problems, "`symbol` must be a single non-missing string")
  }
  if (!identical(names(object@dims), baseDimensions())) {
    problems <- c(problems, "`dims` must be named by the seven base dimensions")
  }
  if (length(object@scale) != 1L || !is.finite(object@scale) || object@scale <= 0) {
    problems <- c(problems, "`scale` must be a single positive number")
  }
  if (length(object@offset) != 1L || !is.finite(object@offset)) {
    problems <- c(problems, "`offset` must be a single finite number")
  }

  if (length(problems) == 0L) TRUE else problems
})

#' @title A numeric quantity with a unit
#' @description A `Quantity` stores a numeric vector of magnitudes expressed
#'   in a single [Unit-class].
#' @slot .Data The numeric magnitudes.
#' @slot unit The unit the magnitudes are expressed in.
#' @export
setClass(
  "Quantity",
  contains = "numeric",
  representation(unit = "Unit")
)

setValidity("Quantity", function(object) {
  if (!validObject(object@unit, test = TRUE) %in% TRUE) {
    return("`unit` is not a valid Unit")
  }
  # Offsets only make sense on an absolute temperature scale (dimension 5)
  if (object@unit@offset != 0 && any(object@unit@dims[-5] != 0)) {
    return("offset units can only be used for pure temperatures")
  }
  TRUE
})

#' @title Create a unit
#' @param symbol The printed symbol, such as `"km"`.
#' @param dims A named numeric vector of exponents, e.g. `c(length = 1, time = -1)`.
#' @param scale Multiplier converting one of this unit to SI.
#' @param offset Additive offset to SI, used for temperature scales.
#' @return A [Unit-class] object.
#' @export
#' @examples
#' unit("km", c(length = 1), scale = 1000)
unit <- function(symbol, dims = numeric(), scale = 1, offset = 0) {
  new("Unit",
    symbol = as.character(symbol),
    dims = makeDims(dims),
    scale = as.numeric(scale),
    offset = as.numeric(offset)
  )
}

#' @title Create a quantity
#' @param x A numeric vector of magnitudes.
#' @param unit A [Unit-class] object, or the symbol of a registered unit.
#' @return A [Quantity-class] object.
#' @export
#' @examples
#' quantity(c(1.5, 2), "km")
quantity <- function(x, unit) {
  if (!is.numeric(x)) {
    stop("`x` must be numeric, not ", class(x)[[1L]], ".", call. = FALSE)
  }

  if (is.character(unit)) {
    unit <- lookupUnit(unit)
  }
  if (!is(unit, "Unit")) {
    stop("`unit` must be a Unit or a registered unit symbol.", call. = FALSE)
  }
  new("Quantity", as.numeric(x), unit = unit)
}

#' @title Test for a dimensionless unit
#' @param unit A [Unit-class] object.
#' @noRd
isDimensionless <- function(unit) {
  all(unit@dims == 0)
}
