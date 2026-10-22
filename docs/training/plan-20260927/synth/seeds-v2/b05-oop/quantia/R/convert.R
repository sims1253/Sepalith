.unitRegistry <- new.env(parent = emptyenv())

#' @title Register a unit
#' @description Adds a unit to the package registry so it can be referred to
#'   by symbol in [quantity()] and [convertTo()].
#' @param unit A [Unit-class] object.
#' @param overwrite Whether to replace an existing unit with the same symbol.
#' @return The unit, invisibly.
#' @export
registerUnit <- function(unit, overwrite = FALSE) {
  if (!is(unit, "Unit")) {
    stop("`unit` must be a Unit object.", call. = FALSE)
  }
  validObject(unit)

  sym <- unit@symbol
  if (!overwrite && exists(sym, envir = .unitRegistry, inherits = FALSE)) {
    stop("Unit '", sym, "' is already registered; use `overwrite = TRUE`.", call. = FALSE)
  }
  assign(sym, unit, envir = .unitRegistry)
  invisible(unit)
}

#' @title Look up a registered unit
#' @param symbol The symbol of a registered unit.
#' @return A [Unit-class] object.
#' @export
lookupUnit <- function(symbol) {
  if (!is.character(symbol) || length(symbol) != 1L || is.na(symbol)) {
    stop("`symbol` must be a single string.", call. = FALSE)
  }
  unit <- get0(symbol, envir = .unitRegistry, inherits = FALSE)
  if (is.null(unit)) {
    stop("Unknown unit '", symbol, "'. See `listUnits()`.", call. = FALSE)
  }
  unit
}

#' @title Remove a registered unit
#' @param symbol The symbol of a registered unit.
#' @return `TRUE` if a unit was removed, `FALSE` otherwise, invisibly.
#' @export
unregisterUnit <- function(symbol) {
  found <- exists(symbol, envir = .unitRegistry, inherits = FALSE)
  if (found) {
    rm(list = symbol, envir = .unitRegistry)
  }
  invisible(found)
}

#' @title List registered units
#' @param dimension Optional base dimension; only units whose dimension is
#'   exactly that base dimension are listed.
#' @return A data frame with columns `symbol`, `dims`, `scale` and `offset`.
#' @export
listUnits <- function(dimension = NULL) {
  syms <- sort(ls(.unitRegistry))
  units <- lapply(syms, lookupUnit)

  # "length" means exactly L^1, so km is listed but m^2 is not
  if (!is.null(dimension)) {
    target <- makeDims(stats::setNames(1, dimension))
    keep <- vapply(units, function(u) isTRUE(all.equal(u@dims, target)), logical(1))
    units <- units[keep]
  }

  data.frame(
    symbol = vapply(units, function(u) u@symbol, character(1)),
    dims = vapply(units, formatDims, character(1)),
    scale = vapply(units, function(u) u@scale, numeric(1)),
    offset = vapply(units, function(u) u@offset, numeric(1)),
    stringsAsFactors = FALSE
  )
}

#' @title Register the built-in units
#' @description Called when the package is loaded. Populates the registry
#'   with common SI and imperial units.
#' @return The number of units registered, invisibly.
#' @keywords internal
registerDefaultUnits <- function() {
  defaults <- list(
    unit("1"),
    # length
    unit("m", c(length = 1)),
    unit("km", c(length = 1), scale = 1000),
    unit("cm", c(length = 1), scale = 0.01),
    unit("mi", c(length = 1), scale = 1609.344),
    # mass
    unit("kg", c(mass = 1)),
    unit("g", c(mass = 1), scale = 0.001),
    unit("lb", c(mass = 1), scale = 0.45359237),
    # time
    unit("s", c(time = 1)),
    unit("min", c(time = 1), scale = 60),
    unit("h", c(time = 1), scale = 3600),
    # temperature and derived units
    unit("K", c(temperature = 1)),
    unit("degC", c(temperature = 1), offset = 273.15),
    unit("N", c(mass = 1, length = 1, time = -2)),
    unit("J", c(mass = 1, length = 2, time = -2))
  )

  # overwrite = TRUE so that reloading the package is harmless
  for (u in defaults) {
    registerUnit(u, overwrite = TRUE)
  }
  invisible(length(defaults))
}

#' @title Convert a quantity to another unit
#' @param x A [Quantity-class] object.
#' @param to The target unit, as a [Unit-class] object or a registered symbol.
#' @param ... Additional arguments for methods.
#' @return A [Quantity-class] object expressed in `to`.
#' @export
#' @examples
#' convertTo(quantity(5, "km"), "mi")
setGeneric("convertTo", function(x, to, ...) standardGeneric("convertTo"))

#' @rdname convertTo
setMethod("convertTo", signature("Quantity", "Unit"), function(x, to, ...) {
  from <- x@unit
  if (!sameDims(from, to)) {
    stop("Can't convert ", from@symbol, " [", formatDims(from), "] to ",
      to@symbol, " [", formatDims(to), "].", call. = FALSE)
  }

  # Go through SI so that offsets (degC) are applied in the right order
  si <- x@.Data * from@scale + from@offset
  new("Quantity", (si - to@offset) / to@scale, unit = to)
})

#' @rdname convertTo
setMethod("convertTo", signature("Quantity", "character"), function(x, to, ...) {
  convertTo(x, lookupUnit(to), ...)
})

#' Package load hook
#'
#' @param libname,pkgname Library path and package name, supplied by R.
#' @noRd
.onLoad <- function(libname, pkgname) {
  registerDefaultUnits()
  invisible(NULL)
}
