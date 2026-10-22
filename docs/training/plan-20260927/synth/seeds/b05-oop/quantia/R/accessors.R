#' Get the magnitudes of a quantity
#'
#' Returns the bare numeric values of a quantity, dropping its unit.
#'
#' @param object A [Quantity-class] object.
#' @param ... Additional arguments for methods.
#' @return A numeric vector.
#' @export
setGeneric("magnitude", function(object, ...) standardGeneric("magnitude"))

#' Set the magnitudes of a quantity
#'
#' @param object A [Quantity-class] object.
#' @param value A numeric vector of the same length as `object`.
#' @return The modified object.
#' @export
setGeneric("magnitude<-", function(object, value) standardGeneric("magnitude<-"))

#' Get the unit of a quantity
#'
#' @param object A [Quantity-class] object.
#' @param ... Additional arguments for methods.
#' @return A [Unit-class] object.
#' @export
setGeneric("unitOf", function(object, ...) standardGeneric("unitOf"))

#' Set the unit of a quantity without converting
#'
#' Replaces the unit but keeps the magnitudes unchanged. Use [convertTo()]
#' to change units while preserving the physical value.
#'
#' @param object A [Quantity-class] object.
#' @param value A [Unit-class] object or a registered unit symbol.
#' @return The modified object.
#' @export
setGeneric("unitOf<-", function(object, value) standardGeneric("unitOf<-"))

#' @rdname magnitude
setMethod("magnitude", "Quantity", function(object, ...) {
  object@.Data
})

#' @rdname magnitude-set
setReplaceMethod("magnitude", "Quantity", function(object, value) {
  if (!is.numeric(value) || length(value) != length(object)) {
    stop("`value` must be numeric with length ", length(object), ".", call. = FALSE)
  }
  object@.Data <- as.numeric(value)
  validObject(object)
  object
})

#' @rdname unitOf
setMethod("unitOf", "Quantity", function(object, ...) {
  object@unit
})

#' @rdname unitOf-set
setReplaceMethod("unitOf", "Quantity", function(object, value) {
  if (is.character(value)) {
    value <- lookupUnit(value)
  }
  object@unit <- value
  validObject(object)
  object
})

#' Format a unit's dimensions
#'
#' Writes the non-zero exponents of a unit as a compact string such as
#' `"L M T^-2"`.
#'
#' @param unit A [Unit-class] object.
#' @return A single string, `"1"` for dimensionless units.
#' @keywords internal
formatDims <- function(unit) {
  abbrev <- c("L", "M", "T", "I", "K", "N", "J")
  powers <- unit@dims
  used <- which(powers != 0)
  if (length(used) == 0L) {
    return("1")
  }
  parts <- ifelse(powers[used] == 1, abbrev[used], paste0(abbrev[used], "^", powers[used]))
  paste(parts, collapse = " ")
}

#' @rdname Unit-class
#' @param object A [Unit-class] object.
setMethod("show", "Unit", function(object) {
  cat("<Unit> ", object@symbol, " [", formatDims(object), "]", sep = "")
  if (object@scale != 1) {
    cat(" x ", format(object@scale, digits = 6), sep = "")
  }
  if (object@offset != 0) {
    cat(" + ", format(object@offset, digits = 6), sep = "")
  }
  cat("\n")
  invisible(object)
})

#' @rdname Quantity-class
#' @param object A [Quantity-class] object.
setMethod("show", "Quantity", function(object) {
  n <- length(object)
  cat("<Quantity> ", n, " value(s) in ", object@unit@symbol, "\n", sep = "")
  if (n == 0L) {
    return(invisible(object))
  }
  shown <- utils::head(object@.Data, 10L)
  print(paste(format(shown, digits = 4), object@unit@symbol), quote = FALSE)
  if (n > 10L) {
    cat("... ", n - 10L, " more\n", sep = "")
  }
  invisible(object)
})

#' @rdname Quantity-class
#' @param x A [Quantity-class] object.
#' @param i Indices of the elements to extract.
#' @param ... Ignored.
#' @param drop Ignored.
setMethod("[", "Quantity", function(x, i, ..., drop = TRUE) {
  new("Quantity", x@.Data[i], unit = x@unit)
})
