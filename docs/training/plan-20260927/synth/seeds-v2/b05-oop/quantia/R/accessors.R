#' @title Get or set the magnitudes of a quantity
#' @description `magnitude()` returns the bare numeric values of a quantity,
#'   dropping its unit; `magnitude<-` replaces them and keeps the unit.
#' @param object A [Quantity-class] object.
#' @param ... Additional arguments for methods.
#' @return A numeric vector, or the modified object for the replacement form.
#' @export
setGeneric("magnitude", function(object, ...) standardGeneric("magnitude"))

#' @rdname magnitude
#' @param value A numeric vector of the same length as `object`.
#' @export
setGeneric("magnitude<-", function(object, value) standardGeneric("magnitude<-"))

#' @title Get or set the unit of a quantity
#' @description `unitOf<-` replaces the unit but keeps the magnitudes
#'   unchanged. Use [convertTo()] to change units while preserving the
#'   physical value.
#' @param object A [Quantity-class] object.
#' @param ... Additional arguments for methods.
#' @return A [Unit-class] object, or the modified object for the replacement
#'   form.
#' @export
setGeneric("unitOf", function(object, ...) standardGeneric("unitOf"))

#' @rdname unitOf
#' @param value A [Unit-class] object or a registered unit symbol.
#' @export
setGeneric("unitOf<-", function(object, value) standardGeneric("unitOf<-"))

#' @rdname magnitude
setMethod("magnitude", "Quantity", function(object, ...) {
  object@.Data
})

#' @rdname magnitude
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

#' @rdname unitOf
setReplaceMethod("unitOf", "Quantity", function(object, value) {
  if (is.character(value)) {
    value <- lookupUnit(value)
  }
  object@unit <- value
  validObject(object)
  object
})

#' @title Format a unit's dimensions
#' @description Writes the non-zero exponents of a unit as a compact string
#'   such as `"L M T^-2"`.
#' @param unit A [Unit-class] object.
#' @return A single string, `"1"` for dimensionless units.
#' @keywords internal
formatDims <- function(unit) {
  # One letter per base dimension, in baseDimensions() order
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

  # Long vectors get a preview, like a truncated tibble
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
  if (missing(i)) {
    return(x)
  }
  new("Quantity", x@.Data[i], unit = x@unit)
})
