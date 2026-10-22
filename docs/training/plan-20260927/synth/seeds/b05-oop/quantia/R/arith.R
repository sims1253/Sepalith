#' Compare the dimensions of two units
#'
#' @param u1,u2 [Unit-class] objects.
#' @return `TRUE` if both units measure the same physical dimension.
#' @keywords internal
sameDims <- function(u1, u2) {
  isTRUE(all.equal(u1@dims, u2@dims, check.attributes = FALSE))
}

#' Combine two units under multiplication or division
#'
#' Adds or subtracts the dimension exponents, multiplies or divides the
#' scales, and composes a symbol for the result.
#'
#' @param u1,u2 [Unit-class] objects.
#' @param op Either `"*"` or `"/"`.
#' @return A new [Unit-class] object.
#' @keywords internal
combineUnits <- function(u1, u2, op = c("*", "/")) {
  op <- match.arg(op)
  if (u1@offset != 0 || u2@offset != 0) {
    stop("Can't multiply or divide units with an offset (", u1@symbol, ", ", u2@symbol, ").",
      call. = FALSE)
  }
  switch(op,
    "*" = unit(
      symbol = paste(u1@symbol, u2@symbol, sep = "*"),
      dims = u1@dims + u2@dims,
      scale = u1@scale * u2@scale
    ),
    "/" = unit(
      symbol = paste(u1@symbol, u2@symbol, sep = "/"),
      dims = u1@dims - u2@dims,
      scale = u1@scale / u2@scale
    )
  )
}

#' Raise a unit to a power
#'
#' @param u A [Unit-class] object.
#' @param power A single number.
#' @return A new [Unit-class] object.
#' @keywords internal
powerUnit <- function(u, power) {
  if (u@offset != 0) {
    stop("Can't raise a unit with an offset (", u@symbol, ") to a power.", call. = FALSE)
  }
  unit(
    symbol = paste0(u@symbol, "^", power),
    dims = u@dims * power,
    scale = u@scale^power
  )
}

#' Arithmetic on quantities
#'
#' Addition and subtraction require compatible dimensions and return a result
#' in the unit of the left operand. Multiplication and division combine the
#' units. Plain numbers are treated as dimensionless.
#'
#' @param e1,e2 Operands; at least one is a [Quantity-class] object.
#' @return A [Quantity-class] object.
#' @name quantity-arith
NULL

#' @rdname quantity-arith
setMethod("Arith", signature("Quantity", "Quantity"), function(e1, e2) {
  op <- .Generic
  switch(op,
    "+" = ,
    "-" = {
      if (!sameDims(e1@unit, e2@unit)) {
        stop("Can't ", if (op == "+") "add" else "subtract", " ", e2@unit@symbol,
          " and ", e1@unit@symbol, ": incompatible dimensions.", call. = FALSE)
      }
      rhs <- convertTo(e2, e1@unit)
      new("Quantity", callGeneric(e1@.Data, rhs@.Data), unit = e1@unit)
    },
    "*" = ,
    "/" = {
      u <- combineUnits(e1@unit, e2@unit, op)
      new("Quantity", callGeneric(e1@.Data, e2@.Data), unit = u)
    },
    stop("Operator `", op, "` is not supported between two quantities.", call. = FALSE)
  )
})

#' @rdname quantity-arith
setMethod("Arith", signature("Quantity", "numeric"), function(e1, e2) {
  op <- .Generic
  switch(op,
    "*" = ,
    "/" = new("Quantity", callGeneric(e1@.Data, e2), unit = e1@unit),
    "^" = {
      if (length(e2) != 1L) {
        stop("Exponent must be a single number.", call. = FALSE)
      }
      new("Quantity", e1@.Data^e2, unit = powerUnit(e1@unit, e2))
    },
    {
      if (!isDimensionless(e1@unit)) {
        stop("Can't apply `", op, "` to ", e1@unit@symbol, " and a plain number.",
          call. = FALSE)
      }
      new("Quantity", callGeneric(e1@.Data, e2), unit = e1@unit)
    }
  )
})

#' @rdname quantity-arith
setMethod("Arith", signature("numeric", "Quantity"), function(e1, e2) {
  op <- .Generic
  switch(op,
    "*" = new("Quantity", e1 * e2@.Data, unit = e2@unit),
    "/" = new("Quantity", e1 / e2@.Data, unit = powerUnit(e2@unit, -1)),
    {
      if (!isDimensionless(e2@unit)) {
        stop("Can't apply `", op, "` to a plain number and ", e2@unit@symbol, ".",
          call. = FALSE)
      }
      new("Quantity", callGeneric(e1, e2@.Data), unit = e2@unit)
    }
  )
})

#' Compare quantities
#'
#' The right operand is converted to the unit of the left operand before the
#' magnitudes are compared.
#'
#' @param e1,e2 [Quantity-class] objects with compatible dimensions.
#' @return A logical vector.
#' @rdname quantity-compare
setMethod("Compare", signature("Quantity", "Quantity"), function(e1, e2) {
  if (!sameDims(e1@unit, e2@unit)) {
    stop("Can't compare ", e1@unit@symbol, " with ", e2@unit@symbol, ".", call. = FALSE)
  }
  rhs <- convertTo(e2, e1@unit)
  callGeneric(e1@.Data, rhs@.Data)
})

#' Summaries of quantities
#'
#' `sum()`, `min()`, `max()` and `range()` keep the unit; other summary
#' functions are not meaningful for quantities.
#'
#' @param x A [Quantity-class] object.
#' @param ... Further quantities, which are converted to the unit of `x`.
#' @param na.rm Whether to drop missing values.
#' @return A [Quantity-class] object.
#' @rdname quantity-summary
setMethod("Summary", "Quantity", function(x, ..., na.rm = FALSE) {
  op <- .Generic
  if (!op %in% c("sum", "min", "max", "range")) {
    stop("`", op, "()` is not defined for quantities.", call. = FALSE)
  }
  others <- lapply(list(...), function(q) magnitude(convertTo(q, x@unit)))
  values <- do.call(op, c(list(x@.Data), others, list(na.rm = na.rm)))
  new("Quantity", values, unit = x@unit)
})
