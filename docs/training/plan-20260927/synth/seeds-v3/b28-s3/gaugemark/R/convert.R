#' @title Convert lengths to another unit
#' @description Generic with methods for readings, ranges and plain numbers.
#' @param x An object to convert.
#' @param to The target unit, one of `"mm"`, `"cm"`, `"m"`, `"km"` or `"in"`.
#' @param ... Passed to methods.
#' @return An object like `x`, expressed in `to`.
#' @export
convertUnit = function(x, to, ...) {
  UseMethod("convertUnit")
}

#' @rdname convertUnit
#' @export
convertUnit.gm_reading = function(x, to, ...) {
  from = attr(x, "unit")
  if (identical(from, to)) {
    return(x)
  }
  ratio = unitFactor(from) / unitFactor(to)
  value = as.double(unclass(x)) * ratio
  newReading(value, to)
}

#' @rdname convertUnit
#' @export
convertUnit.gm_range = function(x, to, ...) {
  from = attr(x, "unit")
  if (identical(from, to)) {
    return(x)
  }
  ratio = unitFactor(from) / unitFactor(to)
  fields = unclass(x)
  lower = fields$lower * ratio
  upper = fields$upper * ratio
  newRange(lower, upper, to)
}

#' @rdname convertUnit
#' @param from The unit of a plain numeric `x`.
#' @export
convertUnit.default = function(x, to, from, ...) {
  if (!is.numeric(x)) {
    stop(sprintf("Can't convert an object of class <%s>.", class(x)[1L]), call. = FALSE)
  }
  if (missing(from)) {
    stop("`from` is required for plain numeric input.", call. = FALSE)
  }
  ratio = unitFactor(from) / unitFactor(to)
  as.double(x) * ratio
}

#' @title Convert readings to a data frame
#' @param x An object of class `gm_reading`.
#' @param row.names Optional row names for the result.
#' @param optional Ignored.
#' @param ... Unused.
#' @param metres If `TRUE`, add a column with the value in metres.
#' @return A data frame with columns `value` and `unit`, plus `metres` when
#'   requested.
#' @export
as.data.frame.gm_reading = function(x, row.names = NULL, optional = FALSE, ...,
                                    metres = FALSE) {
  unit = attr(x, "unit")
  value = as.double(unclass(x))
  out = data.frame(
    value = value,
    unit = rep(unit, length(value)),
    stringsAsFactors = FALSE
  )
  if (metres) {
    out$metres = value * unitFactor(unit)
  }
  if (!is.null(row.names)) {
    rownames(out) = row.names
  }
  out
}

#' @title Convert ranges to a data frame
#' @param x An object of class `gm_range`.
#' @param row.names Optional row names for the result.
#' @param optional Ignored.
#' @param ... Unused.
#' @param metres If `TRUE`, add `lower_metres` and `upper_metres` columns.
#' @return A data frame with columns `lower`, `upper` and `unit`, plus
#'   `lower_metres` and `upper_metres` when requested.
#' @export
as.data.frame.gm_range = function(x, row.names = NULL, optional = FALSE, ...,
                                  metres = FALSE) {
  unit = attr(x, "unit")
  fields = unclass(x)
  out = data.frame(
    lower = fields$lower,
    upper = fields$upper,
    unit = rep(unit, length(fields$lower)),
    stringsAsFactors = FALSE
  )
  if (metres) {
    out$lower_metres = fields$lower * unitFactor(unit)
    out$upper_metres = fields$upper * unitFactor(unit)
  }
  if (!is.null(row.names)) {
    rownames(out) = row.names
  }
  out
}
