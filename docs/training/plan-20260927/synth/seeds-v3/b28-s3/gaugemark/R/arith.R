#' @title Express an operand in a given unit
#' @param x A `gm_reading`, a `gm_range` or a plain number.
#' @param unit The target unit.
#' @return A double vector for readings and numbers, a list with elements
#'   `lower` and `upper` for ranges. Plain numbers are returned unchanged.
#' @keywords internal
inUnit = function(x, unit) {
  from = attr(x, "unit")
  if (is.null(from)) {
    return(as.double(x))
  }
  ratio = unitFactor(from) / unitFactor(unit)
  if (inherits(x, "gm_range")) {
    fields = unclass(x)
    return(list(lower = fields$lower * ratio, upper = fields$upper * ratio))
  }
  as.double(unclass(x)) * ratio
}

#' @title Pick the unit two operands should share
#' @description The finer of the two units wins, so that no precision is lost
#'   when one side is converted. A plain number has no unit and never wins.
#' @param e1,e2 Operands of a group generic.
#' @return A single unit string.
#' @keywords internal
commonUnit = function(e1, e2) {
  u1 = attr(e1, "unit")
  u2 = attr(e2, "unit")
  if (is.null(u1)) {
    return(u2)
  }
  if (is.null(u2)) {
    return(u1)
  }
  if (unitFactor(u1) <= unitFactor(u2)) u1 else u2
}

#' @title Arithmetic and comparison for readings
#' @description Both sides are first expressed in [commonUnit()]. `+` and `-`
#'   need readings on both sides, `*` needs a plain number on one side, and
#'   `/` returns a plain ratio when both sides are readings.
#' @param e1,e2 Operands; at least one is a `gm_reading`.
#' @return A logical vector for comparisons, a double vector for a ratio of
#'   readings, otherwise a `gm_reading`.
#' @export
Ops.gm_reading = function(e1, e2) {
  if (nargs() == 1L) {
    stop(sprintf("Unary `%s` is not supported for <%s>.", .Generic, "gm_reading"),
      call. = FALSE)
  }
  readingLeft = inherits(e1, "gm_reading")
  readingRight = inherits(e2, "gm_reading")
  unit = commonUnit(e1, e2)
  lhs = inUnit(e1, unit)
  rhs = inUnit(e2, unit)

  # Comparisons work on the values in the shared unit
  if (.Generic %in% c("==", "!=", "<", ">", "<=", ">=")) {
    return(get(.Generic)(lhs, rhs))
  }

  switch(.Generic,
    "+" = , "-" = {
      if (!(readingLeft && readingRight)) {
        stop(sprintf("`%s` needs a <%s> on both sides.", .Generic, "gm_reading"),
          call. = FALSE)
      }
      newReading(get(.Generic)(lhs, rhs), unit)
    },
    "*" = {
      if (readingLeft && readingRight) {
        stop(sprintf("`%s` needs a plain number on one side.", .Generic), call. = FALSE)
      }
      newReading(lhs * rhs, unit)
    },
    "/" = {
      if (!readingLeft) {
        stop(sprintf("`%s` needs the <%s> on the left.", .Generic, "gm_reading"),
          call. = FALSE)
      }
      if (readingRight) lhs / rhs else newReading(lhs / rhs, unit)
    },
    stop(sprintf("`%s` is not supported for <%s>.", .Generic, "gm_reading"), call. = FALSE)
  )
}

#' @title Arithmetic and comparison for ranges
#' @description Comparisons need ranges on both sides. `+` and `-` shift both
#'   bounds by a number taken in the range's unit; `*` and `/` scale both
#'   bounds by a positive number. Readings can't be mixed in: S3 group
#'   dispatch would find two incompatible methods.
#' @param e1,e2 Operands; at least one is a `gm_range`.
#' @return A logical vector for comparisons, otherwise a `gm_range`.
#' @export
Ops.gm_range = function(e1, e2) {
  if (nargs() == 1L) {
    stop(sprintf("Unary `%s` is not supported for <%s>.", .Generic, "gm_range"),
      call. = FALSE)
  }
  rangeLeft = inherits(e1, "gm_range")
  rangeRight = inherits(e2, "gm_range")

  # Comparisons need a range on both sides
  if (.Generic %in% c("==", "!=", "<", ">", "<=", ">=")) {
    if (!(rangeLeft && rangeRight)) {
      stop(sprintf("`%s` needs a <%s> on both sides.", .Generic, "gm_range"),
        call. = FALSE)
    }
    unit = commonUnit(e1, e2)
    lhs = inUnit(e1, unit)
    rhs = inUnit(e2, unit)
    out = switch(.Generic,
      "==" = lhs$lower == rhs$lower & lhs$upper == rhs$upper,
      "!=" = lhs$lower != rhs$lower | lhs$upper != rhs$upper,
      "<" = lhs$upper < rhs$lower,
      "<=" = lhs$upper <= rhs$lower,
      ">" = lhs$lower > rhs$upper,
      ">=" = lhs$lower >= rhs$upper
    )
    return(out)
  }

  # Arithmetic needs one range and one number
  if (rangeLeft && rangeRight) {
    stop(sprintf("`%s` needs a plain number on one side.", .Generic), call. = FALSE)
  }
  if (!rangeLeft && .Generic %in% c("-", "/")) {
    stop(sprintf("`%s` needs the <%s> on the left.", .Generic, "gm_range"), call. = FALSE)
  }
  x = if (rangeLeft) e1 else e2
  amount = if (rangeLeft) e2 else e1
  if (!is.numeric(amount)) {
    stop(sprintf("`%s` needs a plain number on one side.", .Generic), call. = FALSE)
  }
  unit = attr(x, "unit")
  bounds = unclass(x)
  amount = as.double(amount)
  switch(.Generic,
    "+" = , "-" = {
      lower = get(.Generic)(bounds$lower, amount)
      upper = get(.Generic)(bounds$upper, amount)
    },
    "*" = , "/" = {
      if (any(amount <= 0)) {
        stop(sprintf("`%s` needs a positive number.", .Generic), call. = FALSE)
      }
      lower = get(.Generic)(bounds$lower, amount)
      upper = get(.Generic)(bounds$upper, amount)
    },
    stop(sprintf("`%s` is not supported for <%s>.", .Generic, "gm_range"), call. = FALSE)
  )
  newRange(lower, upper, unit)
}

#' @title Rounding and absolute values for readings
#' @param x An object of class `gm_reading`.
#' @param ... Passed on, e.g. `digits` for [round()].
#' @return An object of class `gm_reading` in the same unit.
#' @export
Math.gm_reading = function(x, ...) {
  allowed = c("abs", "floor", "ceiling", "round", "signif", "trunc")
  if (!.Generic %in% allowed) {
    stop(sprintf("`%s` is not supported for <%s>.", .Generic, "gm_reading"), call. = FALSE)
  }
  fun = match.fun(.Generic)
  unit = attr(x, "unit")
  value = as.double(fun(as.double(unclass(x)), ...))
  newReading(value, unit)
}

#' @title Rounding and absolute values for ranges
#' @param x An object of class `gm_range`.
#' @param ... Passed on, e.g. `digits` for [round()].
#' @return An object of class `gm_range` in the same unit.
#' @export
Math.gm_range = function(x, ...) {
  allowed = c("abs", "floor", "ceiling", "round", "signif", "trunc")
  if (!.Generic %in% allowed) {
    stop(sprintf("`%s` is not supported for <%s>.", .Generic, "gm_range"), call. = FALSE)
  }
  fun = match.fun(.Generic)
  unit = attr(x, "unit")
  bounds = unclass(x)
  lower = as.double(fun(bounds$lower, ...))
  upper = as.double(fun(bounds$upper, ...))
  newRange(lower, upper, unit)
}

#' @title Summaries of readings
#' @description `min`, `max`, `range` and `sum` are supported. All arguments
#'   are expressed in the unit of the first one.
#' @param ... Objects of class `gm_reading` or plain numbers.
#' @param na.rm Whether to drop missing values.
#' @return An object of class `gm_reading`.
#' @export
Summary.gm_reading = function(..., na.rm = FALSE) {
  args = list(...)
  unit = attr(args[[1L]], "unit")
  values = unlist(lapply(args, inUnit, unit = unit))
  switch(.Generic,
    min = newReading(min(values, na.rm = na.rm), unit),
    max = newReading(max(values, na.rm = na.rm), unit),
    range = newReading(range(values, na.rm = na.rm), unit),
    sum = newReading(sum(values, na.rm = na.rm), unit),
    stop(sprintf("`%s` is not supported for <%s>.", .Generic, "gm_reading"), call. = FALSE)
  )
}

#' @title Summaries of ranges
#' @description `min` returns the smallest `lower` bound and `max` the largest
#'   `upper` bound, both as readings; `range` returns the covering range. All
#'   arguments are expressed in the unit of the first one.
#' @param ... Objects of class `gm_range`.
#' @param na.rm Whether to drop missing values.
#' @return An object of class `gm_reading`, or `gm_range` for `range`.
#' @export
Summary.gm_range = function(..., na.rm = FALSE) {
  args = list(...)
  unit = attr(args[[1L]], "unit")
  bounds = lapply(args, inUnit, unit = unit)
  lower = unlist(lapply(bounds, `[[`, "lower"))
  upper = unlist(lapply(bounds, `[[`, "upper"))
  switch(.Generic,
    min = newReading(min(lower, na.rm = na.rm), unit),
    max = newReading(max(upper, na.rm = na.rm), unit),
    range = newRange(min(lower, na.rm = na.rm), max(upper, na.rm = na.rm), unit),
    stop(sprintf("`%s` is not supported for <%s>.", .Generic, "gm_range"), call. = FALSE)
  )
}
