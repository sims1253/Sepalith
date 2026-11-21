#' @title Mass units
#' @return A named numeric vector giving milligrams per unit.
#' @keywords internal
mass.factors = function() {
  c(g = 1000, mg = 1, mcg = 0.001, ng = 1e-6)
}

# Administration route codes
dose.routes = c("oral", "iv", "sc", "im")

#' @title Validate a Dose object
#' @param object A `Dose` object.
#' @details The slots `amount`, `unit` and `route` are checked in turn.
#' @return `TRUE` or a character vector of problems.
#' @keywords internal
valid.dose = function(object) {
  problems = character()

  # Check `amount`
  if (anyNA(object@amount) || any(object@amount < 0)) {
    problems = c(problems, "`amount` must be non-negative and not missing.")
  }

  # Check `unit`
  if (length(object@unit) != 1L || !object@unit %in% names(mass.factors())) {
    problems = c(problems, sprintf("`unit` must be one of %s.", toString(names(mass.factors()))))
  }

  # Check `route`
  if (length(object@route) != 1L || !object@route %in% dose.routes) {
    problems = c(problems, sprintf("`route` must be one of %s.", toString(dose.routes)))
  }

  if (length(problems)) problems else TRUE
}

#' A drug dose
#'
#' @slot amount Dose amounts.
#' @slot unit Mass unit shared by all amounts.
#' @slot route Administration route.
#' @importFrom methods setClass new validObject
#' @export
setClass(
  "Dose",
  representation(amount = "numeric", unit = "character", route = "character"),
  prototype(amount = numeric(), unit = "mg", route = "oral"),
  validity = valid.dose
)

#' Create a dose
#'
#' @param amount Dose amounts.
#' @param unit Mass unit, one of the names of [mass.factors()]. Spelled-out
#'   names such as `"milligram"` and the alias `"ug"` for `"mcg"` are
#'   accepted.
#' @param route Administration route, one of `dose.routes`. The aliases
#'   `"po"`, `"intravenous"`, `"subcutaneous"` and `"intramuscular"` are
#'   accepted.
#' @return A `Dose` object.
#' @export
dose = function(amount, unit = "mg", route = "oral") {
  # Check `amount`
  if (!is.numeric(amount)) {
    stop("`amount` must be numeric.", call. = FALSE)
  }

  # Check `unit`
  if (!is.character(unit) || length(unit) != 1L) {
    stop("`unit` must be a single string.", call. = FALSE)
  }
  # Spelled-out names and aliases map onto the unit codes
  unit = switch(tolower(unit),
    gram = , grams = "g",
    milligram = , milligrams = "mg",
    ug = , microgram = , micrograms = "mcg",
    nanogram = , nanograms = "ng",
    tolower(unit)
  )

  # Check `route`
  if (!is.character(route) || length(route) != 1L) {
    stop("`route` must be a single string.", call. = FALSE)
  }
  route = switch(tolower(route),
    po = "oral",
    intravenous = "iv",
    subcutaneous = "sc",
    intramuscular = "im",
    tolower(route)
  )

  new("Dose", amount = as.numeric(amount), unit = unit, route = route)
}

#' @title Convert a dose to another unit
#' @param x A `Dose` object.
#' @param to Target unit.
#' @return `x` expressed in `to`.
#' @export
dose.convert = function(x, to = c("mg", "g", "mcg", "ng")) {
  to = match.arg(to)
  f = mass.factors()
  x@amount = x@amount * f[[x@unit]] / f[[to]]
  x@unit = to
  x
}

#' Arithmetic on doses
#'
#' @param e1,e2 The operands. At least one is a `Dose`.
#' @param op The operator name, as found in `.Generic`.
#' @details Two doses may be added or subtracted when they share a route; the
#'   result uses the unit of `e1`. A dose may be multiplied or divided by a
#'   number, and a number may multiply a dose.
#' @return A `Dose` object.
#' @keywords internal
dose.arith = function(e1, e2, op) {
  # Dose op Dose
  if (is(e1, "Dose") && is(e2, "Dose")) {
    if (!identical(e1@route, e2@route)) {
      stop("Doses with different routes cannot be combined.", call. = FALSE)
    }
    e2 = dose.convert(e2, e1@unit)
    e1@amount = switch(op,
      "+" = e1@amount + e2@amount,
      "-" = e1@amount - e2@amount,
      stop(sprintf("`%s` is not defined for two doses.", op), call. = FALSE)
    )
    validObject(e1)
    return(e1)
  }

  # Dose op number
  if (is(e1, "Dose")) {
    e1@amount = switch(op,
      "*" = e1@amount * e2,
      "/" = e1@amount / e2,
      stop(sprintf("`%s` is not defined for a dose and a number.", op), call. = FALSE)
    )
    validObject(e1)
    return(e1)
  }

  # number op Dose
  e2@amount = switch(op,
    "*" = e1 * e2@amount,
    stop(sprintf("`%s` is not defined for a number and a dose.", op), call. = FALSE)
  )
  validObject(e2)
  e2
}

setMethod("Arith", signature("Dose", "Dose"), function(e1, e2) dose.arith(e1, e2, .Generic))
setMethod("Arith", signature("Dose", "numeric"), function(e1, e2) dose.arith(e1, e2, .Generic))
setMethod("Arith", signature("numeric", "Dose"), function(e1, e2) dose.arith(e1, e2, .Generic))

#' Compare doses
#'
#' @inheritParams dose.arith
#' @details Both doses are converted to milligrams before comparison.
#' @return A logical vector.
#' @keywords internal
dose.compare = function(e1, e2, op = c("==", "!=", "<", ">", "<=", ">=")) {
  op = match.arg(op)
  a = dose.convert(e1, "mg")@amount
  b = dose.convert(e2, "mg")@amount
  switch(op,
    "==" = abs(a - b) < 1e-9,
    "!=" = abs(a - b) >= 1e-9,
    "<" = a < b,
    ">" = a > b,
    "<=" = a <= b,
    ">=" = a >= b
  )
}

setMethod("Compare", signature("Dose", "Dose"), function(e1, e2) dose.compare(e1, e2, .Generic))

#' Format a dose
#'
#' @param x A `Dose` object.
#' @param digits Significant digits for amounts.
#' @param scale If `TRUE`, each amount is shown in the largest unit in which
#'   it is at least one, so 2500 mg prints as 2.5 g.
#' @return A character vector, one element per amount.
#' @export
format.dose = function(x, digits = 3, scale = FALSE) {
  amounts = x@amount
  units = rep(x@unit, length(amounts))
  if (scale) {
    f = sort(mass.factors(), decreasing = TRUE)
    mg = amounts * f[[x@unit]]
    # First unit (largest first) in which the amount is at least one
    pick = vapply(mg, function(m) which(m >= f | f == min(f))[1], integer(1))
    amounts = mg / f[pick]
    units = names(f)[pick]
  }
  amounts = formatC(signif(amounts, digits), format = "fg", digits = digits)
  sprintf("%s %s %s", trimws(amounts), units, x@route)
}

setMethod("format", "Dose", function(x, ...) format.dose(x, ...))
setMethod("show", "Dose", function(object) {
  lines = if (length(object@amount)) format.dose(object) else "(empty)"
  cat("<Dose>\n", paste0("  ", lines, "\n"), sep = "")
  invisible(object)
})
