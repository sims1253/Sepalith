#' @title Concentration units
#' @return A named numeric vector giving mg/L per unit.
#' @keywords internal
conc.units = function() {
  c("mg/L" = 1, "ug/mL" = 1, "ng/mL" = 0.001, "ug/L" = 0.001)
}

#' Validate a Concentration object
#'
#' @param object A `Concentration` object.
#' @details The slots `analyte`, `unit`, `time.h` and `value` are checked in
#'   turn.
#' @return `TRUE` or a character vector of problems.
#' @keywords internal
valid.concentration = function(object) {
  problems = character()

  # Check `analyte`
  if (length(object@analyte) != 1L || !nzchar(object@analyte)) {
    problems = c(problems, "`analyte` must be a single non-empty string.")
  }

  # Check `unit`
  if (length(object@unit) != 1L || !object@unit %in% names(conc.units())) {
    problems = c(problems, sprintf("`unit` must be one of %s.", toString(names(conc.units()))))
  }

  # Check `time.h`
  if (anyNA(object@time.h) || is.unsorted(object@time.h, strictly = TRUE)) {
    problems = c(problems, "`time.h` must be strictly increasing without NAs.")
  } else if (any(object@time.h < 0)) {
    problems = c(problems, "`time.h` must not be negative.")
  }

  # Check `value`
  if (length(object@value) != length(object@time.h) || any(object@value < 0, na.rm = TRUE)) {
    problems = c(problems, "`value` must hold one non-negative number per `time.h`.")
  }

  if (length(problems)) problems else TRUE
}

#' Measured drug concentrations
#'
#' @slot analyte Name of the measured substance.
#' @slot unit Concentration unit.
#' @slot time.h Sampling times in hours after the first dose.
#' @slot value Measured concentrations; `NA` for samples below the limit of
#'   quantification.
#' @export
setClass(
  "Concentration",
  representation(analyte = "character", unit = "character", time.h = "numeric", value = "numeric"),
  validity = valid.concentration
)

#' Create concentration measurements
#'
#' @param analyte Name of the measured substance.
#' @param time.h Sampling times in hours.
#' @param value Measured concentrations.
#' @param unit Concentration unit, one of the names of [conc.units()].
#' @return A `Concentration` object.
#' @export
concentration = function(analyte, time.h, value, unit = "mg/L") {
  ord = order(time.h)
  new(
    "Concentration",
    analyte = analyte,
    unit = unit,
    time.h = as.numeric(time.h)[ord],
    value = as.numeric(value)[ord]
  )
}

#' @title Convert concentrations to another unit
#' @param x A `Concentration` object.
#' @param to Target unit.
#' @return `x` expressed in `to`.
#' @export
conc.convert = function(x, to = c("mg/L", "ug/mL", "ng/mL", "ug/L")) {
  to = match.arg(to)
  f = conc.units()
  x@value = x@value * f[[x@unit]] / f[[to]]
  x@unit = to
  x
}

#' Summarise a concentration-time profile
#'
#' @param x A `Concentration` object.
#' @details Samples with missing values are dropped first. Areas use the
#'   linear trapezoidal rule between observed samples. The terminal rate
#'   constant `lambda.z` comes from a log-linear fit through the last three
#'   positive samples after the peak; with fewer such samples it is `NA`, as
#'   is `half.life`.
#' @return A list with elements `n`, `cmax`, `tmax`, `cmin`, `tlast`,
#'   `clast`, `auc`, `aumc`, `mrt`, `lambda.z` and `half.life`.
#' @importFrom stats lm.fit
#' @importFrom utils head tail
#' @export
conc.summary = function(x) {
  ok = !is.na(x@value)
  t = x@time.h[ok]
  v = x@value[ok]
  n = length(v)
  if (n == 0L) {
    stop("No quantified samples to summarise.", call. = FALSE)
  }

  # First occurrence wins when the peak is reached twice
  i.max = which.max(v)
  if (n > 1L) {
    dt = diff(t)
    auc = sum(dt * (head(v, -1) + tail(v, -1)) / 2)
    aumc = sum(dt * (head(t * v, -1) + tail(t * v, -1)) / 2)
  } else {
    auc = 0
    aumc = 0
  }
  mrt = if (auc > 0) aumc / auc else NA_real_

  term = tail(which(v > 0 & seq_len(n) > i.max), 3)
  lambda.z = NA_real_
  if (length(term) == 3L) {
    fit = stats::lm.fit(cbind(1, t[term]), log(v[term]))
    lambda.z = -unname(fit$coefficients[2])
  }

  list(
    n = n,
    cmax = v[i.max],
    tmax = t[i.max],
    cmin = min(v),
    tlast = t[n],
    clast = v[n],
    auc = auc,
    aumc = aumc,
    mrt = mrt,
    lambda.z = lambda.z,
    half.life = log(2) / lambda.z
  )
}

#' Coerce concentrations to base types
#'
#' @param x A `Concentration` object.
#' @param to Target type. `"data.frame"` and `"matrix"` hold `time.h` and
#'   `value` columns; `"numeric"` returns the values named by time;
#'   `"character"` returns labels such as `"1.2 mg/L @ 2h"`; and `"list"`
#'   returns every slot as a list element.
#' @return An object of the requested type.
#' @export
conc.as = function(x, to = c("data.frame", "matrix", "numeric", "character", "list")) {
  to = match.arg(to)
  switch(to,
    data.frame = {
      out = data.frame(time.h = x@time.h, value = x@value)
      attr(out, "unit") = x@unit
    },
    matrix = {
      out = cbind(time.h = x@time.h, value = x@value)
      attr(out, "unit") = x@unit
    },
    numeric = {
      out = x@value
      names(out) = format(x@time.h)
    },
    character = {
      out = sprintf("%s %s @ %sh", format(x@value), x@unit, format(x@time.h))
      names(out) = NULL
    },
    list = {
      out = list(analyte = x@analyte, unit = x@unit, time.h = x@time.h, value = x@value)
    }
  )
  out
}

setAs("Concentration", "data.frame", function(from) conc.as(from, "data.frame"))
setAs("Concentration", "matrix", function(from) conc.as(from, "matrix"))
setAs("Concentration", "numeric", function(from) conc.as(from, "numeric"))
setAs("Concentration", "character", function(from) conc.as(from, "character"))
setAs("Concentration", "list", function(from) conc.as(from, "list"))

#' Print concentrations
#'
#' @param object A `Concentration` object.
#' @return `object`, invisibly.
#' @keywords internal
show.concentration = function(object) {
  quantified = sum(!is.na(object@value))
  cat(sprintf("<Concentration> %s, %d of %d samples quantified\n", object@analyte, quantified,
    length(object@value)))
  if (quantified == 0L) {
    return(invisible(object))
  }
  s = conc.summary(object)
  cat(sprintf("  cmax: %g %s at %g h\n", s$cmax, object@unit, s$tmax))
  cat(sprintf("  auc:  %g %s*h\n", s$auc, object@unit))
  if (!is.na(s$half.life)) {
    cat(sprintf("  t1/2: %.3g h\n", s$half.life))
  }
  invisible(object)
}

setMethod("show", "Concentration", function(object) show.concentration(object))
