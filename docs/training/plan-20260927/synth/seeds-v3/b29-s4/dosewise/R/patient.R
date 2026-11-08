#' Validate a Patient object
#'
#' @param object A `Patient` object.
#' @details `id` and `sex` are checked first. The numeric slots must then lie
#'   within these limits: `age.y` 0 to 120, `weight.kg` 0.5 to 300,
#'   `height.cm` 30 to 250 and `scr.mgdl` 0.1 to 20.
#' @return `TRUE` or a character vector of problems.
#' @keywords internal
valid.patient = function(object) {
  problems = character()
  if (length(object@id) != 1L || is.na(object@id)) {
    problems = c(problems, "`id` must be a single string.")
  }
  if (length(object@sex) != 1L || !object@sex %in% c("female", "male")) {
    problems = c(problems, "`sex` must be \"female\" or \"male\".")
  }

  # Check `age.y`
  if (length(object@age.y) != 1L || is.na(object@age.y) || object@age.y < 0 ||
    object@age.y > 120) {
    problems = c(problems, sprintf("`%s` must lie between %g and %g.", "age.y", 0, 120))
  }

  # Check `weight.kg`
  if (length(object@weight.kg) != 1L || is.na(object@weight.kg) || object@weight.kg < 0.5 ||
    object@weight.kg > 300) {
    problems = c(problems, sprintf("`%s` must lie between %g and %g.", "weight.kg", 0.5, 300))
  }

  # Check `height.cm`
  if (length(object@height.cm) != 1L || is.na(object@height.cm) || object@height.cm < 30 ||
    object@height.cm > 250) {
    problems = c(problems, sprintf("`%s` must lie between %g and %g.", "height.cm", 30, 250))
  }

  # Check `scr.mgdl`
  if (length(object@scr.mgdl) != 1L || is.na(object@scr.mgdl) || object@scr.mgdl < 0.1 ||
    object@scr.mgdl > 20) {
    problems = c(problems, sprintf("`%s` must lie between %g and %g.", "scr.mgdl", 0.1, 20))
  }

  if (length(problems)) problems else TRUE
}

#' A patient profile for dose individualisation
#'
#' @slot id Patient identifier.
#' @slot sex `"female"` or `"male"`.
#' @slot age.y Age in years.
#' @slot weight.kg Actual body weight in kg.
#' @slot height.cm Height in cm.
#' @slot scr.mgdl Serum creatinine in mg/dL.
#' @export
setClass(
  "Patient",
  representation(
    id = "character",
    sex = "character",
    age.y = "numeric",
    weight.kg = "numeric",
    height.cm = "numeric",
    scr.mgdl = "numeric"
  ),
  validity = valid.patient
)

#' Create a patient profile
#'
#' @param id Patient identifier.
#' @param sex `"female"` or `"male"`; partial matching is allowed.
#' @param age.y Age in years.
#' @param weight Body weight, in kg or lb depending on `units`.
#' @param height Height, in cm or inches depending on `units`.
#' @param scr.mgdl Serum creatinine in mg/dL.
#' @param units `"metric"` (kg and cm) or `"imperial"` (lb and inches).
#' @return A `Patient` object with weight and height stored in metric units.
#' @export
patient = function(id, sex, age.y, weight, height, scr.mgdl = 1,
                   units = c("metric", "imperial")) {
  units = match.arg(units)
  if (!is.numeric(weight) || !is.numeric(height)) {
    stop("`weight` and `height` must be numeric.", call. = FALSE)
  }
  if (units == "imperial") {
    weight.kg = weight * 0.45359237
    height.cm = height * 2.54
  } else {
    weight.kg = weight
    height.cm = height
  }

  new(
    "Patient",
    id = as.character(id),
    sex = match.arg(tolower(sex), c("female", "male")),
    age.y = as.numeric(age.y),
    weight.kg = round(weight.kg, 2),
    height.cm = round(height.cm, 1),
    scr.mgdl = as.numeric(scr.mgdl)
  )
}

#' Patient setters
#'
#' @param x A `Patient` object.
#' @param value The new value.
#' @details These back the replacement methods `weight<-` and `height<-`.
#' @return The modified `Patient` object.
#' @keywords internal
set.patient.weight = function(x, value) {
  if (!is.numeric(value) || length(value) != 1L) {
    stop(sprintf("`%s` must be a single number.", "weight.kg"), call. = FALSE)
  }
  x@weight.kg = value
  validObject(x)
  x
}

#' @rdname set.patient.weight
set.patient.height = function(x, value) {
  if (!is.numeric(value) || length(value) != 1L) {
    stop(sprintf("`%s` must be a single number.", "height.cm"), call. = FALSE)
  }
  x@height.cm = value
  validObject(x)
  x
}

setGeneric("weight<-", function(x, value) standardGeneric("weight<-"))
setGeneric("height<-", function(x, value) standardGeneric("height<-"))
setReplaceMethod("weight", "Patient", function(x, value) set.patient.weight(x, value))
setReplaceMethod("height", "Patient", function(x, value) set.patient.height(x, value))

#' Body surface area
#'
#' @param x A `Patient` object.
#' @param method Formula to use.
#' @return Body surface area in square metres.
#' @export
body.surface.area = function(x, method = c("mosteller", "dubois", "haycock", "gehan")) {
  method = match.arg(method)
  w = x@weight.kg
  h = x@height.cm
  switch(method,
    mosteller = sqrt(w * h / 3600),
    dubois = 0.007184 * w^0.425 * h^0.725,
    haycock = 0.024265 * w^0.5378 * h^0.3964,
    gehan = 0.0235 * w^0.51456 * h^0.42246
  )
}

#' Ideal body weight
#'
#' @param x A `Patient` object.
#' @param method Formula to use.
#' @details All formulas add a fixed weight per inch of height above five
#'   feet to a base weight, with separate constants for men and women:
#'   Devine 50 + 2.3 / 45.5 + 2.3, Robinson 52 + 1.9 / 49 + 1.7, Miller
#'   56.2 + 1.41 / 53.1 + 1.36 and Hamwi 48 + 2.7 / 45.5 + 2.2 (male / female,
#'   kg). Patients shorter than five feet get the base weight.
#' @return Ideal body weight in kg.
#' @export
ideal.body.weight = function(x, method = c("devine", "robinson", "miller", "hamwi")) {
  method = match.arg(method)
  inches.over = max(x@height.cm / 2.54 - 60, 0)
  male = x@sex == "male"
  switch(method,
    devine = {
      if (male) {
        50 + 2.3 * inches.over
      } else {
        45.5 + 2.3 * inches.over
      }
    },
    robinson = {
      if (male) {
        52 + 1.9 * inches.over
      } else {
        49 + 1.7 * inches.over
      }
    },
    miller = {
      if (male) {
        56.2 + 1.41 * inches.over
      } else {
        53.1 + 1.36 * inches.over
      }
    },
    hamwi = {
      if (male) {
        48 + 2.7 * inches.over
      } else {
        45.5 + 2.2 * inches.over
      }
    }
  )
}

#' Creatinine clearance
#'
#' @inheritParams body.surface.area
#' @param weight Which weight enters the Cockcroft-Gault formula: the
#'   `"actual"` weight, the `"ideal"` weight, or the `"adjusted"` weight
#'   (ideal plus 40 percent of the excess).
#' @param scr.floor Lowest serum creatinine used in the formula. Low values
#'   in frail or elderly patients overestimate clearance, so many protocols
#'   round them up; use `0` to disable.
#' @return Estimated creatinine clearance in mL/min.
#' @export
creatinine.clearance = function(x, weight = c("actual", "ideal", "adjusted"), scr.floor = 0) {
  weight = match.arg(weight)
  if (!is.numeric(scr.floor) || length(scr.floor) != 1L || scr.floor < 0) {
    stop("`scr.floor` must be a single non-negative number.", call. = FALSE)
  }
  scr = max(x@scr.mgdl, scr.floor)

  ibw = ideal.body.weight(x)
  wt = switch(weight,
    actual = x@weight.kg,
    ideal = min(ibw, x@weight.kg),
    adjusted = ibw + 0.4 * max(x@weight.kg - ibw, 0)
  )
  crcl = (140 - x@age.y) * wt / (72 * scr)
  if (x@sex == "female") {
    crcl = crcl * 0.85
  }
  crcl
}

#' Print a patient profile
#'
#' @param object A `Patient` object.
#' @return `object`, invisibly.
#' @keywords internal
show.patient = function(object) {
  crcl = creatinine.clearance(object)
  renal = if (crcl >= 90) {
    "normal"
  } else if (crcl >= 60) {
    "mildly reduced"
  } else if (crcl >= 30) {
    "moderately reduced"
  } else {
    "severely reduced"
  }
  cat(sprintf("<Patient> %s, %s, %g y\n", object@id, object@sex, object@age.y))
  cat(sprintf("  weight: %g kg (ideal %.1f kg)\n", object@weight.kg, ideal.body.weight(object)))
  cat(sprintf("  height: %g cm (BSA %.2f m2)\n", object@height.cm, body.surface.area(object)))
  cat(sprintf("  CrCl:   %.0f mL/min (%s)\n", crcl, renal))
  invisible(object)
}

setMethod("show", "Patient", function(object) show.patient(object))
