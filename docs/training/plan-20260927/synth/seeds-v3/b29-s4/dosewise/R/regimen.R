#' Validate a Regimen object
#'
#' @param object A `Regimen` object.
#' @details Every slot must hold a single value. The slots `drug`, `amount`,
#'   `interval.h`, `n.doses`, `start.h` and `route` are then checked in turn.
#' @return `TRUE` or a character vector of problems.
#' @keywords internal
valid.regimen = function(object) {
  fields = c("drug", "amount", "interval.h", "n.doses", "start.h", "route")
  lens = vapply(fields, function(f) length(slot(object, f)), integer(1))
  if (any(lens != 1L)) {
    return(sprintf("`%s` must be a single value.", fields[lens != 1L]))
  }
  problems = character()

  # Check `drug`
  if (is.na(object@drug) || !nzchar(object@drug)) {
    problems = c(problems, "`drug` must be a non-empty string.")
  }

  # Check `amount`
  if (is.na(object@amount) || object@amount <= 0) {
    problems = c(problems, "`amount` must be a positive number.")
  }

  # Check `interval.h`
  if (is.na(object@interval.h) || object@interval.h <= 0) {
    problems = c(problems, "`interval.h` must be a positive number.")
  }

  # Check `n.doses`
  if (is.na(object@n.doses) || object@n.doses < 1L) {
    problems = c(problems, "`n.doses` must be a positive integer.")
  }

  # Check `start.h`
  if (is.na(object@start.h) || object@start.h < 0) {
    problems = c(problems, "`start.h` must be a non-negative number.")
  }

  # Check `route`
  if (is.na(object@route) || !object@route %in% c("oral", "iv", "sc", "im")) {
    problems = c(problems, "`route` must be one of oral, iv, sc, im.")
  }

  if (length(problems)) problems else TRUE
}

#' A fixed dosing regimen
#'
#' @slot drug Drug name.
#' @slot amount Amount per dose in mg.
#' @slot interval.h Hours between doses.
#' @slot n.doses Number of doses.
#' @slot start.h Time of the first dose in hours.
#' @slot route Administration route.
#' @export
setClass(
  "Regimen",
  representation(
    drug = "character",
    amount = "numeric",
    interval.h = "numeric",
    n.doses = "integer",
    start.h = "numeric",
    route = "character"
  ),
  validity = valid.regimen
)

#' Create a dosing regimen
#'
#' @param drug Drug name.
#' @param amount Amount per dose in mg.
#' @param interval.h Hours between doses, or one of the abbreviations
#'   `"qd"`, `"bid"`, `"tid"` and `"qid"`.
#' @param n.doses Number of doses.
#' @param start.h Time of the first dose in hours.
#' @param route Administration route.
#' @return A `Regimen` object.
#' @export
regimen = function(drug, amount, interval.h, n.doses, start.h = 0, route = "oral") {
  # Check `drug`
  if (!is.character(drug) || length(drug) != 1L) {
    stop("`drug` must be a single string.", call. = FALSE)
  }

  # Check `amount`
  if (!is.numeric(amount) || length(amount) != 1L) {
    stop("`amount` must be a single number.", call. = FALSE)
  }

  # Check `interval.h`
  if (is.character(interval.h)) {
    interval.h = switch(tolower(interval.h),
      qd = 24,
      bid = 12,
      tid = 8,
      qid = 6,
      stop("Unknown dosing abbreviation in `interval.h`.", call. = FALSE)
    )
  }
  if (!is.numeric(interval.h) || length(interval.h) != 1L) {
    stop("`interval.h` must be a single number.", call. = FALSE)
  }

  # Check `n.doses`
  if (!is.numeric(n.doses) || length(n.doses) != 1L || n.doses %% 1 != 0) {
    stop("`n.doses` must be a single whole number.", call. = FALSE)
  }

  new(
    "Regimen",
    drug = tolower(drug),
    amount = as.numeric(amount),
    interval.h = as.numeric(interval.h),
    n.doses = as.integer(n.doses),
    start.h = as.numeric(start.h),
    route = route
  )
}

#' Regimen setters
#'
#' @param x A `Regimen` object.
#' @param value The new value.
#' @details These back the replacement methods `amount<-`, `interval<-` and
#'   `n.doses<-`. Each accepts a number or a string holding one, coerces
#'   `value`, stores it and revalidates `x`.
#' @return The modified `Regimen` object.
#' @keywords internal
set.regimen.amount = function(x, value) {
  if (!is.numeric(value) && !is.character(value)) {
    stop(sprintf("`%s` must be numeric.", "amount"), call. = FALSE)
  }
  value = suppressWarnings(as.numeric(value))
  if (length(value) != 1L || is.na(value)) {
    stop(sprintf("`%s` must be a single value.", "amount"), call. = FALSE)
  }
  x@amount = value
  validObject(x)
  x
}

#' @rdname set.regimen.amount
set.regimen.interval = function(x, value) {
  if (!is.numeric(value) && !is.character(value)) {
    stop(sprintf("`%s` must be numeric.", "interval.h"), call. = FALSE)
  }
  value = suppressWarnings(as.numeric(value))
  if (length(value) != 1L || is.na(value)) {
    stop(sprintf("`%s` must be a single value.", "interval.h"), call. = FALSE)
  }
  x@interval.h = value
  validObject(x)
  x
}

#' @rdname set.regimen.amount
set.regimen.n.doses = function(x, value) {
  if (!is.numeric(value) && !is.character(value)) {
    stop(sprintf("`%s` must be numeric.", "n.doses"), call. = FALSE)
  }
  value = suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value)) {
    stop(sprintf("`%s` must be a single value.", "n.doses"), call. = FALSE)
  }
  x@n.doses = value
  validObject(x)
  x
}

setGeneric("amount<-", function(x, value) standardGeneric("amount<-"))
setGeneric("interval<-", function(x, value) standardGeneric("interval<-"))
setGeneric("n.doses<-", function(x, value) standardGeneric("n.doses<-"))

setReplaceMethod("amount", "Regimen", function(x, value) set.regimen.amount(x, value))
setReplaceMethod("interval", "Regimen", function(x, value) set.regimen.interval(x, value))
setReplaceMethod("n.doses", "Regimen", function(x, value) set.regimen.n.doses(x, value))

#' @title Dose schedule of a regimen
#' @param from A `Regimen` object.
#' @return A data frame with columns `dose`, `time.h`, `amount` and
#'   `cumulative`, one row per dose.
#' @export
regimen.schedule = function(from) {
  i = seq_len(from@n.doses)
  data.frame(
    dose = i,
    time.h = from@start.h + (i - 1) * from@interval.h,
    amount = rep(from@amount, from@n.doses),
    cumulative = i * from@amount
  )
}

setAs("Regimen", "data.frame", function(from) regimen.schedule(from))

#' Totals for a regimen
#'
#' @param x A `Regimen` object.
#' @param weight.kg Optional body weight; when given, the per-kg totals are
#'   filled in.
#' @return A list with elements `total.mg`, `daily.mg`, `duration.h`,
#'   `last.dose.h`, `total.mg.kg` and `daily.mg.kg`. The per-kg elements are
#'   `NA` when `weight.kg` is not given.
#' @export
regimen.totals = function(x, weight.kg = NA_real_) {
  total.mg = x@amount * x@n.doses
  daily.mg = x@amount * 24 / x@interval.h
  duration.h = x@n.doses * x@interval.h
  last.dose.h = x@start.h + (x@n.doses - 1) * x@interval.h

  # Per-kg values only make sense with a plausible weight
  if (!is.na(weight.kg) && weight.kg <= 0) {
    stop("`weight.kg` must be positive.", call. = FALSE)
  }
  total.mg.kg = total.mg / weight.kg
  daily.mg.kg = daily.mg / weight.kg

  list(
    total.mg = total.mg,
    daily.mg = daily.mg,
    duration.h = duration.h,
    last.dose.h = last.dose.h,
    total.mg.kg = total.mg.kg,
    daily.mg.kg = daily.mg.kg
  )
}

#' Print a regimen
#'
#' @param object A `Regimen` object.
#' @param n.show Number of doses listed before the rest are summarised.
#' @return `object`, invisibly.
#' @keywords internal
show.regimen = function(object, n.show = 3L) {
  tot = regimen.totals(object)
  cat(sprintf("<Regimen> %s %g mg %s q%gh x %d\n", object@drug, object@amount, object@route,
    object@interval.h, object@n.doses))
  cat(sprintf("  total:    %g mg\n", tot$total.mg))
  cat(sprintf("  daily:    %g mg\n", tot$daily.mg))
  cat(sprintf("  duration: %g h\n", tot$duration.h))

  # Preview the first doses, then summarise the rest
  sched = regimen.schedule(object)
  shown = head(sched, n.show)
  cat(sprintf("  dose %d at %g h\n", shown$dose, shown$time.h), sep = "")
  if (nrow(sched) > n.show) {
    cat(sprintf("  ... %d more, last at %g h\n", nrow(sched) - n.show, tot$last.dose.h))
  }
  invisible(object)
}

setMethod("show", "Regimen", function(object) show.regimen(object))
