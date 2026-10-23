#' Height units understood by TideSeries
#'
#' @return A named numeric vector giving the number of metres per unit.
#' @keywords internal
heightUnits <- function() {
  c(m = 1, cm = 0.01, ft = 0.3048)
}

#' Check a TideSeries object
#'
#' @description
#' Validity method for `TideSeries`. The slots `station`, `units`, `times`
#' and `heights` are checked in that order.
#'
#' @param object A `TideSeries` object.
#' @return `TRUE` or a character vector of problems.
#' @keywords internal
validTideSeries <- function(object) {
  problems <- character()

  # Check `station`
  if (length(object@station) != 1L || !nzchar(object@station)) {
    problems <- c(problems, "`station` must be a single non-empty string.")
  }

  # Check `units`
  if (length(object@units) != 1L || !object@units %in% names(heightUnits())) {
    problems <- c(problems, sprintf("`units` must be one of %s.", toString(names(heightUnits()))))
  }

  # Check `times`
  if (anyNA(object@times) || is.unsorted(object@times, strictly = TRUE)) {
    problems <- c(problems, "`times` must be strictly increasing without NAs.")
  }

  # Check `heights`
  if (length(object@heights) != length(object@times)) {
    problems <- c(problems, "`heights` must have one value per element of `times`.")
  } else if (any(!is.finite(object@heights))) {
    problems <- c(problems, "`heights` must be finite.")
  }

  if (length(problems) == 0L) TRUE else problems
}

#' Predicted tide heights
#'
#' @slot station Station code.
#' @slot units Height units, one of the names of [heightUnits()].
#' @slot times Prediction times in hours.
#' @slot heights Predicted heights in `units`.
#' @export
setClass(
  "TideSeries",
  slots = c(station = "character", units = "character", times = "numeric", heights = "numeric"),
  prototype = list(station = "unknown", units = "m"),
  validity = validTideSeries
)

#' Create a TideSeries
#'
#' @param station Station code.
#' @param times Prediction times in hours.
#' @param heights Predicted heights.
#' @param units Height units, one of the names of [heightUnits()].
#' @return A `TideSeries` object, sorted by time.
#' @export
tideSeries <- function(station, times, heights, units = "m") {
  if (!is.numeric(times) || !is.numeric(heights)) {
    stop("`times` and `heights` must be numeric.", call. = FALSE)
  }
  if (length(times) != length(heights)) {
    stop("`times` and `heights` must have the same length.", call. = FALSE)
  }

  # Predictions are often generated backwards from a reference time
  ord <- order(times)
  times <- times[ord]
  heights <- heights[ord]
  new(
    "TideSeries",
    station = as.character(station),
    units = match.arg(units, names(heightUnits())),
    times = as.numeric(times),
    heights = as.numeric(heights)
  )
}

#' Convert a TideSeries to other height units
#'
#' @param x A `TideSeries` object.
#' @param to Target units, one of the names of [heightUnits()].
#' @return `x` with `heights` rescaled and `units` set to `to`.
#' @export
convertHeights <- function(x, to = c("m", "cm", "ft")) {
  to <- match.arg(to)
  if (identical(x@units, to)) {
    return(x)
  }
  factors <- heightUnits()
  x@heights <- x@heights * factors[[x@units]] / factors[[to]]
  x@units <- to
  x
}

#' Arithmetic between a TideSeries and a number
#'
#' @description
#' Helper behind the mixed `Arith` methods. The series may be either operand:
#' `series op number` and `number op series` are handled by mirrored
#' branches. Only `+`, `-`, `*` and `/` are supported.
#'
#' @param e1,e2 The operands; exactly one of them is a `TideSeries`.
#' @param op The operator name, as found in `.Generic`.
#' @return A `TideSeries` with transformed heights.
#' @keywords internal
seriesArith <- function(e1, e2, op) {
  if (!op %in% c("+", "-", "*", "/")) {
    stop(sprintf("Operator `%s` is not supported for TideSeries.", op), call. = FALSE)
  }

  if (is(e1, "TideSeries")) {
    # series op number
    if (length(e2) != 1L && length(e2) != length(e1@heights)) {
      stop("The numeric operand must have length 1 or match the series.", call. = FALSE)
    }
    e1@heights <- switch(op,
      "+" = e1@heights + e2,
      "-" = e1@heights - e2,
      "*" = e1@heights * e2,
      "/" = e1@heights / e2
    )
    validObject(e1)
    e1
  } else {
    # number op series
    if (length(e1) != 1L && length(e1) != length(e2@heights)) {
      stop("The numeric operand must have length 1 or match the series.", call. = FALSE)
    }
    e2@heights <- switch(op,
      "+" = e1 + e2@heights,
      "-" = e1 - e2@heights,
      "*" = e1 * e2@heights,
      "/" = e1 / e2@heights
    )
    validObject(e2)
    e2
  }
}

#' Arithmetic between two TideSeries
#'
#' @param e1,e2 `TideSeries` objects on the same time grid.
#' @param op The operator name, as found in `.Generic`.
#' @return A `TideSeries` in the units of `e1`.
#' @keywords internal
seriesSeriesOp <- function(e1, e2, op) {
  if (!identical(e1@times, e2@times)) {
    stop("Both series must share the same `times`.", call. = FALSE)
  }
  e2 <- convertHeights(e2, e1@units)
  if (!op %in% c("+", "-")) {
    stop(sprintf("Operator `%s` is not supported for TideSeries.", op), call. = FALSE)
  }
  station <- if (identical(e1@station, e2@station)) e1@station else "combined"
  tideSeries(station, e1@times, seriesArith(e1@heights, e2, op)@heights, e1@units)
}

#' @export
setMethod("Arith", signature("TideSeries", "numeric"), function(e1, e2) {
  seriesArith(e1, e2, .Generic)
})

#' @export
setMethod("Arith", signature("numeric", "TideSeries"), function(e1, e2) {
  seriesArith(e1, e2, .Generic)
})

#' @export
setMethod("Arith", signature("TideSeries", "TideSeries"), function(e1, e2) {
  seriesSeriesOp(e1, e2, .Generic)
})

#' @export
setMethod("Compare", signature("TideSeries", "numeric"), function(e1, e2) {
  callGeneric(e1@heights, e2)
})

#' @export
setMethod("Compare", signature("numeric", "TideSeries"), function(e1, e2) {
  callGeneric(e1, e2@heights)
})

#' High and low waters of a TideSeries
#'
#' @description
#' A high water is a prediction higher than its predecessor and not lower
#' than its successor; a low water mirrors that definition.
#'
#' @param x A `TideSeries` object.
#' @return A data frame with columns `time`, `height` and `type` (`"high"` or
#'   `"low"`), ordered by time.
#' @export
tideExtremes <- function(x) {
  h <- x@heights
  n <- length(h)
  if (n < 3L) {
    return(data.frame(time = numeric(), height = numeric(), type = character()))
  }
  before <- h[2:(n - 1)] - h[1:(n - 2)]
  after <- h[3:n] - h[2:(n - 1)]

  # High waters: rising into the point, not rising out of it
  high <- which(before > 0 & after <= 0) + 1L
  # Low waters: falling into the point, not falling out of it
  low <- which(before < 0 & after >= 0) + 1L

  idx <- c(high, low)
  out <- data.frame(
    time = x@times[idx],
    height = h[idx],
    type = rep(c("high", "low"), c(length(high), length(low)))
  )
  out <- out[order(out$time), , drop = FALSE]
  rownames(out) <- NULL
  out
}

#' Print a TideSeries
#'
#' @param object A `TideSeries` object.
#' @param digits Number of decimals used for heights.
#' @return `object`, invisibly.
#' @keywords internal
showTideSeries <- function(object, digits = 2) {
  n <- length(object@times)
  cat(sprintf("<TideSeries> %s, %d predictions in %s\n", object@station, n, object@units))
  if (n == 0L) {
    return(invisible(object))
  }
  fmt <- paste0("%.", digits, "f")
  ext <- tideExtremes(object)

  cat("  from:", format(min(object@times)), "h\n")
  cat("  to:  ", format(max(object@times)), "h\n")
  cat("  low: ", sprintf(fmt, min(object@heights)), object@units, "\n")
  cat("  high:", sprintf(fmt, max(object@heights)), object@units, "\n")
  cat("  tides:", sum(ext$type == "high"), "high,", sum(ext$type == "low"), "low\n")
  invisible(object)
}

#' @export
setMethod("show", "TideSeries", function(object) showTideSeries(object))
