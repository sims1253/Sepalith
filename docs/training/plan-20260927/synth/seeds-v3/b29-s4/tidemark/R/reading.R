#' Vertical datums known to tidemark
#'
#' @return A named numeric vector of datum offsets in metres above LAT.
#' @keywords internal
gaugeDatums <- function() {
  c(LAT = 0, CD = 0.05, NAP = 0.98, MSL = 1.12)
}

# Quality flags written by gauge loggers
qualityFlags <- c("ok", "suspect", "bad", "filled")

#' Check a GaugeReading object
#'
#' @description
#' Validity method for the `GaugeReading` class. Slots are checked in turn:
#' `station`, `datum`, `time`, `level` and `quality`.
#'
#' @param object A `GaugeReading` object.
#' @return `TRUE` when the object is valid, otherwise a character vector with
#'   one message per problem.
#' @keywords internal
validGaugeReading <- function(object) {
  problems <- character()
  n <- length(object@time)

  # Check `station`
  if (length(object@station) != 1L || is.na(object@station)) {
    problems <- c(problems, "`station` must be a single non-missing string.")
  }

  # Check `datum`
  if (length(object@datum) != 1L || is.na(object@datum)) {
    problems <- c(problems, "`datum` must be a single non-missing string.")
  } else if (!object@datum %in% names(gaugeDatums())) {
    problems <- c(problems, sprintf("`datum` must be one of %s.", toString(names(gaugeDatums()))))
  }

  # Check `time`
  if (anyNA(object@time) || is.unsorted(object@time, strictly = TRUE)) {
    problems <- c(problems, "`time` must be strictly increasing without NAs.")
  }

  # Check `level`
  if (length(object@level) != n) {
    problems <- c(problems, sprintf("`level` must have length %d, like `time`.", n))
  }

  # Check `quality`
  if (length(object@quality) != n) {
    problems <- c(problems, sprintf("`quality` must have length %d, like `time`.", n))
  } else if (!all(object@quality %in% qualityFlags)) {
    problems <- c(problems, sprintf("`quality` must be one of %s.", toString(qualityFlags)))
  }

  if (length(problems)) problems else TRUE
}

#' Water-level readings from one gauge
#'
#' @slot station Station code.
#' @slot datum Vertical datum the levels refer to.
#' @slot time Observation times in hours since the series epoch.
#' @slot level Water levels in metres.
#' @slot quality Quality flag per reading.
#' @export
setClass(
  "GaugeReading",
  slots = c(
    station = "character",
    datum = "character",
    time = "numeric",
    level = "numeric",
    quality = "character"
  ),
  prototype = list(datum = "LAT"),
  validity = validGaugeReading
)

#' Create gauge readings
#'
#' @description
#' Arguments are checked one by one so that a malformed logger export fails
#' with a message naming the offending argument. Readings are then sorted by
#' time.
#'
#' @param station Station code, such as `"NL042"`.
#' @param time Observation times in hours.
#' @param level Water levels in metres.
#' @param quality Quality flags, recycled to the length of `time`.
#' @param datum Vertical datum, one of the names of [gaugeDatums()].
#' @return A validated `GaugeReading` object.
#' @export
gaugeReading <- function(station, time, level, quality = "ok", datum = "LAT") {
  # Check `station`
  if (!is.character(station) || length(station) != 1L) {
    stop("`station` must be a single string.", call. = FALSE)
  }
  station <- toupper(trimws(station))

  # Check `time`
  if (!is.numeric(time) || anyNA(time)) {
    stop("`time` must be numeric without missing values.", call. = FALSE)
  }
  n <- length(time)

  # Check `level`
  if (!is.numeric(level) || length(level) != n) {
    stop(sprintf("`level` must be numeric with length %d.", n), call. = FALSE)
  }

  # Check `quality`
  if (!is.character(quality) || !length(quality) %in% c(1L, n)) {
    stop(sprintf("`quality` must be character with length 1 or %d.", n), call. = FALSE)
  }
  quality <- rep_len(tolower(quality), n)

  # Check `datum`
  if (!is.character(datum) || length(datum) != 1L) {
    stop("`datum` must be a single string.", call. = FALSE)
  }
  datum <- toupper(datum)

  ord <- order(time)
  new("GaugeReading", station = station, datum = datum, time = as.numeric(time)[ord],
    level = as.numeric(level)[ord], quality = quality[ord])
}

#' Update parts of a reading
#'
#' @description
#' Backs the replacement methods `readingLevel<-`, `readingQuality<-` and
#' `readingDatum<-`. Each supplied argument is checked and stored; changing
#' the datum shifts the levels so they keep describing the same physical
#' heights.
#'
#' @param x A `GaugeReading` object.
#' @param level New levels, one per reading.
#' @param quality New quality flags, one per reading or a single flag.
#' @param datum New datum, one of the names of [gaugeDatums()].
#' @return The modified `GaugeReading` object.
#' @keywords internal
updateReading <- function(x, level, quality, datum) {
  n <- length(x@time)

  if (!missing(level)) {
    level <- as.numeric(level)
    if (length(level) != n) {
      stop(sprintf("`%s` needs %d values, not %d.", "level", n, length(level)), call. = FALSE)
    }
    x@level <- level
  }

  if (!missing(quality)) {
    quality <- tolower(as.character(quality))
    if (length(quality) == 1L) {
      quality <- rep_len(quality, n)
    }
    if (length(quality) != n) {
      stop(sprintf("`%s` needs %d values, not %d.", "quality", n, length(quality)), call. = FALSE)
    }
    x@quality <- quality
  }

  if (!missing(datum)) {
    datum <- toupper(as.character(datum))
    offsets <- gaugeDatums()
    if (length(datum) != 1L || !datum %in% names(offsets)) {
      stop(sprintf("`datum` must be one of %s.", toString(names(offsets))), call. = FALSE)
    }
    x@level <- x@level + offsets[[x@datum]] - offsets[[datum]]
    x@datum <- datum
  }

  validObject(x)
  x
}

setGeneric("readingLevel", function(x) standardGeneric("readingLevel"))
setGeneric("readingLevel<-", function(x, value) standardGeneric("readingLevel<-"))
setGeneric("readingQuality", function(x) standardGeneric("readingQuality"))
setGeneric("readingQuality<-", function(x, value) standardGeneric("readingQuality<-"))
setGeneric("readingDatum", function(x) standardGeneric("readingDatum"))
setGeneric("readingDatum<-", function(x, value) standardGeneric("readingDatum<-"))

#' @rdname readingLevel
#' @export
setMethod("readingLevel", "GaugeReading", function(x) x@level)

#' @rdname readingLevel
#' @export
setReplaceMethod("readingLevel", "GaugeReading", function(x, value) {
  updateReading(x, level = value)
})

#' @rdname readingQuality
#' @export
setMethod("readingQuality", "GaugeReading", function(x) x@quality)

#' @rdname readingQuality
#' @export
setReplaceMethod("readingQuality", "GaugeReading", function(x, value) {
  updateReading(x, quality = value)
})

#' @rdname readingDatum
#' @export
setMethod("readingDatum", "GaugeReading", function(x) x@datum)

#' @rdname readingDatum
#' @export
setReplaceMethod("readingDatum", "GaugeReading", function(x, value) {
  updateReading(x, datum = value)
})

#' Subset gauge readings
#'
#' @param x A `GaugeReading` object.
#' @param i A logical or numeric index into the readings.
#' @return A `GaugeReading` object with the selected readings, still in time
#'   order.
#' @keywords internal
subsetReading <- function(x, i) {
  if (is.logical(i) && length(i) != length(x@time)) {
    stop("A logical index must have one value per reading.", call. = FALSE)
  }
  keep <- sort(seq_along(x@time)[i])
  x@time <- x@time[keep]
  x@level <- x@level[keep]
  x@quality <- x@quality[keep]
  validObject(x)
  x
}

#' @export
setMethod("[", "GaugeReading", function(x, i, ...) subsetReading(x, i))

#' Automatic quality control of gauge readings
#'
#' @description
#' Runs three checks in turn and downgrades the quality flag of every reading
#' that fails one: a `range` check against plausible levels, a `spike` check
#' against the jump from the previous reading, and a `flat` check for runs of
#' identical levels that betray a stuck float. Readings already flagged
#' `"bad"` are left alone.
#'
#' @param x A `GaugeReading` object.
#' @param range Plausible levels in metres, as `c(low, high)`.
#' @param spike Largest plausible jump between consecutive readings, in
#'   metres.
#' @param flat Shortest run of identical levels treated as a stuck float.
#' @return `x` with updated quality flags.
#' @export
flagReadings <- function(x, range = c(-3, 6), spike = 0.5, flat = 6L) {
  quality <- x@quality
  lev <- x@level
  n <- length(lev)

  # Check `range`: impossible levels are bad
  outside <- lev < range[1] | lev > range[2]
  quality[outside & quality != "bad"] <- "bad"

  # Check `spike`: a sudden jump is suspect
  jump <- c(0, abs(diff(lev)))
  spiked <- jump > spike
  quality[spiked & quality == "ok"] <- "suspect"

  # Check `flat`: long runs of one value are suspect
  runs <- rle(lev)
  stuck <- rep(runs$lengths >= flat, runs$lengths)
  quality[stuck & quality == "ok"] <- "suspect"

  if (length(quality) != n) {
    stop("Internal error: quality flags lost their alignment.", call. = FALSE)
  }
  x@quality <- quality
  validObject(x)
  x
}

#' Summarise gauge readings
#'
#' @param object A `GaugeReading` object.
#' @param flags Quality flags counted as usable.
#' @return A list with elements `station`, `n`, `usable`, `mean`, `low`,
#'   `high` and `range`.
#' @export
summariseReading <- function(object, flags = c("ok", "filled")) {
  keep <- object@quality %in% flags
  lev <- object@level[keep]
  if (length(lev) == 0L) {
    lev <- NA_real_
  }
  low <- min(lev)
  high <- max(lev)
  list(
    station = object@station,
    n = length(object@level),
    usable = sum(keep),
    mean = mean(lev),
    low = low,
    high = high,
    range = high - low
  )
}

#' Print a GaugeReading
#'
#' @param object A `GaugeReading` object.
#' @return `object`, invisibly.
#' @keywords internal
showReading <- function(object) {
  s <- summariseReading(object)
  cat(sprintf("<GaugeReading> %s (%s), %d readings\n", s$station, object@datum, s$n))
  if (s$n == 0L) {
    return(invisible(object))
  }
  cat(sprintf("  span:   %g to %g h\n", object@time[1], object@time[s$n]))
  cat(sprintf("  usable: %d\n", s$usable))
  cat(sprintf("  mean:   %.2f m\n", s$mean))
  cat(sprintf("  low:    %.2f m\n", s$low))
  cat(sprintf("  high:   %.2f m\n", s$high))
  cat(sprintf("  range:  %.2f m\n", s$range))

  # Only mention flags that actually occur
  flagged <- table(factor(object@quality, levels = qualityFlags))
  flagged <- flagged[flagged > 0L & names(flagged) != "ok"]
  if (length(flagged)) {
    cat("  flags: ", paste(names(flagged), flagged, sep = "=", collapse = ", "), "\n")
  }
  invisible(object)
}

#' @export
setMethod("show", "GaugeReading", function(object) showReading(object))
