#' Check a Station object
#'
#' @description
#' Validity method for `Station`. Every slot must hold exactly one value;
#' after that `id`, `name`, `lat`, `lon`, `datum` and `active` are checked in
#' turn.
#'
#' @param object A `Station` object.
#' @return `TRUE` or a character vector of problems.
#' @keywords internal
validStation <- function(object) {
  slots <- c("id", "name", "lat", "lon", "datum", "active")
  lens <- vapply(slots, function(s) length(slot(object, s)), integer(1))
  if (any(lens != 1L)) {
    return(sprintf("`%s` must have length 1.", slots[lens != 1L]))
  }
  problems <- character()

  # Check `id`
  if (is.na(object@id) || !grepl("^[A-Z]{2}[0-9]{3}$", object@id)) {
    problems <- c(problems, "`id` must be two capitals followed by three digits.")
  }

  # Check `name`
  if (is.na(object@name) || !nzchar(trimws(object@name))) {
    problems <- c(problems, "`name` must be a non-empty string.")
  }

  # Check `lat`
  if (is.na(object@lat) || abs(object@lat) > 90) {
    problems <- c(problems, "`lat` must lie between -90 and 90.")
  }

  # Check `lon`
  if (is.na(object@lon) || abs(object@lon) > 180) {
    problems <- c(problems, "`lon` must lie between -180 and 180.")
  }

  # Check `datum`
  if (is.na(object@datum) || !nzchar(object@datum)) {
    problems <- c(problems, "`datum` must be a non-empty string.")
  }

  # Check `active`
  if (is.na(object@active)) {
    problems <- c(problems, "`active` must be TRUE or FALSE.")
  }

  if (length(problems)) problems else TRUE
}

#' A tide gauge station
#'
#' @slot id Station code, such as `"NL042"`.
#' @slot name Human-readable station name.
#' @slot lat,lon Position in decimal degrees.
#' @slot datum Vertical datum used by the gauge.
#' @slot active Whether the gauge currently reports.
#' @export
setClass(
  "Station",
  slots = c(
    id = "character",
    name = "character",
    lat = "numeric",
    lon = "numeric",
    datum = "character",
    active = "logical"
  ),
  validity = validStation
)

#' Create a Station
#'
#' @param id Station code.
#' @param name Station name.
#' @param lat,lon Position in decimal degrees.
#' @param datum Vertical datum.
#' @param active Whether the gauge currently reports.
#' @return A `Station` object.
#' @export
station <- function(id, name, lat, lon, datum = "LAT", active = TRUE) {
  new(
    "Station",
    id = toupper(id),
    name = name,
    lat = as.numeric(lat),
    lon = as.numeric(lon),
    datum = datum,
    active = as.logical(active)
  )
}

#' Station setters
#'
#' @description
#' Setters behind the `stationName<-`, `stationLat<-` and `stationLon<-`
#' replacement methods. Each checks the type and length of `value`, tidies
#' it, checks its range, stores it and revalidates.
#'
#' @param x A `Station` object.
#' @param value The new value.
#' @return The modified `Station` object.
#' @name stationSetters
#' @keywords internal
setStationName <- function(x, value) {
  if (!is.character(value)) {
    stop(sprintf("`%s` must be a character value.", "name"), call. = FALSE)
  }
  if (length(value) != 1L || is.na(value)) {
    stop(sprintf("`%s` must be a single non-missing value.", "name"), call. = FALSE)
  }

  # Collapse runs of whitespace left over from fixed-width station lists
  value <- gsub("[[:space:]]+", " ", trimws(value))
  if (!nzchar(value)) {
    stop("`name` must not be empty.", call. = FALSE)
  }
  x@name <- value
  validObject(x)
  x
}

#' @rdname stationSetters
setStationLat <- function(x, value) {
  if (!is.numeric(value)) {
    stop(sprintf("`%s` must be a numeric value.", "lat"), call. = FALSE)
  }
  if (length(value) != 1L || is.na(value)) {
    stop(sprintf("`%s` must be a single non-missing value.", "lat"), call. = FALSE)
  }

  # Positions are stored to roughly 0.1 m
  value <- round(value, 6)
  if (abs(value) > 90) {
    stop("`lat` must lie between -90 and 90.", call. = FALSE)
  }
  x@lat <- value
  validObject(x)
  x
}

#' @rdname stationSetters
setStationLon <- function(x, value) {
  if (!is.numeric(value)) {
    stop(sprintf("`%s` must be a numeric value.", "lon"), call. = FALSE)
  }
  if (length(value) != 1L || is.na(value)) {
    stop(sprintf("`%s` must be a single non-missing value.", "lon"), call. = FALSE)
  }

  # Positions are stored to roughly 0.1 m
  value <- round(value, 6)
  if (abs(value) > 180) {
    stop("`lon` must lie between -180 and 180.", call. = FALSE)
  }
  x@lon <- value
  validObject(x)
  x
}

setGeneric("stationName<-", function(x, value) standardGeneric("stationName<-"))
setGeneric("stationLat<-", function(x, value) standardGeneric("stationLat<-"))
setGeneric("stationLon<-", function(x, value) standardGeneric("stationLon<-"))

#' @export
setReplaceMethod("stationName", "Station", function(x, value) setStationName(x, value))

#' @export
setReplaceMethod("stationLat", "Station", function(x, value) setStationLat(x, value))

#' @export
setReplaceMethod("stationLon", "Station", function(x, value) setStationLon(x, value))

#' Convert between Station objects and lists
#'
#' @description
#' `stationToList()` encodes a station as a plain named list with one element
#' per slot; `listToStation()` decodes such a list back into a `Station`.
#'
#' @param from The object to convert.
#' @return A named list, or a `Station` object.
#' @keywords internal
stationToList <- function(from) {
  list(
    id = from@id,
    name = from@name,
    lat = from@lat,
    lon = from@lon,
    datum = from@datum,
    active = from@active
  )
}

#' @rdname stationToList
#' @keywords internal
listToStation <- function(from) {
  # Decode `id`
  id <- from[["id"]]
  if (is.null(id)) {
    stop(sprintf("Element `%s` is required.", "id"), call. = FALSE)
  }

  # Decode `name`
  name <- from[["name"]]
  if (is.null(name)) {
    stop(sprintf("Element `%s` is required.", "name"), call. = FALSE)
  }

  # Decode `lat`
  lat <- from[["lat"]]
  if (is.null(lat)) {
    stop(sprintf("Element `%s` is required.", "lat"), call. = FALSE)
  }

  # Decode `lon`
  lon <- from[["lon"]]
  if (is.null(lon)) {
    stop(sprintf("Element `%s` is required.", "lon"), call. = FALSE)
  }

  # Decode `datum`
  datum <- from[["datum"]]
  if (is.null(datum)) {
    datum <- "LAT"
  }

  # Decode `active`
  active <- from[["active"]]
  if (is.null(active)) {
    active <- TRUE
  }

  station(id, name, lat, lon, datum = datum, active = active)
}

setAs("Station", "list", function(from) stationToList(from))
setAs("list", "Station", function(from) listToStation(from))
setAs("Station", "data.frame", function(from) {
  as.data.frame(stationToList(from), stringsAsFactors = FALSE)
})

#' Print a Station
#'
#' @param object A `Station` object.
#' @param digits Number of decimals used for the position.
#' @return `object`, invisibly.
#' @keywords internal
showStation <- function(object, digits = 4) {
  ns <- if (object@lat >= 0) "N" else "S"
  ew <- if (object@lon >= 0) "E" else "W"
  status <- if (object@active) "active" else "inactive"
  fmt <- paste0("%.", digits, "f")

  cat(sprintf("<Station> %s %s (%s)\n", object@id, object@name, status))
  cat("  lat:   ", sprintf(fmt, abs(object@lat)), ns, "\n")
  cat("  lon:   ", sprintf(fmt, abs(object@lon)), ew, "\n")
  cat("  datum: ", object@datum, "\n")
  invisible(object)
}

#' @export
setMethod("show", "Station", function(object) showStation(object))
