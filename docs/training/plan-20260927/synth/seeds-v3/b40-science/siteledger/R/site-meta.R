#' Validate site metadata
#'
#' Checks each field in turn and stops at the first problem.
#'
#' @param id Site identifier, a single non-empty string.
#' @param lat Latitude in decimal degrees, between -90 and 90.
#' @param lon Longitude in decimal degrees, between -180 and 180.
#' @param elevation Elevation in metres, between -500 and 9000.
#' @param start Date the site started reporting.
#' @param end Date the site stopped reporting, or `NA` if still active.
#' @keywords internal
#' @return `TRUE`, invisibly.
site.validate <- function(id, lat, lon, elevation, start, end = NA) {
  # Validate `id`
  if (!is.character(id) || length(id) != 1L || is.na(id) || !nzchar(id)) {
    stop("`id` must be a single non-empty string.", call. = FALSE)
  }

  # Validate `lat`
  if (!is.numeric(lat) || length(lat) != 1L || is.na(lat)) {
    stop(sprintf("`lat` of site `%s` must be a single number.", id), call. = FALSE)
  }
  if (lat < -90 || lat > 90) {
    stop(sprintf("`lat` of site `%s` must be between %g and %g.", id, -90, 90), call. = FALSE)
  }

  # Validate `lon`
  if (!is.numeric(lon) || length(lon) != 1L || is.na(lon)) {
    stop(sprintf("`lon` of site `%s` must be a single number.", id), call. = FALSE)
  }
  if (lon < -180 || lon > 180) {
    stop(sprintf("`lon` of site `%s` must be between %g and %g.", id, -180, 180),
         call. = FALSE)
  }

  # Validate `elevation`
  if (!is.numeric(elevation) || length(elevation) != 1L || is.na(elevation)) {
    stop(sprintf("`elevation` of site `%s` must be a single number.", id), call. = FALSE)
  }
  if (elevation < -500 || elevation > 9000) {
    stop(sprintf("`elevation` of site `%s` must be between %g and %g.", id, -500, 9000),
         call. = FALSE)
  }

  # Validate `start` and `end`
  if (!inherits(start, "Date") || length(start) != 1L || is.na(start)) {
    stop(sprintf("`start` of site `%s` must be a single Date.", id), call. = FALSE)
  }
  if (!is.na(end) && (!inherits(end, "Date") || end < start)) {
    stop(sprintf("`end` of site `%s` must be a Date not before `start`.", id), call. = FALSE)
  }
  invisible(TRUE)
}

#' Build a site metadata table
#'
#' @param id,lat,lon,elevation,start,end Vectors of equal length with one
#'   element per site; see [site.validate()].
#' @export
#' @return A data frame with columns `id`, `lat`, `lon`, `elevation`,
#'   `start`, `end` and `active`, one row per site.
site.register <- function(id, lat, lon, elevation, start, end = as.Date(NA)) {
  n <- length(id)
  end <- rep_len(end, n)
  for (i in seq_len(n)) {
    site.validate(id[i], lat[i], lon[i], elevation[i], start[i], end[i])
  }
  if (anyDuplicated(id)) {
    stop(sprintf("Duplicated site ids: %s.", paste(unique(id[duplicated(id)]),
                                                    collapse = ", ")), call. = FALSE)
  }
  data.frame(
    id = id, lat = lat, lon = lon, elevation = elevation,
    start = start, end = end, active = is.na(end),
    stringsAsFactors = FALSE
  )
}

#' Great-circle distance between points
#'
#' @param lat1,lon1 Coordinates of the first points, decimal degrees.
#' @param lat2,lon2 Coordinates of the second points, decimal degrees.
#' @param unit `"km"`, `"m"`, `"mi"` (statute miles) or `"nmi"`
#'   (nautical miles).
#' @export
#' @return A numeric vector of distances in `unit`.
site.distance <- function(lat1, lon1, lat2, lon2, unit = c("km", "m", "mi", "nmi")) {
  unit <- match.arg(unit)
  km.per.unit <- c(km = 1, m = 1 / 1000, mi = 1.609344, nmi = 1.852)
  radius.km <- 6371.0088
  rad <- pi / 180
  dlat <- (lat2 - lat1) * rad
  dlon <- (lon2 - lon1) * rad
  a <- sin(dlat / 2)^2 + cos(lat1 * rad) * cos(lat2 * rad) * sin(dlon / 2)^2
  km <- 2 * radius.km * asin(pmin(1, sqrt(a)))
  km / km.per.unit[[unit]]
}

#' Bounding box of a set of sites
#'
#' @param meta A site metadata table from [site.register()].
#' @param pad Margin in degrees added on every side.
#' @export
#' @return A named numeric vector with elements `south`, `north`, `west`
#'   and `east`, clipped to valid coordinates.
site.bbox <- function(meta, pad = 0) {
  if (nrow(meta) == 0L) {
    stop("`meta` has no sites.", call. = FALSE)
  }

  # Latitude
  south <- max(min(meta$lat) - pad, -90)
  north <- min(max(meta$lat) + pad, 90)

  # Longitude
  west <- max(min(meta$lon) - pad, -180)
  east <- min(max(meta$lon) + pad, 180)

  c(south = south, north = north, west = west, east = east)
}

#' Sites nearest to a point
#'
#' @inheritParams site.bbox
#' @param lat,lon Coordinates of the point, decimal degrees.
#' @param n Number of sites to return.
#' @param active.only Drop sites that have stopped reporting.
#' @export
#' @return The `n` nearest rows of `meta`, with an added `distance_km`
#'   column, closest first.
site.nearest <- function(meta, lat, lon, n = 1L, active.only = TRUE) {
  if (active.only) {
    meta <- meta[meta$active, , drop = FALSE]
  }
  if (nrow(meta) == 0L) {
    stop("`meta` has no sites.", call. = FALSE)
  }
  meta$distance_km <- site.distance(lat, lon, meta$lat, meta$lon)
  meta <- meta[order(meta$distance_km), , drop = FALSE]
  rownames(meta) <- NULL
  utils::head(meta, n)
}

#' One-line descriptions of sites
#'
#' @inheritParams site.bbox
#' @param digits Decimal places for the coordinates.
#' @export
#' @return A character vector with one line per site.
site.describe <- function(meta, digits = 3) {
  lat <- sprintf(paste0("%.", digits, "f%s"), abs(meta$lat), ifelse(meta$lat < 0, "S", "N"))
  lon <- sprintf(paste0("%.", digits, "f%s"), abs(meta$lon), ifelse(meta$lon < 0, "W", "E"))
  elev <- sprintf("%.0f m", meta$elevation)
  since <- format(meta$start, "%Y-%m-%d")
  until <- ifelse(meta$active, "now", format(meta$end, "%Y-%m-%d"))
  sprintf("%s: %s %s, %s, %s to %s", meta$id, lat, lon, elev, since, until)
}
