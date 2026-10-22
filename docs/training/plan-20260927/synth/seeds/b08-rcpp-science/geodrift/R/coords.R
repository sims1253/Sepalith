#' Validate longitude and latitude vectors
#'
#' Checks types, lengths and ranges of decimal-degree coordinates before they
#' are converted and passed to compiled code.
#'
#' @param lon Numeric vector of longitudes in decimal degrees.
#' @param lat Numeric vector of latitudes in decimal degrees.
#' @return `TRUE`, invisibly, if the coordinates are valid.
#' @keywords internal
validateLonLat = function(lon, lat) {
  if (!is.numeric(lon) || !is.numeric(lat)) {
    stop("`lon` and `lat` must be numeric.", call. = FALSE)
  }
  if (length(lon) != length(lat)) {
    stop("`lon` and `lat` must have the same length.", call. = FALSE)
  }
  if (any(abs(lat) > 90, na.rm = TRUE)) {
    stop("`lat` must lie between -90 and 90.", call. = FALSE)
  }
  if (any(abs(lon) > 180, na.rm = TRUE)) {
    stop("`lon` must lie between -180 and 180.", call. = FALSE)
  }
  invisible(TRUE)
}

#' Convert degrees to radians
#'
#' @param deg Numeric vector of angles in degrees.
#' @return A numeric vector of angles in radians.
#' @keywords internal
toRadians = function(deg) {
  deg * pi / 180
}

#' Convert radians to degrees
#'
#' @param rad Numeric vector of angles in radians.
#' @return A numeric vector of angles in degrees.
#' @keywords internal
toDegrees = function(rad) {
  rad * 180 / pi
}

#' Great-circle distance between points
#'
#' Computes haversine distances between paired origin and destination points.
#' Shorter vectors are recycled.
#'
#' @param lon1,lat1 Origin coordinates in decimal degrees.
#' @param lon2,lat2 Destination coordinates in decimal degrees.
#' @param unit Distance unit, one of `"km"`, `"m"` or `"mi"`.
#' @return A numeric vector of distances in `unit`.
#' @export
#' @examples
#' haversineDistance(13.40, 52.52, 2.35, 48.86)
haversineDistance = function(lon1, lat1, lon2, lat2, unit = c("km", "m", "mi")) {
  unit = match.arg(unit)
  n = max(length(lon1), length(lon2))
  lon1 = rep_len(lon1, n)
  lat1 = rep_len(lat1, n)
  lon2 = rep_len(lon2, n)
  lat2 = rep_len(lat2, n)
  validateLonLat(lon1, lat1)
  validateLonLat(lon2, lat2)
  radius = switch(unit,
    km = 6371.0088,
    m = 6371008.8,
    mi = 3958.7613
  )
  haversineCpp(toRadians(lat1), toRadians(lon1), toRadians(lat2), toRadians(lon2), radius)
}

#' Initial bearing between points
#'
#' @param lon1,lat1 Origin coordinates in decimal degrees.
#' @param lon2,lat2 Destination coordinates in decimal degrees.
#' @return A numeric vector of compass bearings in degrees, from 0 to 360.
#' @export
initialBearing = function(lon1, lat1, lon2, lat2) {
  validateLonLat(lon1, lat1)
  validateLonLat(lon2, lat2)
  phi1 = toRadians(lat1)
  phi2 = toRadians(lat2)
  dLambda = toRadians(lon2 - lon1)
  y = sin(dLambda) * cos(phi2)
  x = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dLambda)
  (toDegrees(atan2(y, x)) + 360) %% 360
}

#' Destination point from a start, bearing and distance
#'
#' @param lon,lat Start coordinates in decimal degrees.
#' @param bearing Compass bearing in degrees.
#' @param distanceKm Distance to travel in kilometres.
#' @return A data frame with columns `lon` and `lat` in decimal degrees.
#' @export
destinationPoint = function(lon, lat, bearing, distanceKm) {
  validateLonLat(lon, lat)
  delta = distanceKm / 6371.0088
  theta = toRadians(bearing)
  phi1 = toRadians(lat)
  lambda1 = toRadians(lon)
  phi2 = asin(sin(phi1) * cos(delta) + cos(phi1) * sin(delta) * cos(theta))
  lambda2 = lambda1 + atan2(
    sin(theta) * sin(delta) * cos(phi1),
    cos(delta) - sin(phi1) * sin(phi2)
  )
  data.frame(
    lon = (toDegrees(lambda2) + 540) %% 360 - 180,
    lat = toDegrees(phi2)
  )
}

#' Snap points to their nearest reference site
#'
#' @param points A data frame of query points with `lon` and `lat` columns.
#' @param sites A data frame of reference sites with `lon` and `lat` columns.
#' @param idCol Name of the column in `sites` that identifies each site.
#' @return `points` with added columns `site` and `distKm`.
#' @export
nearestSite = function(points, sites, idCol = "site_id") {
  if (!idCol %in% names(sites)) {
    stop(sprintf("Column '%s' not found in `sites`.", idCol), call. = FALSE)
  }
  validateLonLat(points$lon, points$lat)
  validateLonLat(sites$lon, sites$lat)
  idx = nearestIndexCpp(
    toRadians(points$lat), toRadians(points$lon),
    toRadians(sites$lat), toRadians(sites$lon)
  )
  points$site = sites[[idCol]][idx]
  points$distKm = haversineDistance(points$lon, points$lat, sites$lon[idx], sites$lat[idx])
  points
}

#' Convert a coordinate table to an sf object
#'
#' @param data A data frame with longitude and latitude columns.
#' @param lonCol,latCol Names of the longitude and latitude columns.
#' @param crs Coordinate reference system passed to [sf::st_as_sf()].
#' @return An `sf` object with point geometries.
#' @export
asSfPoints = function(data, lonCol = "lon", latCol = "lat", crs = 4326) {
  if (!requireNamespace("sf", quietly = TRUE)) {
    stop("Package 'sf' is required for asSfPoints().", call. = FALSE)
  }
  missingCols = setdiff(c(lonCol, latCol), names(data))
  if (length(missingCols) > 0) {
    stop("Missing coordinate column(s): ", paste(missingCols, collapse = ", "), call. = FALSE)
  }
  validateLonLat(data[[lonCol]], data[[latCol]])
  keep = stats::complete.cases(data[, c(lonCol, latCol)])
  if (!all(keep)) {
    warning(sprintf("Dropping %d row(s) with missing coordinates.", sum(!keep)), call. = FALSE)
  }
  sf::st_as_sf(data[keep, , drop = FALSE], coords = c(lonCol, latCol), crs = crs, remove = FALSE)
}
