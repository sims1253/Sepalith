#' Test whether a layer has a defined CRS
#'
#' @param x An `sf` or `sfc` object.
#' @return `TRUE` or `FALSE`.
#' @family crs checks
#' @export
has_crs <- function(x) {
  !is.na(sf::st_crs(x))
}

#' Test whether a layer uses geographic coordinates
#'
#' @param x An `sf` or `sfc` object.
#' @return `TRUE` or `FALSE`.
#' @family crs checks
#' @export
is_geographic <- function(x) {
  has_crs(x) && isTRUE(sf::st_is_longlat(x))
}

#' Test whether a layer uses projected coordinates
#'
#' @param x An `sf` or `sfc` object.
#' @return `TRUE` or `FALSE`.
#' @family crs checks
#' @export
is_projected <- function(x) {
  has_crs(x) && isFALSE(sf::st_is_longlat(x))
}

#' Assert that a layer is in a projected CRS
#'
#' Stops with an informative message when distances or areas would be computed
#' in degrees.
#'
#' @param x An `sf` or `sfc` object.
#' @param arg Name of the argument used in error messages.
#' @return `x`, invisibly.
#' @family crs checks
#' @export
assert_projected <- function(x, arg = "x") {
  if (!has_crs(x)) {
    stop("`", arg, "` has no coordinate reference system.", call. = FALSE)
  }
  if (!is_projected(x)) {
    stop("`", arg, "` must use a projected CRS, not ", crs_label(x), ".",
         call. = FALSE)
  }
  invisible(x)
}

#' Assert that two layers share a CRS
#'
#' @param x,y `sf` or `sfc` objects to compare.
#' @return `x`, invisibly.
#' @family crs checks
#' @export
assert_same_crs <- function(x, y) {
  if (!has_crs(x) || !has_crs(y)) {
    stop("Both layers must have a coordinate reference system.", call. = FALSE)
  }
  if (sf::st_crs(x) != sf::st_crs(y)) {
    stop("CRS mismatch: ", crs_label(x), " vs ", crs_label(y), ".",
         call. = FALSE)
  }
  invisible(x)
}

#' Build a short human-readable CRS label
#'
#' @param x An `sf` or `sfc` object.
#' @return A single string such as `"EPSG:25832"`.
#' @keywords internal
crs_label <- function(x) {
  crs <- sf::st_crs(x)
  if (is.na(crs)) {
    return("<no CRS>")
  }
  if (!is.na(crs$epsg)) {
    return(paste0("EPSG:", crs$epsg))
  }
  crs$Name %||% "<custom CRS>"
}

#' Pick a UTM zone for the centroid of a layer
#'
#' Computes the WGS 84 / UTM EPSG code whose zone contains the centroid of the
#' layer's bounding box.
#'
#' @param x An `sf` or `sfc` object.
#' @return An integer EPSG code.
#' @export
utm_epsg_for <- function(x) {
  bbox <- x |>
    sf::st_transform(4326) |>
    sf::st_bbox()
  lon <- mean(c(bbox[["xmin"]], bbox[["xmax"]]))
  lat <- mean(c(bbox[["ymin"]], bbox[["ymax"]]))
  zone <- floor((lon + 180) / 6) + 1
  zone <- min(max(zone, 1), 60)
  if (lat >= 0) {
    as.integer(32600 + zone)
  } else {
    as.integer(32700 + zone)
  }
}

#' Ensure a layer is in a projected CRS
#'
#' Returns projected layers unchanged and transforms geographic layers to the
#' local UTM zone, or to an explicit target CRS.
#'
#' @param x An `sf` or `sfc` object.
#' @param target Target coordinate reference system, or `NULL` to choose a UTM zone.
#' @param quiet If `FALSE`, report any transformation with a message.
#' @return `x`, possibly transformed.
#' @export
ensure_projected <- function(x, target = NULL, quiet = FALSE) {
  if (!has_crs(x)) {
    stop("Cannot project a layer without a CRS.", call. = FALSE)
  }
  if (is_projected(x) && is.null(target)) {
    return(x)
  }
  target <- target %||% utm_epsg_for(x)
  if (!quiet) {
    message("Transforming from ", crs_label(x), " to EPSG:", target, ".")
  }
  sf::st_transform(x, target)
}
