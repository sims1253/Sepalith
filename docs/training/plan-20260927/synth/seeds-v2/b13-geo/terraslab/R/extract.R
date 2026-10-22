#' Extract raster values at point locations
#'
#' @param r A `SpatRaster`.
#' @param points An `sf` object or `SpatVector` of point features.
#' @param method Either `"simple"` (cell value) or `"bilinear"` interpolation.
#' @return A data frame with one row per point and one column per layer.
#' @importFrom terra extract
#' @export
extractAtPoints <- function(r, points, method = c("simple", "bilinear")) {
  method <- match.arg(method)
  v <- asRasterVector(points, r)
  vals <- extract(r, v, method = method, ID = FALSE)
  cbind(pointId = seq_len(nrow(vals)), vals)
}

#' Extract summarised raster values within polygons
#'
#' @param r A `SpatRaster`.
#' @param polygons An `sf` object or `SpatVector` of polygon features.
#' @param fun Summary function applied to the cells of each polygon.
#' @return A data frame with one row per polygon and one column per layer.
#' @export
extractInPolygons <- function(r, polygons, fun = mean) {
  v <- asRasterVector(polygons, r)
  vals <- extract(r, v, fun = fun, na.rm = TRUE, ID = FALSE)
  cbind(polygonId = seq_len(nrow(vals)), vals)
}

#' Extract area-weighted raster means within polygons
#'
#' Uses the fraction of each cell covered by the polygon as a weight, which is
#' more accurate than [extractInPolygons()] for small polygons. Polygons that
#' cover no valid cells get `NA`.
#'
#' @param r A `SpatRaster`.
#' @param polygons An `sf` object or `SpatVector` of polygon features.
#' @return A data frame with one row per polygon and one column per layer.
#' @export
extractWeighted <- function(r, polygons) {
  v <- asRasterVector(polygons, r)
  raw <- extract(r, v, exact = TRUE)
  layers <- setdiff(names(raw), c("ID", "fraction"))

  # One output row per input polygon, even those that touch no cells and
  # therefore never show up in `raw$ID`
  ids <- seq_len(nrow(v))
  out <- matrix(NA_real_, nrow = length(ids), ncol = length(layers),
                dimnames = list(NULL, layers))

  for (i in seq_along(ids)) {
    rows <- raw[raw$ID == ids[i], , drop = FALSE]
    for (lyr in layers) {
      ok <- !is.na(rows[[lyr]])
      if (!any(ok)) {
        next
      }
      out[i, lyr] <- sum(rows[[lyr]][ok] * rows$fraction[ok]) / sum(rows$fraction[ok])
    }
  }

  data.frame(polygonId = ids, out)
}

#' Summarise a raster by zones of another raster
#'
#' @param r A `SpatRaster` of values.
#' @param zones A single-layer `SpatRaster` of zone codes aligned with `r`.
#' @param fun Name of the summary statistic, such as `"mean"` or `"sum"`.
#' @return A data frame with one row per zone.
#' @family zonal helpers
#' @export
zonalByRaster <- function(r, zones, fun = "mean") {
  terra::compareGeom(r, zones, stopOnError = TRUE)
  out <- terra::zonal(r, zones, fun = fun, na.rm = TRUE)
  names(out)[1] <- "zone"
  out
}

#' Summarise a raster by polygon zones
#'
#' Rasterises the polygons onto the grid of `r` and computes zonal statistics
#' per polygon attribute value.
#'
#' @param r A `SpatRaster` of values.
#' @param polygons An `sf` object or `SpatVector` of polygon features.
#' @param field Name of the polygon attribute that defines zones.
#' @param fun Name of the summary statistic, such as `"mean"` or `"sum"`.
#' @return A data frame with one row per zone.
#' @family zonal helpers
#' @export
zonalByPolygons <- function(r, polygons, field, fun = "mean") {
  v <- asRasterVector(polygons, r)
  zones <- terra::rasterize(v, r[[1]], field = field)

  out <- zonalByRaster(r, zones, fun = fun)
  names(out)[1] <- field
  out
}

#' Compute class areas per zone
#'
#' Cross-tabulates a categorical raster against zones and converts cell counts
#' to hectares.
#'
#' @param classes A single-layer categorical `SpatRaster`.
#' @param zones A single-layer `SpatRaster` of zone codes aligned with `classes`.
#' @return A long data frame with `zone`, `class` and `areaHa` columns.
#' @family zonal helpers
#' @export
zonalClassArea <- function(classes, zones) {
  terra::compareGeom(classes, zones, stopOnError = TRUE)

  # A mean cell size is fine for projected grids; on lon/lat grids it is
  # only an approximation away from the centre latitude
  cellHa <- terra::global(terra::cellSize(classes, unit = "ha"), "mean")[1, 1]
  tab <- terra::crosstab(c(zones, classes), long = TRUE)
  names(tab) <- c("zone", "class", "cells")
  tab$areaHa <- tab$cells * cellHa

  tab[order(tab$zone, tab$class), c("zone", "class", "areaHa")]
}

#' Compute the share of valid cells per zone
#'
#' @param r A single-layer `SpatRaster` of values.
#' @param zones A single-layer `SpatRaster` of zone codes aligned with `r`.
#' @return A data frame with `zone` and `validShare` columns.
#' @export
zonalCoverage <- function(r, zones) {
  valid <- !is.na(r)
  out <- terra::zonal(valid, zones, fun = "mean", na.rm = TRUE)
  names(out) <- c("zone", "validShare")
  out
}
