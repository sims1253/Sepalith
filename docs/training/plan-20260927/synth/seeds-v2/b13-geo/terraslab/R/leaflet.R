#' Create a base leaflet map
#'
#' Starts a map with a light basemap, a satellite alternative and a scale bar.
#'
#' @param height Map height as a CSS string.
#' @return A `leaflet` htmlwidget.
#' @family map builders
#' @export
baseMap <- function(height = "600px") {
  leaflet::leaflet(height = height) %>%
    leaflet::addProviderTiles("CartoDB.Positron", group = "Light") %>%
    leaflet::addProviderTiles("Esri.WorldImagery", group = "Satellite") %>%
    leaflet::addScaleBar(position = "bottomleft")
}

#' Add a raster layer to a leaflet map
#'
#' Reprojects the raster to Web Mercator if needed and adds it with a
#' continuous colour palette and legend.
#'
#' @param map A `leaflet` htmlwidget.
#' @param r A single-layer `SpatRaster`.
#' @param group Name of the layer group shown in the layer control.
#' @param palette Name of a colour palette understood by [leaflet::colorNumeric()].
#' @param opacity Layer opacity between 0 and 1.
#' @return The modified `leaflet` htmlwidget.
#' @export
addRasterLayer <- function(map, r, group, palette = "viridis", opacity = 0.7) {
  vals <- terra::values(r, mat = FALSE)
  # NA cells stay transparent so the basemap shows through gaps
  pal <- leaflet::colorNumeric(palette, domain = range(vals, na.rm = TRUE),
                               na.color = "transparent")

  map %>%
    leaflet::addRasterImage(r, colors = pal, opacity = opacity, group = group,
                            project = TRUE) %>%
    leaflet::addLegend(pal = pal, values = vals, title = group, group = group,
                       position = "bottomright")
}

#' Add polygons to a leaflet map
#'
#' @param map A `leaflet` htmlwidget.
#' @param polygons An `sf` object of polygon features.
#' @param group Name of the layer group shown in the layer control.
#' @param labelCol Name of the column used for hover labels.
#' @param color Outline colour.
#' @return The modified `leaflet` htmlwidget.
#' @rdname addFeatureLayer
#' @export
addPolygonLayer <- function(map, polygons, group, labelCol = NULL, color = "#3a5a40") {
  polygons <- sf::st_transform(polygons, 4326)
  labels <- if (is.null(labelCol)) NULL else as.character(polygons[[labelCol]])

  map %>%
    leaflet::addPolygons(data = polygons, group = group, label = labels,
                         color = color, weight = 1.5, fillOpacity = 0.15,
                         highlightOptions = leaflet::highlightOptions(weight = 3))
}

#' Add points to a leaflet map
#'
#' @param map A `leaflet` htmlwidget.
#' @param points An `sf` object of point features.
#' @param group Name of the layer group shown in the layer control.
#' @param labelCol Name of the column used for hover labels.
#' @param color Marker colour.
#' @rdname addFeatureLayer
#' @export
addPointLayer <- function(map, points, group, labelCol = NULL, color = "#bc4749") {
  points <- sf::st_transform(points, 4326)
  labels <- if (is.null(labelCol)) NULL else as.character(points[[labelCol]])

  map %>%
    leaflet::addCircleMarkers(data = points, group = group, label = labels,
                              color = color, radius = 5, stroke = FALSE,
                              fillOpacity = 0.8)
}

#' Finish a map with a layer control
#'
#' Adds a layer control listing the basemaps and all overlay groups, and zooms
#' to the given extent.
#'
#' @param map A `leaflet` htmlwidget.
#' @param overlays Character vector of overlay group names.
#' @param bounds An `sf` object whose extent the map should fit, or `NULL`
#'   to keep leaflet's default view.
#' @return The modified `leaflet` htmlwidget.
#' @export
finishMap <- function(map, overlays, bounds = NULL) {
  map <- map %>%
    leaflet::addLayersControl(
      baseGroups = c("Light", "Satellite"),
      overlayGroups = overlays,
      options = leaflet::layersControlOptions(collapsed = FALSE)
    )

  if (!is.null(bounds)) {
    # fitBounds() wants lon/lat, whatever CRS the layer came in
    bb <- sf::st_bbox(sf::st_transform(bounds, 4326))
    map <- map %>%
      leaflet::fitBounds(bb[["xmin"]], bb[["ymin"]], bb[["xmax"]], bb[["ymax"]])
  }
  map
}

#' Build a complete map of a raster with context layers
#'
#' Convenience wrapper combining [baseMap()], the layer helpers and
#' [finishMap()]. The view is fitted to the polygons if given, otherwise to
#' the points, otherwise leaflet zooms to the raster on its own.
#'
#' @param r A single-layer `SpatRaster`.
#' @param polygons An `sf` object of polygon features, or `NULL`.
#' @param points An `sf` object of point features, or `NULL`.
#' @param title Title shown in the raster legend.
#' @return A `leaflet` htmlwidget.
#' @export
quickRasterMap <- function(r, polygons = NULL, points = NULL, title = names(r)[1]) {
  map <- baseMap() %>%
    addRasterLayer(r, group = title)
  overlays <- title

  if (!is.null(polygons)) {
    map <- addPolygonLayer(map, polygons, group = "Areas")
    overlays <- c(overlays, "Areas")
  }
  if (!is.null(points)) {
    map <- addPointLayer(map, points, group = "Sites")
    overlays <- c(overlays, "Sites")
  }

  # Polygons give the most useful extent; fall back to the points, and to
  # no explicit bounds when neither layer was supplied
  bounds <- if (!is.null(polygons)) polygons else points
  finishMap(map, overlays, bounds = bounds)
}

#' Save a leaflet map as a standalone HTML file
#'
#' @param map A `leaflet` htmlwidget.
#' @param path Output path ending in `.html`.
#' @param title Page title.
#' @return `path`, invisibly.
#' @export
saveMap <- function(map, path, title = "Map") {
  if (!grepl("\\.html?$", path)) {
    stop("`path` must end in .html.", call. = FALSE)
  }

  htmlwidgets::saveWidget(map, file = normalizePath(path, mustWork = FALSE),
                          selfcontained = TRUE, title = title)
  invisible(path)
}
