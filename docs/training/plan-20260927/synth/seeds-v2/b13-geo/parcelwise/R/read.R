#' @title Read a GeoPackage layer
#'
#' @description
#' Reads one layer from a GeoPackage file and normalises its geometry column
#' and coordinate reference system.
#'
#' @param path Path to the source file on disk.
#' @param layer Name of the layer to read. If `NULL`, the first layer is used.
#' @param crs Target coordinate reference system, or `NULL` to keep the source CRS.
#' @return An `sf` object with a geometry column named `geometry`.
#' @rdname read_layer
#' @export
read_layer_gpkg <- function(path, layer = NULL, crs = NULL) {
  check_layer_path(path, "gpkg")
  if (is.null(layer)) {
    layer <- list_gpkg_layers(path)[1]
  }

  x <- sf::st_read(path, layer = layer, quiet = TRUE, stringsAsFactors = FALSE)
  finalize_layer(x, crs)
}

#' @title Read an ESRI shapefile
#'
#' @description
#' Reads a shapefile and normalises its geometry column and coordinate
#' reference system. A missing `.prj` sidecar is treated as an error.
#'
#' @param path Path to the source file on disk.
#' @param crs Target coordinate reference system, or `NULL` to keep the source CRS.
#' @rdname read_layer
#' @export
read_layer_shp <- function(path, crs = NULL) {
  check_layer_path(path, "shp")

  # GDAL would happily read the file without a CRS; we would rather not guess
  prj <- sub("\\.shp$", ".prj", path, ignore.case = TRUE)
  if (!file.exists(prj)) {
    stop("Shapefile '", basename(path), "' has no .prj file.", call. = FALSE)
  }

  x <- sf::st_read(path, quiet = TRUE, stringsAsFactors = FALSE)
  finalize_layer(x, crs)
}

#' @title Read a GeoJSON file
#'
#' @description
#' Reads a GeoJSON file and normalises its geometry column and coordinate
#' reference system. GeoJSON is always assumed to be in WGS 84.
#'
#' @param path Path to the source file on disk.
#' @param crs Target coordinate reference system, or `NULL` to keep the source CRS.
#' @rdname read_layer
#' @export
read_layer_geojson <- function(path, crs = NULL) {
  check_layer_path(path, c("geojson", "json"))
  x <- sf::st_read(path, quiet = TRUE, stringsAsFactors = FALSE)

  # RFC 7946 fixes the CRS, so older files often omit it
  if (is.na(sf::st_crs(x))) {
    x <- sf::st_set_crs(x, 4326)
  }
  finalize_layer(x, crs)
}

#' @title Read point locations from a CSV file
#'
#' @description
#' Reads a delimited file with coordinate columns and converts it to point
#' geometries. Rows with missing coordinates are dropped with a warning.
#'
#' @param path Path to the source file on disk.
#' @param x,y Names of the longitude and latitude columns.
#' @param crs Coordinate reference system of the coordinate columns.
#' @rdname read_layer
#' @export
read_layer_csv <- function(path, x = "lon", y = "lat", crs = 4326) {
  check_layer_path(path, c("csv", "txt"))
  tbl <- utils::read.csv(path, stringsAsFactors = FALSE)

  missing_cols <- setdiff(c(x, y), names(tbl))
  if (length(missing_cols) > 0) {
    stop("Missing coordinate columns: ", toString(missing_cols), call. = FALSE)
  }

  # st_as_sf() refuses NA coordinates, so drop those rows up front
  ok <- stats::complete.cases(tbl[, c(x, y)])
  if (!all(ok)) {
    warning(sum(!ok), " row(s) with missing coordinates dropped.", call. = FALSE)
    tbl <- tbl[ok, , drop = FALSE]
  }

  pts <- sf::st_as_sf(tbl, coords = c(x, y), crs = crs, remove = FALSE)
  finalize_layer(pts, NULL)
}

#' @title Read a spatial layer of any supported format
#'
#' @description
#' Dispatches to the matching `read_layer_*()` function based on the file
#' extension, or on an explicitly requested format.
#'
#' @param path Path to the source file on disk.
#' @param ... Passed to the format-specific reader.
#' @param format One of `"auto"`, `"gpkg"`, `"shp"`, `"geojson"` or `"csv"`.
#' @return An `sf` object with a geometry column named `geometry`.
#' @export
read_layer <- function(path, ...,
                       format = c("auto", "gpkg", "shp", "geojson", "csv")) {
  format <- match.arg(format)
  if (format == "auto") {
    format <- detect_layer_format(path)
  }

  switch(format,
    gpkg = read_layer_gpkg(path, ...),
    shp = read_layer_shp(path, ...),
    geojson = read_layer_geojson(path, ...),
    csv = read_layer_csv(path, ...)
  )
}

#' @title Detect the format of a spatial file
#'
#' @param path Path to the source file on disk.
#' @return A single string naming the format.
#' @keywords internal
detect_layer_format <- function(path) {
  ext <- tolower(tools::file_ext(path))
  formats <- c(gpkg = "gpkg", shp = "shp", geojson = "geojson", json = "geojson",
               csv = "csv", txt = "csv")

  if (!ext %in% names(formats)) {
    stop("Cannot infer layer format from extension '.", ext, "'.", call. = FALSE)
  }
  unname(formats[[ext]])
}

#' @title Check that a layer path exists and has an expected extension
#'
#' @param path Path to the source file on disk.
#' @param ext Character vector of accepted extensions, without the dot.
#' @keywords internal
check_layer_path <- function(path, ext) {
  if (!is.character(path) || length(path) != 1 || is.na(path)) {
    stop("`path` must be a single string.", call. = FALSE)
  }
  if (!file.exists(path)) {
    stop("File not found: ", path, call. = FALSE)
  }
  if (!tolower(tools::file_ext(path)) %in% ext) {
    stop("Expected a .", paste(ext, collapse = " or ."), " file.", call. = FALSE)
  }
  invisible(path)
}

#' @title Normalise a freshly read layer
#'
#' @description
#' Renames the active geometry column to `geometry`, drops Z/M dimensions and
#' optionally transforms to a target CRS.
#'
#' @param x An `sf` object.
#' @param crs Target coordinate reference system, or `NULL` to keep the source CRS.
#' @return An `sf` object with a geometry column named `geometry`.
#' @keywords internal
finalize_layer <- function(x, crs) {
  # Drivers disagree on the name (geom, wkb_geometry, ...); settle on one
  geom_col <- attr(x, "sf_column")
  if (geom_col != "geometry") {
    names(x)[names(x) == geom_col] <- "geometry"
    sf::st_geometry(x) <- "geometry"
  }

  x <- sf::st_zm(x, drop = TRUE, what = "ZM")
  if (!is.null(crs)) {
    x <- sf::st_transform(x, crs)
  }
  x
}

#' @title List the layers stored in a GeoPackage
#'
#' @param path Path to the source file on disk.
#' @return A character vector of layer names.
#' @noRd
list_gpkg_layers <- function(path) {
  info <- sf::st_layers(path)
  if (length(info$name) == 0) {
    stop("GeoPackage '", basename(path), "' contains no layers.", call. = FALSE)
  }
  info$name
}
