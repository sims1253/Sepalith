#' Load a raster from disk
#'
#' Opens a raster file, checks that it has a CRS and optionally renames its
#' layers.
#'
#' @param path Path to a raster file readable by GDAL.
#' @param layerNames Optional character vector of new layer names.
#' @return A `SpatRaster`.
#' @importFrom terra rast crs nlyr
#' @export
loadRaster <- function(path, layerNames = NULL) {
  if (!file.exists(path)) {
    stop("Raster file not found: ", path, call. = FALSE)
  }

  r <- rast(path)
  # Without a CRS every later reprojection silently goes wrong
  if (crs(r) == "") {
    stop("Raster '", basename(path), "' has no CRS.", call. = FALSE)
  }

  if (!is.null(layerNames)) {
    if (length(layerNames) != nlyr(r)) {
      stop("Expected ", nlyr(r), " layer names, got ", length(layerNames), ".",
           call. = FALSE)
    }
    names(r) <- layerNames
  }
  r
}

#' Load a stack of single-band rasters
#'
#' Reads several files with identical geometry into one multi-layer raster,
#' using the file names (without extension) as layer names.
#'
#' @param paths Character vector of raster file paths.
#' @return A `SpatRaster` with one layer per file.
#' @export
loadRasterStack <- function(paths) {
  layers <- lapply(paths, loadRaster)

  # Fail early with terra's own message if any grid differs from the first
  for (i in seq_along(layers)[-1]) {
    terra::compareGeom(layers[[1]], layers[[i]], stopOnError = TRUE)
  }

  r <- rast(layers)
  names(r) <- tools::file_path_sans_ext(basename(paths))
  r
}

#' Crop a raster to a vector extent
#'
#' Reprojects the vector to the raster CRS when needed, then crops.
#'
#' @param r A `SpatRaster`.
#' @param aoi An `sf` object or `SpatVector` giving the area of interest.
#' @param snap How to align the crop extent with raster cells.
#' @return A cropped `SpatRaster`.
#' @export
cropToAoi <- function(r, aoi, snap = c("out", "near", "in")) {
  snap <- match.arg(snap)
  v <- asRasterVector(aoi, r)
  terra::crop(r, v, snap = snap)
}

#' Mask a raster to a vector area
#'
#' Crops to the area of interest first, then sets cells outside the polygons
#' to `NA`.
#'
#' @param r A `SpatRaster`.
#' @param aoi An `sf` object or `SpatVector` giving the area of interest.
#' @param inverse If `TRUE`, mask cells inside the polygons instead.
#' @return A masked `SpatRaster`.
#' @export
maskToAoi <- function(r, aoi, inverse = FALSE) {
  v <- asRasterVector(aoi, r)
  # Cropping first keeps the mask step from touching the whole raster
  r %>%
    terra::crop(v, snap = "out") %>%
    terra::mask(v, inverse = inverse)
}

#' Clip a raster and drop empty layers
#'
#' Masks to the area of interest and removes layers that contain only `NA`
#' cells afterwards.
#'
#' @param r A `SpatRaster`.
#' @param aoi An `sf` object or `SpatVector` giving the area of interest.
#' @return A masked `SpatRaster`, possibly with fewer layers.
#' @export
clipAndPrune <- function(r, aoi) {
  clipped <- maskToAoi(r, aoi)
  nonNa <- terra::global(!is.na(clipped), "sum")[, 1]

  if (all(nonNa == 0)) {
    stop("Area of interest does not overlap any raster cells.", call. = FALSE)
  }
  clipped[[which(nonNa > 0)]]
}

#' Convert an area of interest to a SpatVector in the raster CRS
#'
#' @param aoi An `sf` object or `SpatVector` giving the area of interest.
#' @param r A `SpatRaster`.
#' @noRd
asRasterVector <- function(aoi, r) {
  v <- if (inherits(aoi, "SpatVector")) aoi else terra::vect(aoi)
  if (!terra::same.crs(v, r)) {
    v <- terra::project(v, crs(r))
  }
  v
}

#' Reproject a raster with a method suited to its data type
#'
#' Categorical rasters use nearest-neighbour resampling, continuous ones use
#' bilinear interpolation.
#'
#' @param r A `SpatRaster`.
#' @param crs Target CRS as an EPSG string, WKT or another raster.
#' @param categorical If `TRUE`, use nearest-neighbour resampling.
#' @return A reprojected `SpatRaster`.
#' @export
reprojectRaster <- function(r, crs, categorical = any(terra::is.factor(r))) {
  method <- if (categorical) "near" else "bilinear"
  terra::project(r, crs, method = method)
}
