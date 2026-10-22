#' Compute a tiling scheme for a raster
#'
#' Splits the raster into blocks of at most `tileRows` by `tileCols` cells and
#' returns their row and column ranges.
#'
#' @param r A `SpatRaster`.
#' @param tileRows,tileCols Maximum tile height and width in cells.
#' @return A data frame with one row per tile.
#' @family tiling helpers
#' @importFrom terra nrow ncol
#' @export
tileScheme <- function(r, tileRows = 512, tileCols = 512) {
  if (tileRows < 1 || tileCols < 1) {
    stop("Tile dimensions must be positive.", call. = FALSE)
  }
  rowStarts <- seq(1, nrow(r), by = tileRows)
  colStarts <- seq(1, ncol(r), by = tileCols)
  scheme <- expand.grid(colStart = colStarts, rowStart = rowStarts)
  scheme$rowEnd <- pmin(scheme$rowStart + tileRows - 1, nrow(r))
  scheme$colEnd <- pmin(scheme$colStart + tileCols - 1, ncol(r))
  scheme$tileId <- sprintf("t%04d", seq_len(nrow(scheme)))
  return(scheme[, c("tileId", "rowStart", "rowEnd", "colStart", "colEnd")])
}

#' Get the extent of one tile
#'
#' @param r A `SpatRaster`.
#' @param tile One row of a scheme as returned by [tileScheme()].
#' @param buffer Number of extra cells to add on each side.
#' @return A `SpatExtent`.
#' @family tiling helpers
#' @importFrom terra ext xFromCol yFromRow res nrow ncol
#' @export
tileExtent <- function(r, tile, buffer = 0) {
  rows <- c(max(tile$rowStart - buffer, 1), min(tile$rowEnd + buffer, nrow(r)))
  cols <- c(max(tile$colStart - buffer, 1), min(tile$colEnd + buffer, ncol(r)))
  half <- res(r) / 2
  xs <- xFromCol(r, cols)
  ys <- yFromRow(r, rows)
  return(ext(xs[1] - half[1], xs[2] + half[1], ys[2] - half[2], ys[1] + half[2]))
}

#' Write a raster as a set of tiles
#'
#' Crops each tile from the raster and writes it as a GeoTIFF in `outDir`.
#'
#' @param r A `SpatRaster`.
#' @param outDir Directory to write tiles into. Created if missing.
#' @param tileRows,tileCols Maximum tile height and width in cells.
#' @param overwrite If `TRUE`, replace existing tile files.
#' @return A character vector of written file paths, invisibly.
#' @family tiling helpers
#' @importFrom terra crop writeRaster
#' @export
writeTiles <- function(r, outDir, tileRows = 512, tileCols = 512, overwrite = FALSE) {
  dir.create(outDir, showWarnings = FALSE, recursive = TRUE)
  scheme <- tileScheme(r, tileRows, tileCols)
  paths <- file.path(outDir, paste0(scheme$tileId, ".tif"))
  for (i in seq_len(nrow(scheme))) {
    if (file.exists(paths[i]) && !overwrite) {
      next
    }
    piece <- crop(r, tileExtent(r, scheme[i, ]))
    writeRaster(piece, paths[i], overwrite = overwrite, gdal = "COMPRESS=DEFLATE")
  }
  return(invisible(paths))
}

#' Apply a function to a raster tile by tile
#'
#' Processes each tile with a buffer so that focal operations have enough
#' context, trims the buffer and merges the results.
#'
#' @param r A `SpatRaster`.
#' @param fun A function taking and returning a `SpatRaster`.
#' @param buffer Number of extra cells to add on each side.
#' @param ... Passed to [tileScheme()].
#' @return A `SpatRaster` with the same geometry as `r`.
#' @importFrom terra crop merge sprc
#' @export
applyByTile <- function(r, fun, buffer = 0, ...) {
  scheme <- tileScheme(r, ...)
  pieces <- vector("list", nrow(scheme))
  for (i in seq_len(nrow(scheme))) {
    tile <- scheme[i, ]
    padded <- crop(r, tileExtent(r, tile, buffer = buffer))
    result <- fun(padded)
    pieces[[i]] <- crop(result, tileExtent(r, tile))
  }
  if (length(pieces) == 1) {
    return(pieces[[1]])
  }
  return(merge(sprc(pieces)))
}

#' Mosaic tiles from a directory
#'
#' @param tileDir Directory containing tile GeoTIFFs.
#' @param pattern Regular expression matching tile file names.
#' @param fun Function used where tiles overlap.
#' @return A `SpatRaster`.
#' @importFrom terra rast sprc mosaic
#' @export
mosaicTiles <- function(tileDir, pattern = "\\.tif$", fun = "mean") {
  files <- list.files(tileDir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) {
    stop("No tiles found in '", tileDir, "'.", call. = FALSE)
  }
  collection <- sprc(lapply(files, rast))
  return(mosaic(collection, fun = fun))
}

#' Count valid cells per tile
#'
#' Useful for skipping tiles that lie entirely outside the study area.
#'
#' @param r A `SpatRaster`.
#' @param ... Passed to [tileScheme()].
#' @return The tiling scheme with an added `validCells` column.
#' @importFrom terra crop global
#' @export
tileValidCounts <- function(r, ...) {
  scheme <- tileScheme(r, ...)
  counts <- integer(nrow(scheme))
  for (i in seq_len(nrow(scheme))) {
    piece <- crop(r[[1]], tileExtent(r, scheme[i, ]))
    counts[i] <- as.integer(global(!is.na(piece), "sum")[1, 1])
  }
  scheme$validCells <- counts
  return(scheme)
}
