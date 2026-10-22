#' Reclassify a raster into value bins
#'
#' Cuts a continuous raster into classes using break points and assigns
#' labelled factor levels.
#'
#' @param r A single-layer `SpatRaster`.
#' @param breaks Numeric vector of class boundaries, in increasing order.
#' @param labels Optional class labels, one fewer than `breaks`.
#' @return A categorical `SpatRaster`.
#' @importFrom terra classify levels<-
#' @export
binRaster = function(r, breaks, labels = NULL) {
  if (is.unsorted(breaks)) {
    stop("`breaks` must be in increasing order.", call. = FALSE)
  }

  nClass = length(breaks) - 1
  labels = labels %||% sprintf("%g-%g", breaks[-length(breaks)], breaks[-1])
  if (length(labels) != nClass) {
    stop("Expected ", nClass, " labels, got ", length(labels), ".", call. = FALSE)
  }

  # Three-column matrix: from, to, new class id
  rcl = cbind(breaks[-length(breaks)], breaks[-1], seq_len(nClass))
  out = classify(r, rcl, include.lowest = TRUE)
  levels(out) = data.frame(id = seq_len(nClass), class = labels)
  out
}

#' Reclassify a raster using a lookup table
#'
#' @param r A single-layer `SpatRaster` of integer codes.
#' @param lookup A data frame with columns `from` and `to`.
#' @param others Value assigned to codes missing from `lookup`.
#' @return A `SpatRaster` of recoded values.
#' @export
recodeRaster = function(r, lookup, others = NA) {
  if (!all(c("from", "to") %in% names(lookup))) {
    stop("`lookup` needs `from` and `to` columns.", call. = FALSE)
  }
  if (anyDuplicated(lookup$from)) {
    stop("`lookup$from` contains duplicated codes.", call. = FALSE)
  }

  rcl = as.matrix(lookup[, c("from", "to")])
  classify(r, rcl, others = others)
}

#' Reclassify a raster into quantile classes
#'
#' @param r A single-layer `SpatRaster`.
#' @param n Number of classes.
#' @param sampleSize Number of cells sampled to estimate quantiles.
#' @return A categorical `SpatRaster`.
#' @importFrom terra spatSample
#' @export
quantileRaster = function(r, n = 5, sampleSize = 1e5) {
  # A regular sample is enough for break points and avoids reading every cell
  vals = spatSample(r, size = sampleSize, method = "regular", na.rm = TRUE)[, 1]
  breaks = unique(stats::quantile(vals, probs = seq(0, 1, length.out = n + 1)))
  if (length(breaks) < 3) {
    stop("Too few distinct values for ", n, " quantile classes.", call. = FALSE)
  }

  labels = paste0("Q", seq_len(length(breaks) - 1))
  binRaster(r, breaks, labels = labels)
}

#' Derive terrain indices from an elevation model
#'
#' @param dem A single-layer `SpatRaster` of elevations in metres.
#' @param indices Which terrain indices to compute.
#' @return A multi-layer `SpatRaster` with one layer per index.
#' @export
terrainStack = function(dem, indices = c("slope", "aspect", "TRI", "roughness")) {
  indices = match.arg(indices, several.ok = TRUE)
  layers = lapply(indices, function(idx) {
    # `unit` only matters for slope and aspect; terra ignores it otherwise
    unit = if (idx %in% c("slope", "aspect")) "degrees" else "radians"
    terra::terrain(dem, v = idx, unit = unit)
  })

  out = do.call(c, layers)
  names(out) = indices
  out
}

#' Summarise a raster layer
#'
#' Computes cell counts and basic statistics and returns them as a
#' `rasterSummary` object.
#'
#' @param r A single-layer `SpatRaster`.
#' @return An object of class `rasterSummary`.
#' @export
summariseRaster = function(r) {
  stats = terra::global(r, c("min", "max", "mean", "sd"), na.rm = TRUE)
  nValid = terra::global(!is.na(r), "sum")[1, 1]

  structure(
    list(
      name = names(r)[1],
      cells = terra::ncell(r),
      valid = nValid,
      resolution = terra::res(r),
      crs = terra::crs(r, describe = TRUE)$code,
      stats = unlist(stats[1, ])
    ),
    class = "rasterSummary"
  )
}

#' Format a raster summary
#'
#' @param x An object of class `rasterSummary`.
#' @param digits Number of significant digits for statistics.
#' @param ... Ignored.
#' @return A character vector of lines.
#' @rdname rasterSummary-methods
#' @export
format.rasterSummary = function(x, digits = 4, ...) {
  pctValid = 100 * x$valid / x$cells

  c(
    sprintf("<rasterSummary: %s>", x$name),
    sprintf("  cells: %d (%.1f%% valid)", x$cells, pctValid),
    sprintf("  resolution: %s x %s", x$resolution[1], x$resolution[2]),
    sprintf("  crs: EPSG:%s", x$crs),
    sprintf("  %s: %s", names(x$stats), signif(x$stats, digits))
  )
}

#' Print a raster summary
#'
#' @param x An object of class `rasterSummary`.
#' @param ... Passed to [format.rasterSummary()].
#' @rdname rasterSummary-methods
#' @export
print.rasterSummary = function(x, ...) {
  cat(format(x, ...), sep = "\n")
  invisible(x)
}

#' Convert a raster summary to a one-row data frame
#'
#' @param x An object of class `rasterSummary`.
#' @param ... Ignored.
#' @rdname rasterSummary-methods
#' @export
as.data.frame.rasterSummary = function(x, ...) {
  data.frame(
    name = x$name,
    cells = x$cells,
    valid = x$valid,
    resX = x$resolution[1],
    resY = x$resolution[2],
    t(x$stats)
  )
}
