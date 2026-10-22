#' @title Build a square grid over a layer
#'
#' @description
#' Creates square cells of a fixed size covering the bounding box of `x`.
#'
#' @param x An `sf` or `sfc` object defining the extent.
#' @param cell_size Cell edge length in map units.
#' @param clip If `TRUE`, keep only cells that intersect `x`.
#' @return An `sf` object of grid cells with a `cell_id` column.
#' @rdname make_grid
#' @export
square_grid = function(x, cell_size, clip = TRUE) {
  assert_projected(x)
  cells = sf::st_make_grid(x, cellsize = cell_size, square = TRUE)
  finish_grid(cells, x, clip)
}

#' @title Build a hexagonal grid over a layer
#'
#' @description
#' Creates flat-topped hexagons of a fixed size covering the bounding box of
#' `x`.
#'
#' @param x An `sf` or `sfc` object defining the extent.
#' @param cell_size Distance between opposite hexagon edges in map units.
#' @param clip If `TRUE`, keep only cells that intersect `x`.
#' @rdname make_grid
#' @export
hex_grid = function(x, cell_size, clip = TRUE) {
  assert_projected(x)
  cells = sf::st_make_grid(x, cellsize = cell_size, square = FALSE, flat_topped = TRUE)
  finish_grid(cells, x, clip)
}

#' @title Build a grid with a target number of cells
#'
#' @description
#' Chooses a square cell size so that roughly `n` cells cover the bounding box
#' of `x`.
#'
#' @param x An `sf` or `sfc` object defining the extent.
#' @param n Approximate number of cells.
#' @param clip If `TRUE`, keep only cells that intersect `x`.
#' @rdname make_grid
#' @export
grid_with_n_cells = function(x, n, clip = TRUE) {
  assert_projected(x)
  if (!is.numeric(n) || length(n) != 1 || is.na(n) || n < 1) {
    stop("`n` must be a single number of at least 1.", call. = FALSE)
  }

  bbox = sf::st_bbox(x)
  area = (bbox[["xmax"]] - bbox[["xmin"]]) * (bbox[["ymax"]] - bbox[["ymin"]])
  # Edge length of a square whose area is an n-th of the bounding box
  cell_size = sqrt(area / n)
  square_grid(x, cell_size, clip = clip)
}

#' @title Finish a freshly built grid
#'
#' @param cells An `sfc` of grid polygons.
#' @param x An `sf` or `sfc` object defining the extent.
#' @param clip If `TRUE`, keep only cells that intersect `x`.
#' @noRd
finish_grid = function(cells, x, clip) {
  if (clip) {
    keep = lengths(sf::st_intersects(cells, sf::st_union(sf::st_geometry(x)))) > 0
    cells = cells[keep]
  }
  sf::st_sf(cell_id = seq_along(cells), geometry = cells)
}

#' @title Split a bounding box into tiles
#'
#' @description
#' Divides the bounding box of `x` into `nx` by `ny` tiles, optionally with an
#' overlap so that edge effects can be trimmed after processing.
#'
#' @param x An `sf` or `sfc` object defining the extent.
#' @param nx,ny Number of tiles along the x and y axes.
#' @param overlap Overlap between neighbouring tiles in map units.
#' @return A data frame with one row per tile and its bounding coordinates.
#' @export
tile_bbox = function(x, nx = 2, ny = 2, overlap = 0) {
  bbox = sf::st_bbox(x)
  xs = seq(bbox[["xmin"]], bbox[["xmax"]], length.out = nx + 1)
  ys = seq(bbox[["ymin"]], bbox[["ymax"]], length.out = ny + 1)

  tiles = expand.grid(col = seq_len(nx), row = seq_len(ny))
  tiles$xmin = xs[tiles$col] - overlap
  tiles$xmax = xs[tiles$col + 1] + overlap
  tiles$ymin = ys[tiles$row] - overlap
  tiles$ymax = ys[tiles$row + 1] + overlap
  tiles$tile_id = sprintf("r%02d_c%02d", tiles$row, tiles$col)
  tiles
}

#' @title Convert a tile table to polygons
#'
#' @param tiles A data frame as returned by [tile_bbox()].
#' @param crs Coordinate reference system of the tile coordinates.
#' @return An `sf` object with one polygon per tile.
#' @export
tiles_as_sf = function(tiles, crs) {
  polys = vector("list", nrow(tiles))
  for (i in seq_len(nrow(tiles))) {
    bb = sf::st_bbox(c(xmin = tiles$xmin[i], ymin = tiles$ymin[i],
                       xmax = tiles$xmax[i], ymax = tiles$ymax[i]), crs = crs)
    polys[[i]] = sf::st_as_sfc(bb)
  }

  sf::st_sf(tile_id = tiles$tile_id, geometry = do.call(c, polys))
}

#' @title Assign features to tiles
#'
#' @description
#' Each feature is assigned to the tile containing its representative point,
#' so that every feature belongs to exactly one tile.
#'
#' @param x An `sf` object to partition.
#' @param tiles An `sf` object as returned by [tiles_as_sf()].
#' @return `x` with a `tile_id` column added.
#' @export
assign_to_tiles = function(x, tiles) {
  assert_same_crs(x, tiles)
  anchor = sf::st_point_on_surface(sf::st_geometry(x))

  # With overlapping tiles a point can hit several; the nearest one decides,
  # and points outside every tile get NA
  idx = sf::st_nearest_feature(anchor, tiles)
  inside = lengths(sf::st_intersects(anchor, tiles)) > 0
  x$tile_id = ifelse(inside, tiles$tile_id[idx], NA_character_)
  x
}

#' @title Aggregate a point attribute onto grid cells
#'
#' @param grid An `sf` object of grid cells with a `cell_id` column.
#' @param points An `sf` object of point features.
#' @param value Name of the numeric column to aggregate.
#' @param fun Summary function applied per cell.
#' @return `grid` with a column named after `value` holding the summaries.
#' @export
aggregate_to_grid = function(grid, points, value, fun = sum) {
  assert_same_crs(grid, points)
  hits = sf::st_intersects(grid, points)
  vals = points[[value]]

  # Empty cells get NA rather than fun(numeric(0)), which may be 0 or -Inf
  grid[[value]] = vapply(hits, function(h) {
    if (length(h) == 0) NA_real_ else as.numeric(fun(vals[h]))
  }, numeric(1))
  grid
}
