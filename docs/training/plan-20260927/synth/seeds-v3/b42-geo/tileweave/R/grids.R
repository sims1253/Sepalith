#' Define a regular lon/lat grid
#'
#' @param xmin,ymin,xmax,ymax Grid extent in degrees.
#' @param cellSize Cell edge length in degrees.
#' @return A list of class `tw_grid` with elements `xmin`, `ymin`,
#'   `cellSize`, `nx`, `ny`, `xEdges` and `yEdges`.
#' @export
makeGrid = function(xmin, ymin, xmax, ymax, cellSize) {
  if (!is.numeric(cellSize) || length(cellSize) != 1 || cellSize <= 0) {
    stop("`cellSize` must be a single positive number.", call. = FALSE)
  }

  # Columns along x
  if (xmin >= xmax) {
    stop(sprintf("`%s` must be greater than `%s`.", "xmax", "xmin"), call. = FALSE)
  }
  nx = ceiling((xmax - xmin) / cellSize - 1e-9)
  xEdges = xmin + cellSize * (0:nx)

  # Rows along y
  if (ymin >= ymax) {
    stop(sprintf("`%s` must be greater than `%s`.", "ymax", "ymin"), call. = FALSE)
  }
  ny = ceiling((ymax - ymin) / cellSize - 1e-9)
  yEdges = ymin + cellSize * (0:ny)
  grid = list(
    xmin = xmin,
    ymin = ymin,
    cellSize = cellSize,
    nx = as.integer(nx),
    ny = as.integer(ny),
    xEdges = xEdges,
    yEdges = yEdges
  )
  class(grid) = "tw_grid"
  return(grid)
}

#' Grid cell containing each point
#'
#' Cells are numbered row by row from the lower left, starting at 1.
#'
#' @param lon,lat Coordinates in degrees.
#' @param grid A grid from [makeGrid()].
#' @param outside What to do with points outside the grid: `"na"`,
#'   `"clamp"` or `"error"`.
#' @return A data frame with integer columns `col`, `row` and `cell`.
#' @export
cellIndex = function(lon, lat, grid, outside = c("na", "clamp", "error")) {
  outside = match.arg(outside)

  # Column from lon
  col = floor((lon - grid$xmin) / grid$cellSize) + 1
  offX = !is.na(col) & (col < 1 | col > grid$nx)
  if (outside == "clamp") {
    col = pmin(pmax(col, 1), grid$nx)
  } else if (outside == "error" && any(offX)) {
    stop(sprintf("%d point(s) lie outside the grid along %s.", sum(offX), "x"), call. = FALSE)
  } else {
    col[offX] = NA
  }

  # Row from lat
  row = floor((lat - grid$ymin) / grid$cellSize) + 1
  offY = !is.na(row) & (row < 1 | row > grid$ny)
  if (outside == "clamp") {
    row = pmin(pmax(row, 1), grid$ny)
  } else if (outside == "error" && any(offY)) {
    stop(sprintf("%d point(s) lie outside the grid along %s.", sum(offY), "y"), call. = FALSE)
  } else {
    row[offY] = NA
  }
  cell = (row - 1) * grid$nx + col
  return(data.frame(col = as.integer(col), row = as.integer(row), cell = as.integer(cell)))
}

#' Centre and bounds of grid cells
#'
#' @param cell Integer cell numbers as returned by [cellIndex()].
#' @param grid A grid from [makeGrid()].
#' @return A data frame with columns `cell`, `lon`, `lat`, `xmin`, `ymin`,
#'   `xmax` and `ymax`.
#' @export
cellGeometry = function(cell, grid) {
  if (any(cell < 1 | cell > grid$nx * grid$ny, na.rm = TRUE)) {
    stop("`cell` must index a cell of `grid`.", call. = FALSE)
  }
  half = grid$cellSize / 2

  # Along x
  col = (cell - 1) %% grid$nx + 1
  xmin = grid$xEdges[col]
  xmax = grid$xEdges[col + 1]
  lon = xmin + half

  # Along y
  row = (cell - 1) %/% grid$nx + 1
  ymin = grid$yEdges[row]
  ymax = grid$yEdges[row + 1]
  lat = ymin + half
  return(data.frame(cell = cell, lon = lon, lat = lat, xmin = xmin, ymin = ymin,
    xmax = xmax, ymax = ymax))
}

#' Aggregate point values onto a grid
#'
#' @param lon,lat Coordinates in degrees.
#' @param value Numeric values at the points; ignored for `"count"`.
#' @param grid A grid from [makeGrid()].
#' @param fun One of `"count"`, `"sum"`, `"mean"`, `"median"`, `"sd"`, `"min"`
#'   or `"max"`.
#' @return A numeric matrix with `grid$ny` rows (south to north) and
#'   `grid$nx` columns; empty cells are `0` for `"count"` and `"sum"`, `NA`
#'   otherwise.
#' @export
aggregateToGrid = function(lon, lat, value, grid,
                           fun = c("count", "sum", "mean", "median", "sd", "min", "max")) {
  fun = match.arg(fun)
  idx = cellIndex(lon, lat, grid)
  keep = !is.na(idx$cell) & (fun == "count" | !is.na(value))
  cell = factor(idx$cell[keep], levels = seq_len(grid$nx * grid$ny))
  v = if (fun == "count") rep(1, sum(keep)) else value[keep]
  agg = switch(fun,
    count = {
      out = tapply(v, cell, length)
      out[is.na(out)] = 0
      out
    },
    sum = {
      out = tapply(v, cell, sum)
      out[is.na(out)] = 0
      out
    },
    mean = {
      out = tapply(v, cell, mean)
      out
    },
    median = {
      out = tapply(v, cell, median)
      out
    },
    sd = {
      out = tapply(v, cell, sd)
      out
    },
    min = {
      out = tapply(v, cell, min)
      out
    },
    max = {
      out = tapply(v, cell, max)
      out
    }
  )
  return(matrix(as.numeric(agg), nrow = grid$ny, ncol = grid$nx, byrow = TRUE))
}

#' Coarsen or refine a gridded matrix
#'
#' @param values A numeric matrix laid out as returned by
#'   [aggregateToGrid()].
#' @param factor Integer number of fine cells per coarse cell edge.
#' @param direction `"coarsen"` (block means) or `"refine"` (repeat each
#'   value).
#' @return A numeric matrix.
#' @export
resampleGrid = function(values, factor, direction = c("coarsen", "refine")) {
  direction = match.arg(direction)
  stopifnot(is.matrix(values), factor >= 1, factor %% 1 == 0)
  ny = nrow(values)
  nx = ncol(values)
  if (direction == "coarsen") {
    if (ny %% factor != 0 || nx %% factor != 0) {
      stop(sprintf("Matrix dimensions must be multiples of %d.", factor), call. = FALSE)
    }
    rowGroup = (seq_len(ny) - 1) %/% factor + 1
    colGroup = (seq_len(nx) - 1) %/% factor + 1
    byRow = rowsum(values, rowGroup) / factor
    out = t(rowsum(t(byRow), colGroup) / factor)
  } else {
    rowIndex = rep(seq_len(ny), each = factor)
    colIndex = rep(seq_len(nx), each = factor)
    out = values[rowIndex, colIndex, drop = FALSE]
  }
  dimnames(out) = NULL
  return(out)
}

#' Neighbouring cells of a grid cell
#'
#' @param cell A single cell number.
#' @param grid A grid from [makeGrid()].
#' @param type `"rook"` (edges), `"bishop"` (corners) or `"queen"` (both).
#' @return An integer vector of cell numbers, sorted.
#' @export
cellNeighbours = function(cell, grid, type = c("rook", "bishop", "queen")) {
  type = match.arg(type)
  stopifnot(length(cell) == 1)
  col = (cell - 1) %% grid$nx + 1
  row = (cell - 1) %/% grid$nx + 1
  steps = switch(type,
    rook = {
      dx = c(0, -1, 1, 0)
      dy = c(-1, 0, 0, 1)
      cbind(dx, dy)
    },
    bishop = {
      dx = c(-1, 1, -1, 1)
      dy = c(-1, -1, 1, 1)
      cbind(dx, dy)
    },
    queen = {
      dx = c(-1, 0, 1, -1, 1, -1, 0, 1)
      dy = c(-1, -1, -1, 0, 0, 1, 1, 1)
      cbind(dx, dy)
    }
  )
  nc = col + steps[, "dx"]
  nr = row + steps[, "dy"]
  keep = nc >= 1 & nc <= grid$nx & nr >= 1 & nr <= grid$ny
  return(sort(as.integer((nr[keep] - 1) * grid$nx + nc[keep])))
}
