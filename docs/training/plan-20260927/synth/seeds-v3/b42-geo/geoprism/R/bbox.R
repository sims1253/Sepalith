#' Bounding box of a set of points
#'
#' @param lon,lat Coordinates in degrees.
#' @param pad Padding in degrees added on every side.
#' @return A named numeric vector `c(xmin, ymin, xmax, ymax)`.
#' @export
bbox_from_points <- function(lon, lat, pad = 0) {
  ok <- !is.na(lon) & !is.na(lat)
  if (!any(ok)) {
    stop("At least one complete point is needed.", call. = FALSE)
  }
  lon <- lon[ok]
  lat <- lat[ok]

  # Lower corner
  xmin <- min(lon) - pad
  ymin <- min(lat) - pad
  xmin <- max(xmin, -180)
  ymin <- max(ymin, -90)

  # Upper corner
  xmax <- max(lon) + pad
  ymax <- max(lat) + pad
  xmax <- min(xmax, 180)
  ymax <- min(ymax, 90)
  c(xmin = xmin, ymin = ymin, xmax = xmax, ymax = ymax)
}

#' Union or intersection of two bounding boxes
#'
#' @param a,b Named numeric vectors `c(xmin, ymin, xmax, ymax)`.
#' @param op `"union"` or `"intersection"`.
#' @return A bounding box, or `NULL` when an intersection is empty.
#' @export
bbox_combine <- function(a, b, op = c("union", "intersection")) {
  op <- match.arg(op)
  sides <- c("xmin", "ymin", "xmax", "ymax")

  # Validate `a`
  if (!is.numeric(a) || !all(sides %in% names(a))) {
    stop(sprintf("`%s` must be a bounding box.", "a"), call. = FALSE)
  }
  if (a[["xmin"]] > a[["xmax"]] || a[["ymin"]] > a[["ymax"]]) {
    stop(sprintf("`%s` has min greater than max.", "a"), call. = FALSE)
  }

  # Validate `b`
  if (!is.numeric(b) || !all(sides %in% names(b))) {
    stop(sprintf("`%s` must be a bounding box.", "b"), call. = FALSE)
  }
  if (b[["xmin"]] > b[["xmax"]] || b[["ymin"]] > b[["ymax"]]) {
    stop(sprintf("`%s` has min greater than max.", "b"), call. = FALSE)
  }

  if (op == "union") {
    xmin <- min(a[["xmin"]], b[["xmin"]])
    ymin <- min(a[["ymin"]], b[["ymin"]])
    xmax <- max(a[["xmax"]], b[["xmax"]])
    ymax <- max(a[["ymax"]], b[["ymax"]])
  } else {
    xmin <- max(a[["xmin"]], b[["xmin"]])
    ymin <- max(a[["ymin"]], b[["ymin"]])
    xmax <- min(a[["xmax"]], b[["xmax"]])
    ymax <- min(a[["ymax"]], b[["ymax"]])
    if (xmin > xmax || ymin > ymax) {
      return(NULL)
    }
  }
  c(xmin = xmin, ymin = ymin, xmax = xmax, ymax = ymax)
}

#' Grow or shrink a bounding box
#'
#' @param bbox A named numeric vector `c(xmin, ymin, xmax, ymax)`.
#' @param by Amount to add on each side; negative values shrink.
#' @param unit How `by` is read: `"degrees"`, `"fraction"` of the box size,
#'   `"meters"` on a sphere, or `"percent"` of the box size.
#' @param radius Sphere radius in metres, used for `"meters"`.
#' @return The adjusted bounding box.
#' @export
bbox_expand <- function(bbox, by, unit = c("degrees", "fraction", "meters", "percent"),
                        radius = 6371008.8) {
  unit <- match.arg(unit)
  width <- bbox[["xmax"]] - bbox[["xmin"]]
  height <- bbox[["ymax"]] - bbox[["ymin"]]
  mid_lat <- (bbox[["ymin"]] + bbox[["ymax"]]) / 2
  step <- switch(unit,
    degrees = {
      dx <- by
      dy <- by
      c(dx = dx, dy = dy)
    },
    fraction = {
      dx <- by * width
      dy <- by * height
      c(dx = dx, dy = dy)
    },
    meters = {
      dy <- by / radius * 180 / pi
      dx <- dy / cos(mid_lat * pi / 180)
      c(dx = dx, dy = dy)
    },
    percent = {
      dx <- by / 100 * width
      dy <- by / 100 * height
      c(dx = dx, dy = dy)
    }
  )

  # Move the min sides down and the max sides up
  out <- bbox
  out[["xmin"]] <- max(bbox[["xmin"]] - step[["dx"]], -180)
  out[["ymin"]] <- max(bbox[["ymin"]] - step[["dy"]], -90)
  out[["xmax"]] <- min(bbox[["xmax"]] + step[["dx"]], 180)
  out[["ymax"]] <- min(bbox[["ymax"]] + step[["dy"]], 90)
  if (out[["xmin"]] > out[["xmax"]] || out[["ymin"]] > out[["ymax"]]) {
    stop("`by` shrinks the box to nothing.", call. = FALSE)
  }
  out
}

#' Which points fall inside a bounding box
#'
#' Boxes with `xmin > xmax` are taken to cross the antimeridian.
#'
#' @param bbox A named numeric vector `c(xmin, ymin, xmax, ymax)`.
#' @param lon,lat Coordinates in degrees.
#' @param edges Whether points on an edge count as inside.
#' @return A logical vector, one element per point.
#' @export
bbox_contains <- function(bbox, lon, lat, edges = TRUE) {
  xmin <- bbox[["xmin"]]
  xmax <- bbox[["xmax"]]
  ymin <- bbox[["ymin"]]
  ymax <- bbox[["ymax"]]
  above <- if (edges) `>=` else `>`
  below <- if (edges) `<=` else `<`

  # Test along x
  if (xmin <= xmax) {
    in_x <- above(lon, xmin) & below(lon, xmax)
  } else {
    in_x <- above(lon, xmin) | below(lon, xmax)
  }

  # Test along y
  if (ymin <= ymax) {
    in_y <- above(lat, ymin) & below(lat, ymax)
  } else {
    stop("`bbox` has ymin greater than ymax.", call. = FALSE)
  }
  in_x & in_y & !is.na(lon) & !is.na(lat)
}

#' Split a bounding box into a regular grid of boxes
#'
#' @param bbox A named numeric vector `c(xmin, ymin, xmax, ymax)`.
#' @param nx,ny Number of columns and rows.
#' @return A data frame with columns `col`, `row`, `xmin`, `ymin`, `xmax`
#'   and `ymax`, one row per cell, row-major from the lower left.
#' @export
bbox_split <- function(bbox, nx, ny) {
  stopifnot(nx >= 1, ny >= 1)

  # Column edges
  x_step <- (bbox[["xmax"]] - bbox[["xmin"]]) / nx
  x_edges <- bbox[["xmin"]] + x_step * (0:nx)
  x_lo <- x_edges[-(nx + 1)]
  x_hi <- x_edges[-1]

  # Row edges
  y_step <- (bbox[["ymax"]] - bbox[["ymin"]]) / ny
  y_edges <- bbox[["ymin"]] + y_step * (0:ny)
  y_lo <- y_edges[-(ny + 1)]
  y_hi <- y_edges[-1]

  cells <- expand.grid(col = seq_len(nx), row = seq_len(ny))
  data.frame(
    col = cells$col,
    row = cells$row,
    xmin = x_lo[cells$col],
    ymin = y_lo[cells$row],
    xmax = x_hi[cells$col],
    ymax = y_hi[cells$row]
  )
}

#' Clip a bounding box to limits
#'
#' Each side that lies outside `limits` is moved onto it and reported.
#'
#' @param bbox A named numeric vector `c(xmin, ymin, xmax, ymax)`.
#' @param limits A bounding box to clip against.
#' @param quiet Whether to suppress the message about clipped sides.
#' @return The clipped bounding box with attribute `clipped`, a character
#'   vector of the sides that moved.
#' @export
bbox_clip <- function(bbox, limits = c(xmin = -180, ymin = -90, xmax = 180, ymax = 90),
                      quiet = FALSE) {
  clipped <- character()

  # Clip `xmin`
  if (bbox[["xmin"]] < limits[["xmin"]]) {
    bbox[["xmin"]] <- limits[["xmin"]]
    clipped <- c(clipped, "xmin")
  }

  # Clip `ymin`
  if (bbox[["ymin"]] < limits[["ymin"]]) {
    bbox[["ymin"]] <- limits[["ymin"]]
    clipped <- c(clipped, "ymin")
  }

  # Clip `xmax`
  if (bbox[["xmax"]] > limits[["xmax"]]) {
    bbox[["xmax"]] <- limits[["xmax"]]
    clipped <- c(clipped, "xmax")
  }

  # Clip `ymax`
  if (bbox[["ymax"]] > limits[["ymax"]]) {
    bbox[["ymax"]] <- limits[["ymax"]]
    clipped <- c(clipped, "ymax")
  }
  if (!quiet && length(clipped) > 0) {
    message(sprintf("Clipped side(s): %s.", paste(clipped, collapse = ", ")))
  }
  attr(bbox, "clipped") <- clipped
  bbox
}
