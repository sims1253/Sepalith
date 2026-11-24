#' Validate an Extent
#'
#' Checks the four bounds `xmin`, `xmax`, `ymin` and `ymax` in turn, then
#' their ordering along x and y, then `crs`.
#'
#' @param object An `Extent` object.
#' @return `TRUE`, or a character vector describing every problem found.
#' @noRd
valid_extent <- function(object) {
  problems <- character()

  if (length(object@xmin) != 1L || !is.finite(object@xmin)) {
    problems <- c(problems, "`xmin` must be a single finite number.")
  }
  if (length(object@xmax) != 1L || !is.finite(object@xmax)) {
    problems <- c(problems, "`xmax` must be a single finite number.")
  }
  if (length(object@ymin) != 1L || !is.finite(object@ymin)) {
    problems <- c(problems, "`ymin` must be a single finite number.")
  }
  if (length(object@ymax) != 1L || !is.finite(object@ymax)) {
    problems <- c(problems, "`ymax` must be a single finite number.")
  }
  if (length(problems) > 0L) {
    return(problems)
  }

  # Degenerate extents (a line or a point) are allowed
  if (object@xmin > object@xmax) {
    problems <- c(problems, "`xmin` must not exceed `xmax`.")
  }
  if (object@ymin > object@ymax) {
    problems <- c(problems, "`ymin` must not exceed `ymax`.")
  }

  if (length(object@crs) != 1L || is.na(object@crs)) {
    problems <- c(problems, "`crs` must be a single string, possibly empty.")
  }
  if (length(problems) == 0L) {
    return(TRUE)
  }
  return(problems)
}

#' A rectangular extent
#'
#' @slot xmin,xmax Bounds along the x axis.
#' @slot ymin,ymax Bounds along the y axis.
#' @slot crs Coordinate reference system label, or `""` when unknown.
#' @export
setClass(
  "Extent",
  representation(
    xmin = "numeric",
    xmax = "numeric",
    ymin = "numeric",
    ymax = "numeric",
    crs = "character"
  ),
  prototype(xmin = 0, xmax = 1, ymin = 0, ymax = 1, crs = ""),
  validity = valid_extent
)

#' Create an extent
#'
#' Bounds may be given as one vector `c(xmin, xmax, ymin, ymax)` or as two
#' vectors `x` and `y` of length two. Bounds along each axis may come in
#' either order; they are sorted before the object is built.
#'
#' @param x A numeric vector of length two with the x bounds, or of length
#'   four with all bounds.
#' @param y A numeric vector of length two with the y bounds, or `NULL`.
#' @param crs Coordinate reference system label.
#' @return An `Extent` object.
#' @family extent
#' @export
extent <- function(x, y = NULL, crs = "") {
  # A single vector holds xmin, xmax, ymin, ymax
  if (is.null(y)) {
    if (length(x) != 4L) {
      stop("Without `y`, `x` must have length 4.", call. = FALSE)
    }
    y <- x[3:4]
    x <- x[1:2]
  }
  if (!is.numeric(x) || length(x) != 2L) {
    stop(sprintf("`%s` must be numeric with length 2.", "x"), call. = FALSE)
  }
  if (!is.numeric(y) || length(y) != 2L) {
    stop(sprintf("`%s` must be numeric with length 2.", "y"), call. = FALSE)
  }
  if (!is.character(crs) || length(crs) != 1L) {
    stop("`crs` must be a single string.", call. = FALSE)
  }

  x <- sort(as.numeric(x))
  y <- sort(as.numeric(y))
  return(new("Extent", xmin = x[1], xmax = x[2], ymin = y[1], ymax = y[2], crs = crs))
}

#' Extent bound setters
#'
#' One setter per slot, used by the `xmin<-`, `xmax<-`, `ymin<-` and
#' `ymax<-` replacement methods.
#'
#' @param x An `Extent` object.
#' @param value A single number.
#' @return The modified `Extent` object.
#' @noRd
set_xmin <- function(x, value) {
  if (!is.numeric(value) || length(value) != 1L) {
    stop("`value` must be a single number.", call. = FALSE)
  }
  if (value > x@xmax) {
    stop(sprintf("New `%s` (%g) is beyond `%s` (%g).", "xmin", value, "xmax", x@xmax),
      call. = FALSE)
  }
  x@xmin <- value
  return(x)
}

#' @rdname set_xmin
#' @noRd
set_xmax <- function(x, value) {
  if (!is.numeric(value) || length(value) != 1L) {
    stop("`value` must be a single number.", call. = FALSE)
  }
  if (value < x@xmin) {
    stop(sprintf("New `%s` (%g) is beyond `%s` (%g).", "xmax", value, "xmin", x@xmin),
      call. = FALSE)
  }
  x@xmax <- value
  return(x)
}

#' @rdname set_xmin
#' @noRd
set_ymin <- function(x, value) {
  if (!is.numeric(value) || length(value) != 1L) {
    stop("`value` must be a single number.", call. = FALSE)
  }
  if (value > x@ymax) {
    stop(sprintf("New `%s` (%g) is beyond `%s` (%g).", "ymin", value, "ymax", x@ymax),
      call. = FALSE)
  }
  x@ymin <- value
  return(x)
}

#' @rdname set_xmin
#' @noRd
set_ymax <- function(x, value) {
  if (!is.numeric(value) || length(value) != 1L) {
    stop("`value` must be a single number.", call. = FALSE)
  }
  if (value < x@ymin) {
    stop(sprintf("New `%s` (%g) is beyond `%s` (%g).", "ymax", value, "ymin", x@ymin),
      call. = FALSE)
  }
  x@ymax <- value
  return(x)
}

setGeneric("xmin<-", function(x, value) standardGeneric("xmin<-"))
setGeneric("xmax<-", function(x, value) standardGeneric("xmax<-"))
setGeneric("ymin<-", function(x, value) standardGeneric("ymin<-"))
setGeneric("ymax<-", function(x, value) standardGeneric("ymax<-"))

#' @rdname extent-bounds
#' @export
setReplaceMethod("xmin", "Extent", function(x, value) set_xmin(x, value))

#' @rdname extent-bounds
#' @export
setReplaceMethod("xmax", "Extent", function(x, value) set_xmax(x, value))

#' @rdname extent-bounds
#' @export
setReplaceMethod("ymin", "Extent", function(x, value) set_ymin(x, value))

#' @rdname extent-bounds
#' @export
setReplaceMethod("ymax", "Extent", function(x, value) set_ymax(x, value))

#' Grow or shrink an extent
#'
#' Moves every bound outwards by `dx` along x and `dy` along y. Negative
#' values shrink the extent; shrinking past the centre collapses that axis
#' onto its midpoint.
#'
#' @param x An `Extent` object.
#' @param dx Distance added on both sides along x.
#' @param dy Distance added on both sides along y. Defaults to `dx`.
#' @return The buffered `Extent` object.
#' @family extent
#' @export
extent_buffer <- function(x, dx, dy = dx) {
  # Along x
  if (x@xmax - x@xmin + 2 * dx < 0) {
    mid <- (x@xmin + x@xmax) / 2
    x@xmin <- mid
    x@xmax <- mid
  } else {
    x@xmin <- x@xmin - dx
    x@xmax <- x@xmax + dx
  }

  # Along y
  if (x@ymax - x@ymin + 2 * dy < 0) {
    mid <- (x@ymin + x@ymax) / 2
    x@ymin <- mid
    x@ymax <- mid
  } else {
    x@ymin <- x@ymin - dy
    x@ymax <- x@ymax + dy
  }
  return(x)
}

#' Snap an extent to a grid
#'
#' @param x An `Extent` object.
#' @param resolution Grid cell size, a single positive number.
#' @param how `"out"` snaps every bound outwards so the result covers `x`,
#'   `"in"` snaps inwards so the result lies inside `x`, and `"near"` snaps
#'   each bound to the nearest grid line.
#' @return The snapped `Extent` object.
#' @family extent
#' @export
extent_snap <- function(x, resolution, how = c("out", "in", "near")) {
  how <- match.arg(how)
  if (!is.numeric(resolution) || length(resolution) != 1L || !(resolution > 0)) {
    stop("`resolution` must be a single positive number.", call. = FALSE)
  }
  r <- resolution
  switch(how,
    out = {
      x@xmin <- floor(x@xmin / r) * r
      x@xmax <- ceiling(x@xmax / r) * r
      x@ymin <- floor(x@ymin / r) * r
      x@ymax <- ceiling(x@ymax / r) * r
    },
    "in" = {
      x@xmin <- ceiling(x@xmin / r) * r
      x@xmax <- floor(x@xmax / r) * r
      x@ymin <- ceiling(x@ymin / r) * r
      x@ymax <- floor(x@ymax / r) * r
    },
    near = {
      x@xmin <- round(x@xmin / r) * r
      x@xmax <- round(x@xmax / r) * r
      x@ymin <- round(x@ymin / r) * r
      x@ymax <- round(x@ymax / r) * r
    }
  )

  # Snapping inwards can invert an extent narrower than one cell
  if (x@xmin > x@xmax || x@ymin > x@ymax) {
    stop("The extent is narrower than `resolution` along at least one axis.", call. = FALSE)
  }
  return(x)
}

#' @export
setMethod("show", "Extent", function(object) {
  crs <- if (nzchar(object@crs)) object@crs else "unknown crs"
  cat("<Extent>", crs, "\n")
  cat(sprintf("  x: [%g, %g]  width:  %g\n", object@xmin, object@xmax, object@xmax - object@xmin))
  cat(sprintf("  y: [%g, %g]  height: %g\n", object@ymin, object@ymax, object@ymax - object@ymin))
  invisible(object)
})
