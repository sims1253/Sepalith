#' Check that an annotation has the fields its type needs
#'
#' @param args Named list of annotation fields.
#' @param fields Required field names; each must be numeric unless it is
#'   `label`.
#' @param type Annotation type used in error messages.
#' @return `args`, invisibly.
#' @noRd
need_fields <- function(args, fields, type) {
  missing_fields <- setdiff(fields, names(args))
  if (length(missing_fields) > 0L) {
    stop(sprintf("A `%s` annotation needs: %s.", type, paste(missing_fields, collapse = ", ")),
         call. = FALSE)
  }
  for (field in setdiff(fields, "label")) {
    if (!is.numeric(args[[field]])) {
      stop(sprintf("Field `%s` of a `%s` annotation must be numeric.", field, type),
           call. = FALSE)
    }
  }
  invisible(args)
}

#' Create an annotation
#'
#' @description
#' Annotations are small lists that [draw_annotation()] turns into base
#' graphics calls. Required fields by type:
#' `hline` needs `y`; `vline` needs `x`; `band_x` needs `from`, `to`;
#' `band_y` needs `from`, `to`; `text` needs `x`, `y`, `label`;
#' `arrow` needs `x0`, `y0`, `x1`, `y1`. Bands are stored with `from` not
#' greater than `to`, and text without `pos` is centred (`pos = NULL`).
#'
#' @param type Annotation type.
#' @param ... Named fields of the annotation.
#' @return A `plotloom_annotation` list with element `type` and the fields.
#' @export
annotation <- function(type = c("hline", "vline", "band_x", "band_y", "text", "arrow"), ...) {
  type <- match.arg(type)
  args <- list(...)
  switch(type,
    hline = {
      need_fields(args, "y", type)
    },
    vline = {
      need_fields(args, "x", type)
    },
    band_x = {
      need_fields(args, c("from", "to"), type)
      args[c("from", "to")] <- list(min(args$from, args$to), max(args$from, args$to))
    },
    band_y = {
      need_fields(args, c("from", "to"), type)
      args[c("from", "to")] <- list(min(args$from, args$to), max(args$from, args$to))
    },
    text = {
      need_fields(args, c("x", "y", "label"), type)
      args$label <- as.character(args$label)
    },
    arrow = {
      need_fields(args, c("x0", "y0", "x1", "y1"), type)
    }
  )
  structure(c(list(type = type), args), class = "plotloom_annotation")
}

#' Draw an annotation
#'
#' @param ann A `plotloom_annotation` from [annotation()].
#' @param col Colour used for lines, text and band borders.
#' @param fill Fill colour for bands.
#' @param lty Line type for lines and arrows.
#' @return Invisibly, the annotation type.
#' @export
draw_annotation <- function(ann, col = "grey20", fill = "#0000001A", lty = 2L) {
  if (!inherits(ann, "plotloom_annotation")) {
    stop("`ann` must be a plotloom_annotation.", call. = FALSE)
  }
  usr <- par("usr")
  switch(ann$type,
    hline = {
      abline(h = ann$y, col = col, lty = lty)
    },
    vline = {
      abline(v = ann$x, col = col, lty = lty)
    },
    band_x = {
      rect(ann$from, usr[3], ann$to, usr[4], col = fill, border = NA)
    },
    band_y = {
      rect(usr[1], ann$from, usr[2], ann$to, col = fill, border = NA)
    },
    text = {
      text(ann$x, ann$y, labels = ann$label, col = col, pos = ann$pos)
    },
    arrow = {
      arrows(ann$x0, ann$y0, ann$x1, ann$y1, col = col, lty = lty, length = 0.08)
    }
  )
  invisible(ann$type)
}

#' Draw several annotations
#'
#' @param anns A list of `plotloom_annotation` objects.
#' @param ... Passed to [draw_annotation()].
#' @return Invisibly, a character vector of the drawn types.
#' @export
draw_annotations <- function(anns, ...) {
  types <- character(length(anns))
  for (i in seq_along(anns)) {
    types[[i]] <- draw_annotation(anns[[i]], ...)
  }
  invisible(types)
}

#' Mark the centre of x and y
#'
#' Draws a reference line at the mean or median of `x`, of `y` or of both,
#' and labels each line with the statistic and its value.
#'
#' @param x,y Numeric vectors of coordinates.
#' @param which Which axes to mark.
#' @param stat Centre statistic.
#' @param col Line and label colour.
#' @param digits Digits used in the labels.
#' @return Invisibly, a named numeric vector of the marked centres.
#' @export
annotate_centre <- function(x, y, which = c("both", "x", "y"), stat = c("mean", "median"),
                            col = "firebrick", digits = 2L) {
  which <- match.arg(which)
  stat <- match.arg(stat)
  centre <- switch(stat, mean = mean, median = stats::median)
  usr <- par("usr")
  out <- numeric()

  if (which %in% c("x", "both")) {
    cx <- centre(x, na.rm = TRUE)
    abline(v = cx, col = col, lty = 2L)
    label <- sprintf("%s = %s", stat, format(round(cx, digits)))
    text(cx, usr[4], labels = label, col = col, pos = 1L, cex = 0.8)
    out[["x"]] <- cx
  }
  if (which %in% c("y", "both")) {
    cy <- centre(y, na.rm = TRUE)
    abline(h = cy, col = col, lty = 2L)
    label <- sprintf("%s = %s", stat, format(round(cy, digits)))
    text(usr[2], cy, labels = label, col = col, pos = 2L, cex = 0.8)
    out[["y"]] <- cy
  }
  invisible(out)
}

#' Label the extreme points
#'
#' Labels the `n` points with the smallest and/or the `n` points with the
#' largest `y` values and circles them.
#'
#' @param x,y Numeric vectors of coordinates.
#' @param labels Point labels; defaults to the indices.
#' @param n Number of points to label at each end.
#' @param which Which ends to label.
#' @param col Label colour.
#' @return Invisibly, a list with elements `min` and `max` holding the
#'   indices of the labelled points (empty when an end was not labelled).
#' @export
annotate_extremes <- function(x, y, labels = seq_along(y), n = 1L,
                              which = c("both", "min", "max"), col = "grey20") {
  which <- match.arg(which)
  if (length(x) != length(y) || length(labels) != length(y)) {
    stop("`x`, `y` and `labels` must have the same length.", call. = FALSE)
  }
  ord <- order(y, na.last = NA)
  n <- min(n, length(ord))
  lo <- integer()
  hi <- integer()

  # lowest values: label below the point
  if (which %in% c("min", "both")) {
    lo <- utils::head(ord, n)
    text(x[lo], y[lo], labels = labels[lo], pos = 1L, col = col, cex = 0.8)
    points(x[lo], y[lo], pch = 1L, cex = 1.5, col = col)
  }
  # highest values: label above the point
  if (which %in% c("max", "both")) {
    hi <- rev(utils::tail(ord, n))
    text(x[hi], y[hi], labels = labels[hi], pos = 3L, col = col, cex = 0.8)
    points(x[hi], y[hi], pch = 1L, cex = 1.5, col = col)
  }
  invisible(list(min = lo, max = hi))
}

#' Shade an interval along one axis
#'
#' @param lower,upper Interval limits in user coordinates; `-Inf` and `Inf`
#'   extend to the plot edge.
#' @param axis Axis the interval lies on.
#' @param fill Fill colour.
#' @param label Optional label drawn inside the interval.
#' @return Invisibly, the drawn interval as `c(lower, upper)` clipped to the
#'   plot region.
#' @export
shade_interval <- function(lower, upper, axis = c("x", "y"), fill = "#0000001A", label = NULL) {
  axis <- match.arg(axis)
  if (!is.numeric(lower) || length(lower) != 1L || is.na(lower)) {
    stop("`lower` must be a single number.", call. = FALSE)
  }
  if (!is.numeric(upper) || length(upper) != 1L || is.na(upper)) {
    stop("`upper` must be a single number.", call. = FALSE)
  }
  if (lower > upper) {
    stop("`lower` must not be greater than `upper`.", call. = FALSE)
  }
  usr <- par("usr")
  if (axis == "x") {
    lower <- max(lower, usr[1])
    upper <- min(upper, usr[2])
    rect(lower, usr[3], upper, usr[4], col = fill, border = NA)
    if (!is.null(label)) {
      text((lower + upper) / 2, usr[4], labels = label, pos = 1L, cex = 0.8)
    }
  } else {
    lower <- max(lower, usr[3])
    upper <- min(upper, usr[4])
    rect(usr[1], lower, usr[2], upper, col = fill, border = NA)
    if (!is.null(label)) {
      text(usr[2], (lower + upper) / 2, labels = label, pos = 2L, cex = 0.8)
    }
  }
  invisible(c(lower, upper))
}
