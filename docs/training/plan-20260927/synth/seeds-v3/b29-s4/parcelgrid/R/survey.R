#' Validate SurveyPoint objects
#'
#' The coordinate slots `x`, `y` and `z` are checked in turn, then
#' `accuracy`.
#'
#' @param object A `SurveyPoint` object.
#' @return `TRUE`, or a character vector describing every problem found.
#' @noRd
valid_survey_point <- function(object) {
  n <- length(object@x)
  problems <- character()

  if (!all(is.finite(object@x))) {
    problems <- c(problems, "`x` must contain finite numbers only.")
  }
  if (length(object@y) != n || !all(is.finite(object@y))) {
    problems <- c(problems, sprintf("`y` must contain %d finite numbers.", n))
  }
  if (length(object@z) != n || !all(is.finite(object@z))) {
    problems <- c(problems, sprintf("`z` must contain %d finite numbers.", n))
  }
  if (length(object@accuracy) != n || any(object@accuracy < 0)) {
    problems <- c(problems, sprintf("`accuracy` must contain %d non-negative numbers.", n))
  }

  if (length(problems) == 0L) {
    return(TRUE)
  }
  return(problems)
}

#' Surveyed points
#'
#' @slot x,y Planar coordinates in metres.
#' @slot z Elevation in metres.
#' @slot accuracy Horizontal accuracy (one standard error) in metres.
#' @export
setClass(
  "SurveyPoint",
  representation(x = "numeric", y = "numeric", z = "numeric", accuracy = "numeric"),
  validity = valid_survey_point
)

#' Create surveyed points
#'
#' @param x,y Planar coordinates in metres.
#' @param z Elevation in metres; recycled to the number of points.
#' @param accuracy Horizontal accuracy in metres; recycled to the number of
#'   points.
#' @return A `SurveyPoint` object.
#' @export
survey_point <- function(x, y, z = 0, accuracy = 0.05) {
  n <- length(x)
  if (!is.numeric(x) || !is.numeric(y)) {
    stop("`x` and `y` must be numeric.", call. = FALSE)
  }
  if (length(y) != n) {
    stop(sprintf("`%s` must have length %d, like `x`.", "y", n), call. = FALSE)
  }
  if (!length(z) %in% c(1L, n)) {
    stop(sprintf("`%s` must have length 1 or %d.", "z", n), call. = FALSE)
  }
  if (!length(accuracy) %in% c(1L, n)) {
    stop(sprintf("`%s` must have length 1 or %d.", "accuracy", n), call. = FALSE)
  }

  return(new(
    "SurveyPoint",
    x = as.numeric(x),
    y = as.numeric(y),
    z = rep_len(as.numeric(z), n),
    accuracy = rep_len(as.numeric(accuracy), n)
  ))
}

#' Arithmetic between survey points and numbers
#'
#' `points op number` and `number op points` are handled by mirrored
#' branches. Addition and subtraction translate every coordinate;
#' multiplication and division scale the coordinates and the accuracy. A
#' number cannot be divided by points.
#'
#' @param e1,e2 The operands; exactly one of them is a `SurveyPoint`.
#' @param op The operator name, as found in `.Generic`.
#' @return A `SurveyPoint` object.
#' @noRd
point_arith <- function(e1, e2, op) {
  if (is(e1, "SurveyPoint")) {
    switch(op,
      "+" = {
        e1@x <- e1@x + e2
        e1@y <- e1@y + e2
        e1@z <- e1@z + e2
      },
      "-" = {
        e1@x <- e1@x - e2
        e1@y <- e1@y - e2
        e1@z <- e1@z - e2
      },
      "*" = {
        e1@x <- e1@x * e2
        e1@y <- e1@y * e2
        e1@z <- e1@z * e2
        e1@accuracy <- e1@accuracy * abs(e2)
      },
      "/" = {
        e1@x <- e1@x / e2
        e1@y <- e1@y / e2
        e1@z <- e1@z / e2
        e1@accuracy <- e1@accuracy / abs(e2)
      },
      stop(sprintf("Operator `%s` is not defined for survey points.", op), call. = FALSE)
    )
    validObject(e1)
    return(e1)
  } else {
    switch(op,
      "+" = {
        e2@x <- e1 + e2@x
        e2@y <- e1 + e2@y
        e2@z <- e1 + e2@z
      },
      "-" = {
        e2@x <- e1 - e2@x
        e2@y <- e1 - e2@y
        e2@z <- e1 - e2@z
      },
      "*" = {
        e2@x <- e1 * e2@x
        e2@y <- e1 * e2@y
        e2@z <- e1 * e2@z
        e2@accuracy <- abs(e1) * e2@accuracy
      },
      stop(sprintf("Operator `%s` is not defined for survey points.", op), call. = FALSE)
    )
    validObject(e2)
    return(e2)
  }
}

#' @export
setMethod("Arith", signature("SurveyPoint", "numeric"), function(e1, e2) {
  point_arith(e1, e2, .Generic)
})

#' @export
setMethod("Arith", signature("numeric", "SurveyPoint"), function(e1, e2) {
  point_arith(e1, e2, .Generic)
})

#' @export
setMethod("Compare", signature("SurveyPoint", "numeric"), function(e1, e2) {
  callGeneric(e1@accuracy, e2)
})

#' Similarity transform of survey points
#'
#' Applies, in this order: 1. scaling about the origin, 2. a
#' counter-clockwise rotation by `angle` about the origin, 3. a translation
#' by `dx` and `dy`. Elevations are only shifted by `dz`.
#'
#' @param p A `SurveyPoint` object.
#' @param scale Scale factor.
#' @param angle Rotation angle in radians.
#' @param dx,dy,dz Translation in metres.
#' @return The transformed `SurveyPoint` object; accuracies are scaled too.
#' @export
point_transform <- function(p, scale = 1, angle = 0, dx = 0, dy = 0, dz = 0) {
  # 1. Scale
  x <- p@x * scale
  y <- p@y * scale

  # 2. Rotate
  rx <- x * cos(angle) - y * sin(angle)
  ry <- x * sin(angle) + y * cos(angle)

  # 3. Translate
  p@x <- rx + dx
  p@y <- ry + dy
  p@z <- p@z + dz
  p@accuracy <- p@accuracy * abs(scale)
  validObject(p)
  return(p)
}

#' Distance between survey points
#'
#' @param a,b `SurveyPoint` objects of equal length, or `b` of length one.
#' @param metric Distance metric. `"planar"` ignores elevation; the others
#'   use all three coordinates.
#' @return A numeric vector of distances in metres.
#' @export
point_distance <- function(a, b, metric = c("euclidean", "manhattan", "chebyshev", "planar")) {
  metric <- match.arg(metric)
  dx <- a@x - b@x
  dy <- a@y - b@y
  dz <- a@z - b@z
  switch(metric,
    euclidean = {
      return(sqrt(dx^2 + dy^2 + dz^2))
    },
    manhattan = {
      return(abs(dx) + abs(dy) + abs(dz))
    },
    chebyshev = {
      return(pmax(abs(dx), abs(dy), abs(dz)))
    },
    planar = {
      return(sqrt(dx^2 + dy^2))
    }
  )
}

#' Convert between survey points and matrices
#'
#' Matrices have the columns `x`, `y`, `z` and `accuracy`, one row per point.
#'
#' @param from The object to convert.
#' @return A numeric matrix, or a `SurveyPoint` object.
#' @noRd
point_to_matrix <- function(from) {
  out <- cbind(x = from@x, y = from@y, z = from@z, accuracy = from@accuracy)
  rownames(out) <- NULL
  return(out)
}

#' @rdname point_to_matrix
#' @noRd
matrix_to_point <- function(from) {
  missing <- setdiff(c("x", "y"), colnames(from))
  if (length(missing) > 0L) {
    stop("Matrix lacks column(s): ", toString(missing), call. = FALSE)
  }
  z <- if ("z" %in% colnames(from)) from[, "z"] else 0
  accuracy <- if ("accuracy" %in% colnames(from)) from[, "accuracy"] else 0.05
  return(survey_point(from[, "x"], from[, "y"], unname(z), unname(accuracy)))
}

setAs("SurveyPoint", "matrix", function(from) point_to_matrix(from))
setAs("matrix", "SurveyPoint", function(from) matrix_to_point(from))

#' Print survey points
#'
#' @param object A `SurveyPoint` object.
#' @param n Maximum number of points to list.
#' @return `object`, invisibly.
#' @importFrom stats median
#' @noRd
show_survey_point <- function(object, n = 5L) {
  total <- length(object@x)
  cat(sprintf("<SurveyPoint> %d points, median accuracy %.3f m\n", total,
    stats::median(object@accuracy)))
  shown <- seq_len(min(n, total))
  for (i in shown) {
    cat(sprintf("  %8.3f %8.3f %8.3f\n", object@x[i], object@y[i], object@z[i]))
  }
  if (total > n) {
    cat(sprintf("  ... and %d more\n", total - n))
  }
  return(invisible(object))
}

#' @export
setMethod("show", "SurveyPoint", function(object) show_survey_point(object))
