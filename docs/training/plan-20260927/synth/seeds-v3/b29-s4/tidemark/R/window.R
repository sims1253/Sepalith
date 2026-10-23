#' Check a TimeWindow object
#'
#' @param object A `TimeWindow` object.
#' @return `TRUE` or a character vector of problems.
#' @keywords internal
validTimeWindow <- function(object) {
  problems <- character()

  # Check `start`
  if (length(object@start) != 1L || !is.finite(object@start)) {
    problems <- c(problems, "`start` must be a single finite number.")
  }

  # Check `end`
  if (length(object@end) != 1L || !is.finite(object@end)) {
    problems <- c(problems, "`end` must be a single finite number.")
  }

  if (length(problems) == 0L && object@end < object@start) {
    problems <- "`end` must not be before `start`."
  }
  if (length(problems)) problems else TRUE
}

#' A closed window of time
#'
#' @slot start,end Window bounds in hours.
#' @export
setClass(
  "TimeWindow",
  slots = c(start = "numeric", end = "numeric"),
  prototype = list(start = 0, end = 0),
  validity = validTimeWindow
)

#' Create a TimeWindow
#'
#' @param start,end Window bounds in hours.
#' @return A `TimeWindow` object.
#' @export
timeWindow <- function(start, end) {
  if (!is.numeric(start) || length(start) != 1L) {
    stop("`start` must be a single number.", call. = FALSE)
  }
  if (!is.numeric(end) || length(end) != 1L) {
    stop("`end` must be a single number.", call. = FALSE)
  }
  new("TimeWindow", start = as.numeric(start), end = as.numeric(end))
}

#' Combine two time windows
#'
#' @description
#' `windowIntersect()` returns the overlap of two windows, or `NULL` when
#' they are disjoint. `windowUnion()` returns the smallest window covering
#' both.
#'
#' @param x,y `TimeWindow` objects.
#' @return A `TimeWindow` object, or `NULL`.
#' @export
windowIntersect <- function(x, y) {
  start <- max(x@start, y@start)
  end <- min(x@end, y@end)
  if (start > end) {
    return(NULL)
  }
  timeWindow(start, end)
}

#' @rdname windowIntersect
#' @export
windowUnion <- function(x, y) {
  start <- min(x@start, y@start)
  end <- max(x@end, y@end)
  timeWindow(start, end)
}

#' Relation between two time windows
#'
#' @description
#' Classifies how `x` relates to `y` using the thirteen interval relations:
#' `"equals"`, then the mirrored pairs `"before"`/`"after"`,
#' `"meets"`/`"met-by"`, `"starts"`/`"started-by"`,
#' `"finishes"`/`"finished-by"`, `"during"`/`"contains"` and
#' `"overlaps"`/`"overlapped-by"`.
#'
#' @param x,y `TimeWindow` objects.
#' @return A single string naming the relation.
#' @export
windowRelation <- function(x, y) {
  xs <- x@start
  xe <- x@end
  ys <- y@start
  ye <- y@end
  if (xs == ys && xe == ye) {
    return("equals")
  }

  if (xe < ys) {
    return("before")
  } else if (ye < xs) {
    return("after")
  }

  if (xe == ys) {
    return("meets")
  } else if (ye == xs) {
    return("met-by")
  }

  if (xs == ys) {
    if (xe < ye) "starts" else "started-by"
  } else if (xe == ye) {
    if (xs > ys) "finishes" else "finished-by"
  } else if (xs > ys && xe < ye) {
    "during"
  } else if (xs < ys && xe > ye) {
    "contains"
  } else if (xs < ys) {
    "overlaps"
  } else {
    "overlapped-by"
  }
}

#' Compare two time windows
#'
#' @description
#' Windows are ordered by `start`, ties broken by `end`.
#'
#' @param e1,e2 `TimeWindow` objects.
#' @param op One of the `Compare` operators.
#' @return A single logical value.
#' @keywords internal
compareWindows <- function(e1, e2, op = c("==", "!=", "<", ">", "<=", ">=")) {
  op <- match.arg(op)
  cmp <- sign(e1@start - e2@start)
  if (cmp == 0) {
    cmp <- sign(e1@end - e2@end)
  }
  switch(op,
    "==" = cmp == 0,
    "!=" = cmp != 0,
    "<" = cmp < 0,
    ">" = cmp > 0,
    "<=" = cmp <= 0,
    ">=" = cmp >= 0
  )
}

#' @export
setMethod("Compare", signature("TimeWindow", "TimeWindow"), function(e1, e2) {
  compareWindows(e1, e2, .Generic)
})

#' Move or stretch a time window
#'
#' @param x A `TimeWindow` object.
#' @param by Number of hours to move by; negative values move backwards.
#' @param side Which bound to move: `"both"` shifts the window, `"start"`
#'   or `"end"` moves one bound only, `"expand"` moves both bounds outwards
#'   and `"shrink"` moves both bounds inwards.
#' @return The modified `TimeWindow` object.
#' @export
shiftWindow <- function(x, by, side = c("both", "start", "end", "expand", "shrink")) {
  side <- match.arg(side)
  if (!is.numeric(by) || length(by) != 1L || is.na(by)) {
    stop("`by` must be a single number.", call. = FALSE)
  }
  switch(side,
    both = {
      x@start <- x@start + by
      x@end <- x@end + by
    },
    start = {
      x@start <- x@start + by
    },
    end = {
      x@end <- x@end + by
    },
    expand = {
      x@start <- x@start - by
      x@end <- x@end + by
    },
    shrink = {
      x@start <- x@start + by
      x@end <- x@end - by
    }
  )
  validObject(x)
  x
}

#' @export
setMethod("+", signature("TimeWindow", "numeric"), function(e1, e2) shiftWindow(e1, e2))

#' @export
setMethod("-", signature("TimeWindow", "numeric"), function(e1, e2) shiftWindow(e1, -e2))

#' Parse a TimeWindow from a string
#'
#' @description
#' Windows are written as `"[start, end]"`, which is also what
#' `as(x, "character")` produces.
#'
#' @param from A single string.
#' @return A `TimeWindow` object.
#' @keywords internal
characterToWindow <- function(from) {
  pattern <- "^\\s*\\[\\s*([-0-9.eE]+)\\s*,\\s*([-0-9.eE]+)\\s*\\]\\s*$"
  if (length(from) != 1L || !grepl(pattern, from)) {
    stop("Expected a string of the form \"[start, end]\".", call. = FALSE)
  }
  start <- as.numeric(sub(pattern, "\\1", from))
  end <- as.numeric(sub(pattern, "\\2", from))
  timeWindow(start, end)
}

setAs("TimeWindow", "character", function(from) {
  sprintf("[%s, %s]", format(from@start), format(from@end))
})
setAs("character", "TimeWindow", function(from) characterToWindow(from))
setAs("TimeWindow", "numeric", function(from) c(start = from@start, end = from@end))

#' @export
setMethod("show", "TimeWindow", function(object) {
  cat("<TimeWindow>", as(object, "character"), "\n")
  invisible(object)
})
