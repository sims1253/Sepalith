#' Arithmetic and comparison for spans
#'
#' @details Comparisons need a span on both sides: `==` and `!=` compare
#'   both ends, `<` and `<=` test whether `e1` ends before `e2` starts, `>`
#'   and `>=` test whether `e1` starts after `e2` ends. Arithmetic needs one
#'   span and one number: `+` and `-` shift both ends, `*` and `/` scale them.
#'
#' @param e1,e2 Operands; at least one is a `spanwise_span`.
#' @return A logical vector for comparisons, otherwise a `spanwise_span`.
#' @export
Ops.spanwise_span <- function(e1, e2) {
  if (nargs() == 1L) {
    stop(sprintf("Unary `%s` is not defined for <%s>.", .Generic, "spanwise_span"),
      call. = FALSE)
  }
  span_left <- inherits(e1, "spanwise_span")
  span_right <- inherits(e2, "spanwise_span")

  # Comparisons need a span on both sides
  if (.Generic %in% c("==", "!=", "<", ">", "<=", ">=")) {
    if (!(span_left && span_right)) {
      stop(sprintf("`%s` needs a <%s> on both sides.", .Generic, "spanwise_span"),
        call. = FALSE)
    }
    lhs <- unclass(e1)
    rhs <- unclass(e2)
    out <- switch(.Generic,
      "==" = lhs$start == rhs$start & lhs$end == rhs$end,
      "!=" = lhs$start != rhs$start | lhs$end != rhs$end,
      "<" = lhs$end < rhs$start,
      "<=" = lhs$end <= rhs$start,
      ">" = lhs$start > rhs$end,
      ">=" = lhs$start >= rhs$end
    )
    return(out)
  }

  # Arithmetic needs one span and one number
  amount <- if (span_left) e2 else e1
  if ((span_left && span_right) || !is.numeric(amount)) {
    stop(sprintf("`%s` needs a number on one side.", .Generic), call. = FALSE)
  }
  fields <- unclass(if (span_left) e1 else e2)
  if (!span_left && .Generic %in% c("-", "/")) {
    stop(sprintf("`%s` needs the <%s> on the left.", .Generic, "spanwise_span"),
      call. = FALSE)
  }
  amount <- as.double(amount)
  switch(.Generic,
    "+" = {
      start <- fields$start + amount
      end <- fields$end + amount
    },
    "-" = {
      start <- fields$start - amount
      end <- fields$end - amount
    },
    "*" = {
      start <- fields$start * amount
      end <- fields$end * amount
    },
    "/" = {
      start <- fields$start / amount
      end <- fields$end / amount
    },
    stop(sprintf("`%s` is not defined for <%s>.", .Generic, "spanwise_span"), call. = FALSE)
  )
  out <- new_span(start, end, fields$label)
  validate_span(out)
  return(out)
}

#' Arithmetic and comparison for gaps
#'
#' @details Comparisons need a gap on both sides and compare widths.
#'   Arithmetic needs one gap and one number: `+` and `-` move `at`, `*` and
#'   `/` scale `width`.
#'
#' @param e1,e2 Operands; at least one is a `spanwise_gap`.
#' @return A logical vector for comparisons, otherwise a `spanwise_gap`.
#' @export
Ops.spanwise_gap <- function(e1, e2) {
  if (nargs() == 1L) {
    stop(sprintf("Unary `%s` is not defined for <%s>.", .Generic, "spanwise_gap"),
      call. = FALSE)
  }
  gap_left <- inherits(e1, "spanwise_gap")
  gap_right <- inherits(e2, "spanwise_gap")

  # Comparisons need a gap on both sides
  if (.Generic %in% c("==", "!=", "<", ">", "<=", ">=")) {
    if (!(gap_left && gap_right)) {
      stop(sprintf("`%s` needs a <%s> on both sides.", .Generic, "spanwise_gap"),
        call. = FALSE)
    }
    lhs <- unclass(e1)
    rhs <- unclass(e2)
    out <- get(.Generic)(lhs$width, rhs$width)
    return(out)
  }

  # Arithmetic needs one gap and one number
  amount <- if (gap_left) e2 else e1
  if ((gap_left && gap_right) || !is.numeric(amount)) {
    stop(sprintf("`%s` needs a number on one side.", .Generic), call. = FALSE)
  }
  fields <- unclass(if (gap_left) e1 else e2)
  if (!gap_left && .Generic %in% c("-", "/")) {
    stop(sprintf("`%s` needs the <%s> on the left.", .Generic, "spanwise_gap"),
      call. = FALSE)
  }
  amount <- as.double(amount)
  at <- fields$at
  width <- fields$width
  switch(.Generic,
    "+" = at <- at + amount,
    "-" = at <- at - amount,
    "*" = width <- width * amount,
    "/" = width <- width / amount,
    stop(sprintf("`%s` is not defined for <%s>.", .Generic, "spanwise_gap"), call. = FALSE)
  )
  out <- new_gap(at, width)
  validate_gap(out)
  return(out)
}

#' Rounding for spans
#'
#' Only the rounding members of the Math group are supported; they are
#' applied to `start` and `end` alike.
#'
#' @param x An object of class `spanwise_span`.
#' @param ... Passed on, e.g. `digits` for [round()].
#' @return An object of class `spanwise_span`.
#' @export
Math.spanwise_span <- function(x, ...) {
  allowed <- c("floor", "ceiling", "round", "signif", "trunc")
  if (!.Generic %in% allowed) {
    stop(sprintf("`%s` is not defined for <%s>.", .Generic, "spanwise_span"), call. = FALSE)
  }
  fun <- match.fun(.Generic)
  fields <- unclass(x)
  start <- as.double(fun(fields$start, ...))
  end <- as.double(fun(fields$end, ...))
  out <- new_span(start, end, fields$label)
  validate_span(out)
  return(out)
}

#' Rounding for gaps
#'
#' Only the rounding members of the Math group are supported; they are
#' applied to `at` and `width` alike.
#'
#' @param x An object of class `spanwise_gap`.
#' @param ... Passed on, e.g. `digits` for [round()].
#' @return An object of class `spanwise_gap`.
#' @export
Math.spanwise_gap <- function(x, ...) {
  allowed <- c("floor", "ceiling", "round", "signif", "trunc")
  if (!.Generic %in% allowed) {
    stop(sprintf("`%s` is not defined for <%s>.", .Generic, "spanwise_gap"), call. = FALSE)
  }
  fun <- match.fun(.Generic)
  fields <- unclass(x)
  at <- as.double(fun(fields$at, ...))
  width <- as.double(fun(fields$width, ...))
  out <- new_gap(at, width)
  validate_gap(out)
  return(out)
}

#' Clip spans and gaps to a window
#'
#' Elements are clipped to `[lower, upper]`; a `NULL` bound leaves that side
#' open. Elements entirely outside the window shrink to zero width at the
#' nearest bound and are dropped unless `drop = FALSE`.
#'
#' @param x An object of class `spanwise_span` or `spanwise_gap`.
#' @param lower,upper The window bounds, or `NULL`.
#' @param drop Whether to drop elements that lie outside the window.
#' @param ... Unused.
#' @return An object of the same class as `x`.
#' @export
span_clip <- function(x, lower = NULL, upper = NULL, drop = TRUE, ...) {
  if (!is.null(lower) && !is.null(upper) && lower > upper) {
    stop("`lower` must not exceed `upper`.", call. = FALSE)
  }
  UseMethod("span_clip")
}

#' @rdname span_clip
#' @export
span_clip.spanwise_span <- function(x, lower = NULL, upper = NULL, drop = TRUE, ...) {
  fields <- unclass(x)
  start <- fields$start
  end <- fields$end
  keep <- rep(TRUE, length(start))

  # Clip at the lower bound
  if (!is.null(lower)) {
    keep <- keep & end >= lower
    start <- pmax(start, lower)
    end <- pmax(end, lower)
  }
  # Clip at the upper bound
  if (!is.null(upper)) {
    keep <- keep & start <= upper
    start <- pmin(start, upper)
    end <- pmin(end, upper)
  }

  if (!drop) {
    keep[] <- TRUE
  }
  out <- new_span(start[keep], end[keep], fields$label[keep])
  return(out)
}

#' @rdname span_clip
#' @export
span_clip.spanwise_gap <- function(x, lower = NULL, upper = NULL, drop = TRUE, ...) {
  fields <- unclass(x)
  start <- fields$at
  end <- fields$at + fields$width
  keep <- rep(TRUE, length(start))

  # Clip at the lower bound
  if (!is.null(lower)) {
    keep <- keep & end >= lower
    start <- pmax(start, lower)
    end <- pmax(end, lower)
  }
  # Clip at the upper bound
  if (!is.null(upper)) {
    keep <- keep & start <= upper
    start <- pmin(start, upper)
    end <- pmin(end, upper)
  }

  if (!drop) {
    keep[] <- TRUE
  }
  out <- new_gap(start[keep], end[keep] - start[keep])
  return(out)
}
