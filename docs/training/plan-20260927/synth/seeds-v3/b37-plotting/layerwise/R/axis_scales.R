#' Expand axis limits
#'
#' @param limits Numeric vector of length two.
#' @param mult Multiplicative expansion as a fraction of the range.
#' @param add Additive expansion in data units.
#' @param side Which ends of the range to expand.
#' @return The expanded limits, lower first.
#' @keywords internal
axis_expand <- function(limits, mult = 0.05, add = 0, side = c("both", "lower", "upper")) {
  side <- match.arg(side)
  if (!is.numeric(limits) || length(limits) != 2L || anyNA(limits)) {
    stop("`limits` must be two non-missing numbers.", call. = FALSE)
  }
  if (!is.numeric(mult) || length(mult) != 1L || is.na(mult) || mult < 0) {
    stop("`mult` must be a single non-negative number.", call. = FALSE)
  }
  if (!is.numeric(add) || length(add) != 1L || is.na(add) || add < 0) {
    stop("`add` must be a single non-negative number.", call. = FALSE)
  }

  delta <- mult * diff(range(limits)) + add
  lower <- min(limits)
  upper <- max(limits)
  if (side %in% c("both", "lower")) {
    lower <- lower - delta
  }
  if (side %in% c("both", "upper")) {
    upper <- upper + delta
  }
  return(c(lower, upper))
}

#' Label function for an axis style
#'
#' @param style Label style.
#' @param digits Digits after the decimal mark for the `"number"`,
#'   `"percent"` and `"unit"` styles.
#' @param unit Suffix for the `"unit"` style.
#' @return A function taking numeric breaks and returning labels; missing
#'   breaks get empty labels.
#' @export
axis_labeller <- function(style = c("number", "percent", "comma", "unit", "log10"),
                          digits = 1L, unit = "") {
  style <- match.arg(style)
  format_one <- switch(style,
    number = function(x) {
      formatC(x, format = "f", digits = digits)
    },
    percent = function(x) {
      paste0(formatC(100 * x, format = "f", digits = digits), "%")
    },
    comma = function(x) {
      formatC(x, format = "f", digits = 0L, big.mark = ",")
    },
    unit = function(x) {
      paste(formatC(x, format = "f", digits = digits), unit)
    },
    log10 = function(x) {
      paste0("10^", formatC(log10(x), format = "f", digits = 0L))
    }
  )
  labeller <- function(x) {
    out <- rep("", length(x))
    ok <- !is.na(x)
    out[ok] <- format_one(x[ok])
    out
  }
  return(labeller)
}

#' Axis settings for both directions
#'
#' @description
#' Builds a settings list per axis. The `x` and `y` elements each contain
#' `limits`, `breaks` and `labels`; with `equal = TRUE` both axes share the
#' wider of the two ranges, so one data unit has the same length on both.
#'
#' @param xlim,ylim Data limits.
#' @param xstyle,ystyle Label styles passed to [axis_labeller()].
#' @param n Approximate number of breaks per axis.
#' @param equal Whether both axes should use the same range width.
#' @return A list with elements `x` and `y`.
#' @export
axis_pair_spec <- function(xlim, ylim, xstyle = "number", ystyle = "number", n = 5L,
                           equal = FALSE) {
  xlim <- axis_expand(xlim)
  ylim <- axis_expand(ylim)
  if (equal) {
    half <- max(diff(xlim), diff(ylim)) / 2
    xlim <- mean(xlim) + c(-half, half)
    ylim <- mean(ylim) + c(-half, half)
  }

  # x axis
  xbreaks <- pretty(xlim, n = n)
  xbreaks <- xbreaks[xbreaks >= xlim[[1]] & xbreaks <= xlim[[2]]]
  x <- list(
    limits = xlim,
    breaks = xbreaks,
    labels = axis_labeller(xstyle)(xbreaks)
  )

  # y axis
  ybreaks <- pretty(ylim, n = n)
  ybreaks <- ybreaks[ybreaks >= ylim[[1]] & ybreaks <= ylim[[2]]]
  y <- list(
    limits = ylim,
    breaks = ybreaks,
    labels = axis_labeller(ystyle)(ybreaks)
  )

  return(list(x = x, y = y))
}

#' Data limits of two columns
#'
#' @param data A data frame.
#' @param x,y Names of the columns mapped to the x and y axes.
#' @param include Extra values each range must cover, as a list with
#'   optional elements `x` and `y`.
#' @return A list with elements `xlim` and `ylim`.
#' @export
data_limits <- function(data, x, y, include = list()) {
  # x column
  if (!x %in% names(data)) {
    stop(sprintf("Column `%s` not found in `data`.", x), call. = FALSE)
  }
  xv <- data[[x]]
  if (!is.numeric(xv)) {
    stop(sprintf("Column `%s` must be numeric.", x), call. = FALSE)
  }
  xlim <- range(c(xv, include$x), na.rm = TRUE, finite = TRUE)

  # y column
  if (!y %in% names(data)) {
    stop(sprintf("Column `%s` not found in `data`.", y), call. = FALSE)
  }
  yv <- data[[y]]
  if (!is.numeric(yv)) {
    stop(sprintf("Column `%s` must be numeric.", y), call. = FALSE)
  }
  ylim <- range(c(yv, include$y), na.rm = TRUE, finite = TRUE)

  return(list(xlim = xlim, ylim = ylim))
}

#' Continuous x scale from an axis spec
#'
#' @param spec One axis of [axis_pair_spec()].
#' @param position Axis position.
#' @param name Axis title.
#' @param trans Transformation name passed to ggplot2.
#' @param ... Passed to [ggplot2::scale_x_continuous()].
#' @return A ggplot2 scale.
#' @rdname lw_scale_axis
#' @export
lw_scale_x <- function(spec, position = c("bottom", "top"), name = ggplot2::waiver(),
                       trans = c("identity", "log10", "sqrt"), ...) {
  position <- match.arg(position)
  trans <- match.arg(trans)
  needed <- c("limits", "breaks", "labels")
  if (!all(needed %in% names(spec))) {
    stop("`spec` must have elements limits, breaks, labels.", call. = FALSE)
  }
  if (trans != "identity" && any(spec$limits <= 0)) {
    stop(sprintf("A %s x axis needs positive limits.", trans), call. = FALSE)
  }

  scale <- ggplot2::scale_x_continuous(
    name = name,
    limits = spec$limits,
    breaks = spec$breaks,
    labels = spec$labels,
    expand = c(0, 0),
    position = position,
    transform = trans,
    ...
  )
  return(scale)
}

#' Continuous y scale from an axis spec
#'
#' @param spec One axis of [axis_pair_spec()].
#' @param position Axis position.
#' @param name Axis title.
#' @param trans Transformation name passed to ggplot2.
#' @param ... Passed to [ggplot2::scale_y_continuous()].
#' @return A ggplot2 scale.
#' @rdname lw_scale_axis
#' @export
lw_scale_y <- function(spec, position = c("left", "right"), name = ggplot2::waiver(),
                       trans = c("identity", "log10", "sqrt"), ...) {
  position <- match.arg(position)
  trans <- match.arg(trans)
  needed <- c("limits", "breaks", "labels")
  if (!all(needed %in% names(spec))) {
    stop("`spec` must have elements limits, breaks, labels.", call. = FALSE)
  }
  if (trans != "identity" && any(spec$limits <= 0)) {
    stop(sprintf("A %s y axis needs positive limits.", trans), call. = FALSE)
  }

  scale <- ggplot2::scale_y_continuous(
    name = name,
    limits = spec$limits,
    breaks = spec$breaks,
    labels = spec$labels,
    expand = c(0, 0),
    position = position,
    transform = trans,
    ...
  )
  return(scale)
}

#' Both axis scales for a data frame
#'
#' @inheritParams data_limits
#' @param xstyle,ystyle Label styles passed to [axis_labeller()].
#' @param equal Whether both axes should use the same range width.
#' @return A list of two ggplot2 scales.
#' @export
lw_axes <- function(data, x, y, xstyle = "number", ystyle = "number", equal = FALSE) {
  lims <- data_limits(data, x, y)
  spec <- axis_pair_spec(lims$xlim, lims$ylim, xstyle = xstyle, ystyle = ystyle,
                         equal = equal)
  scales <- list(
    lw_scale_x(spec$x, name = x),
    lw_scale_y(spec$y, name = y)
  )
  return(scales)
}
