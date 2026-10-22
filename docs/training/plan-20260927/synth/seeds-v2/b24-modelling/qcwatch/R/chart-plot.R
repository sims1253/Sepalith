#' Vertical range covering a chart
#'
#' Finds y-axis limits wide enough for the plotted statistic and both control
#' limits, with a little padding.
#'
#' @param values Numeric vector of plotted statistics.
#' @param lcl,ucl Lower and upper control limits.
#' @param pad Fraction of the range added above and below.
#' @return A numeric vector of length two.
#' @noRd
chart.ylim = function(values, lcl, ucl, pad = 0.05) {
  rng = range(c(values, lcl, ucl), na.rm = TRUE, finite = TRUE)
  width = diff(rng)
  # A flat chart still needs some vertical room.
  if (width == 0) {
    width = max(abs(rng[1L]), 1)
  }
  c(rng[1L] - pad * width, rng[2L] + pad * width)
}

#' Draw center line and control limits
#'
#' Adds the center line and step-shaped control limits to an open plot.
#'
#' @param index Integer vector of time points.
#' @param center Center line.
#' @param lcl,ucl Lower and upper control limits.
#' @param col Colour of the limit lines.
#' @noRd
draw.limits = function(index, center, lcl, ucl, col = "firebrick") {
  graphics::abline(h = center, col = "grey40", lty = 1)

  # Constant limits are drawn across the whole plot, varying ones as steps.
  if (length(unique(ucl)) == 1L) {
    graphics::abline(h = ucl[1L], col = col, lty = 2)
  } else {
    graphics::lines(index, ucl, type = "s", col = col, lty = 2)
  }

  if (length(unique(lcl)) == 1L) {
    graphics::abline(h = lcl[1L], col = col, lty = 2)
  } else {
    graphics::lines(index, lcl, type = "s", col = col, lty = 2)
  }
  invisible(NULL)
}

#' Highlight signalling points
#'
#' Overplots the points that triggered an alarm in a contrasting colour.
#'
#' @param index Integer vector of time points.
#' @param values Numeric vector of plotted statistics.
#' @param signals Integer vector of signalling positions.
#' @param col Colour of the highlighted points.
#' @noRd
mark.signals = function(index, values, signals, col = "firebrick") {
  if (length(signals) == 0L) {
    return(invisible(NULL))
  }
  graphics::points(index[signals], values[signals], pch = 19, col = col, cex = 1.2)
  invisible(NULL)
}

#' Plot a Shewhart chart
#'
#' Draws the statistic over time with its center line and control limits and
#' highlights out-of-control points.
#'
#' @param x An object of class `qc_shewhart`.
#' @param main Plot title; defaults to the chart type.
#' @param ... Passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @export
plot.qc_shewhart = function(x, main = NULL, ...) {
  index = seq_along(x$stat)
  if (is.null(main)) {
    main = sprintf("%s chart", toupper(x$type))
  }

  ylim = chart.ylim(x$stat, x$lcl, x$ucl)
  graphics::plot(
    index, x$stat,
    type = "b", pch = 20, ylim = ylim,
    xlab = "Subgroup", ylab = "Statistic", main = main, ...
  )
  draw.limits(index, x$center, x$lcl, x$ucl)
  mark.signals(index, x$stat, x$signals)
  invisible(x)
}

#' Plot a CUSUM chart
#'
#' Draws the upper sum above zero and the lower sum mirrored below zero, with
#' the decision interval on both sides.
#'
#' @param x An object of class `qc_cusum`.
#' @param main Plot title; defaults to the chart type.
#' @param ... Passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @export
plot.qc_cusum = function(x, main = NULL, ...) {
  index = seq_along(x$upper)
  if (is.null(main)) {
    main = "Tabular CUSUM"
  }

  # Mirror the lower sum below zero so both sides share one panel.
  lower = -x$lower
  ylim = chart.ylim(c(x$upper, lower), -x$h, x$h)
  graphics::plot(
    index, x$upper,
    type = "b", pch = 20, ylim = ylim,
    xlab = "Observation", ylab = "Cumulative sum", main = main, ...
  )
  graphics::lines(index, lower, type = "b", pch = 20, col = "grey30")
  draw.limits(index, 0, -x$h, x$h)
  mark.signals(index, x$upper, x$signals.upper)
  mark.signals(index, lower, x$signals.lower)
  invisible(x)
}

#' Plot an EWMA chart
#'
#' Draws the raw observations faintly and the EWMA statistic on top, with
#' time-varying control limits.
#'
#' @param x An object of class `qc_ewma`.
#' @param main Plot title; defaults to the chart type.
#' @param ... Passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @export
plot.qc_ewma = function(x, main = NULL, ...) {
  index = seq_along(x$stat)
  if (is.null(main)) {
    main = sprintf("EWMA chart (lambda = %g)", x$lambda)
  }

  ylim = chart.ylim(x$stat, x$lcl, x$ucl)
  graphics::plot(
    index, x$stat,
    type = "b", pch = 20, ylim = ylim,
    xlab = "Observation", ylab = "EWMA", main = main, ...
  )
  # Raw data in the background for context.
  graphics::points(index, x$x, pch = 1, col = "grey70", cex = 0.6)
  draw.limits(index, x$target, x$lcl, x$ucl)
  mark.signals(index, x$stat, x$signals)
  invisible(x)
}
