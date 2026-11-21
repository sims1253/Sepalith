#' Compute padded axis limits
#'
#' @description
#' Computes `xlim` and `ylim` from the finite values of `x` and `y`, widens
#' each range by `pad` times its width and optionally stretches it to include
#' zero.
#'
#' @param x,y Numeric vectors of coordinates.
#' @param pad Fraction of the data range added on each side.
#' @param include_zero Which axes must include zero.
#' @return A list with elements `xlim` and `ylim`.
#' @export
axis_limits <- function(x, y, pad = 0.04,
                        include_zero = c("none", "x", "y", "both")) {
  include_zero <- match.arg(include_zero)
  if (!is.numeric(pad) || length(pad) != 1L || pad < 0) {
    stop("`pad` must be a single non-negative number.", call. = FALSE)
  }

  # x axis
  xlim <- range(x[is.finite(x)])
  if (include_zero %in% c("x", "both")) {
    xlim <- range(xlim, 0)
  }
  xwidth <- diff(xlim)
  if (xwidth == 0) {
    xwidth <- max(abs(xlim), 1)
  }
  xlim <- xlim + c(-1, 1) * pad * xwidth

  # y axis
  ylim <- range(y[is.finite(y)])
  if (include_zero %in% c("y", "both")) {
    ylim <- range(ylim, 0)
  }
  ywidth <- diff(ylim)
  if (ywidth == 0) {
    ywidth <- max(abs(ylim), 1)
  }
  ylim <- ylim + c(-1, 1) * pad * ywidth

  list(xlim = xlim, ylim = ylim)
}

#' Format axis tick labels
#'
#' @param breaks Numeric tick positions.
#' @param style Label style.
#' @param digits Significant digits for the `"plain"` and `"scientific"`
#'   styles.
#' @return A character vector the same length as `breaks`.
#' @keywords internal
axis_label_format <- function(breaks,
                              style = c("plain", "comma", "percent", "scientific", "si"),
                              digits = 3L) {
  style <- match.arg(style)
  switch(style,
    plain = {
      format(signif(breaks, digits), trim = TRUE, scientific = FALSE)
    },
    comma = {
      formatC(breaks, format = "f", digits = 0L, big.mark = ",")
    },
    percent = {
      paste0(formatC(100 * breaks, format = "f", digits = 0L), "%")
    },
    scientific = {
      formatC(breaks, format = "e", digits = digits - 1L)
    },
    si = {
      units <- c(G = 1e9, M = 1e6, k = 1e3)
      out <- as.character(signif(breaks, digits))
      done <- rep(FALSE, length(breaks))
      for (u in names(units)) {
        hit <- !done & abs(breaks) >= units[[u]]
        out[hit] <- paste0(as.character(signif(breaks[hit] / units[[u]], digits)), u)
        done <- done | hit
      }
      out
    }
  )
}

#' Draw the horizontal axis
#'
#' @param lim Axis limits as a length-two numeric vector.
#' @param n Approximate number of ticks.
#' @param style Label style passed to [axis_label_format()].
#' @param side Side of the plot, `1` (bottom) or `3` (top).
#' @param col Colour of the axis line, ticks and title.
#' @param minor Whether to add unlabelled ticks halfway between the labelled
#'   ones.
#' @param title Optional axis title drawn in the margin.
#' @return Invisibly, a list with elements `breaks`, `labels` and `minor`.
#' @rdname draw_axis
#' @export
draw_axis_x <- function(lim, n = 5L, style = "plain", side = 1L, col = "grey30",
                        minor = TRUE, title = NULL) {
  if (!side %in% c(1L, 3L)) {
    stop("`side` must be 1 or 3 for the x axis.", call. = FALSE)
  }
  if (!is.numeric(lim) || length(lim) != 2L || anyNA(lim)) {
    stop("`lim` must be two non-missing numbers.", call. = FALSE)
  }
  breaks <- pretty(lim, n = n)
  breaks <- breaks[breaks >= min(lim) & breaks <= max(lim)]
  labels <- axis_label_format(breaks, style = style)
  axis(side, at = breaks, labels = labels, col = col, col.axis = col, las = 1)

  minor_at <- numeric()
  if (minor && length(breaks) > 1L) {
    step <- (breaks[[2]] - breaks[[1]]) / 2
    minor_at <- c(breaks[[1]] - step, breaks + step)
    minor_at <- minor_at[minor_at >= min(lim) & minor_at <= max(lim)]
    axis(side, at = minor_at, labels = FALSE, tcl = -0.25, col = col)
  }
  if (!is.null(title)) {
    mtext(title, side = side, line = 2.5, col = col)
  }
  invisible(list(breaks = breaks, labels = labels, minor = minor_at))
}

#' Draw the vertical axis
#'
#' @param lim Axis limits as a length-two numeric vector.
#' @param n Approximate number of ticks.
#' @param style Label style passed to [axis_label_format()].
#' @param side Side of the plot, `2` (left) or `4` (right).
#' @param col Colour of the axis line, ticks and title.
#' @param minor Whether to add unlabelled ticks halfway between the labelled
#'   ones.
#' @param title Optional axis title drawn in the margin.
#' @return Invisibly, a list with elements `breaks`, `labels` and `minor`.
#' @rdname draw_axis
#' @export
draw_axis_y <- function(lim, n = 5L, style = "plain", side = 2L, col = "grey30",
                        minor = TRUE, title = NULL) {
  if (!side %in% c(2L, 4L)) {
    stop("`side` must be 2 or 4 for the y axis.", call. = FALSE)
  }
  if (!is.numeric(lim) || length(lim) != 2L || anyNA(lim)) {
    stop("`lim` must be two non-missing numbers.", call. = FALSE)
  }
  breaks <- pretty(lim, n = n)
  breaks <- breaks[breaks >= min(lim) & breaks <= max(lim)]
  labels <- axis_label_format(breaks, style = style)
  axis(side, at = breaks, labels = labels, col = col, col.axis = col, las = 1)

  minor_at <- numeric()
  if (minor && length(breaks) > 1L) {
    step <- (breaks[[2]] - breaks[[1]]) / 2
    minor_at <- c(breaks[[1]] - step, breaks + step)
    minor_at <- minor_at[minor_at >= min(lim) & minor_at <= max(lim)]
    axis(side, at = minor_at, labels = FALSE, tcl = -0.25, col = col)
  }
  if (!is.null(title)) {
    mtext(title, side = side, line = 2.5, col = col)
  }
  invisible(list(breaks = breaks, labels = labels, minor = minor_at))
}

#' Open a plot region with both axes
#'
#' @description
#' Starts a new plot, sets the user coordinates from `x` and `y` and draws
#' both axes with their titles and a box. Steps:
#' 1. log-transform the axes named in `log` and mark their titles;
#' 2. compute limits with [axis_limits()];
#' 3. reverse the axes named in `flip`;
#' 4. draw with [draw_axis_x()] and [draw_axis_y()].
#'
#' @param x,y Numeric vectors of coordinates.
#' @param xlab,ylab Axis titles.
#' @param xstyle,ystyle Label styles passed to [axis_label_format()].
#' @param flip Which axes run from high to low values.
#' @param log Which axes use a log10 scale.
#' @return Invisibly, a list with elements `xlim`, `ylim`, `xbreaks` and
#'   `ybreaks`.
#' @export
setup_axes <- function(x, y, xlab = "x", ylab = "y", xstyle = "plain",
                       ystyle = "plain", flip = c("none", "x", "y", "both"),
                       log = c("none", "x", "y", "both")) {
  flip <- match.arg(flip)
  log <- match.arg(log)
  if (log %in% c("x", "both")) {
    if (any(x <= 0, na.rm = TRUE)) {
      stop("`x` must be positive for a log axis.", call. = FALSE)
    }
    x <- log10(x)
    xlab <- sprintf("%s (log10)", xlab)
  }
  if (log %in% c("y", "both")) {
    if (any(y <= 0, na.rm = TRUE)) {
      stop("`y` must be positive for a log axis.", call. = FALSE)
    }
    y <- log10(y)
    ylab <- sprintf("%s (log10)", ylab)
  }

  lims <- axis_limits(x, y)
  xlim <- lims$xlim
  ylim <- lims$ylim
  if (flip %in% c("x", "both")) {
    xlim <- rev(xlim)
  }
  if (flip %in% c("y", "both")) {
    ylim <- rev(ylim)
  }

  plot.new()
  plot.window(xlim = xlim, ylim = ylim)
  xaxis <- draw_axis_x(xlim, style = xstyle, title = xlab)
  yaxis <- draw_axis_y(ylim, style = ystyle, title = ylab)
  box()
  invisible(list(xlim = xlim, ylim = ylim, xbreaks = xaxis$breaks, ybreaks = yaxis$breaks))
}

#' Add reference grid lines
#'
#' @param xbreaks,ybreaks Positions of the vertical and horizontal lines, or
#'   `NULL` to skip that direction.
#' @param col Line colour.
#' @param lty Line type of the major lines.
#' @param minor Whether to add dotted lines halfway between major lines.
#' @return Invisibly, a list with elements `n_vertical` and `n_horizontal`.
#' @export
grid_lines <- function(xbreaks = NULL, ybreaks = NULL, col = "grey90", lty = 1L,
                       minor = FALSE) {
  n_vertical <- 0L
  n_horizontal <- 0L
  if (!is.null(xbreaks)) {
    xbreaks <- sort(xbreaks[is.finite(xbreaks)])
    abline(v = xbreaks, col = col, lty = lty)
    n_vertical <- length(xbreaks)
    if (minor && length(xbreaks) > 1L) {
      xmid <- (xbreaks[-1L] + xbreaks[-length(xbreaks)]) / 2
      abline(v = xmid, col = col, lty = 3L)
      n_vertical <- n_vertical + length(xmid)
    }
  }
  if (!is.null(ybreaks)) {
    ybreaks <- sort(ybreaks[is.finite(ybreaks)])
    abline(h = ybreaks, col = col, lty = lty)
    n_horizontal <- length(ybreaks)
    if (minor && length(ybreaks) > 1L) {
      ymid <- (ybreaks[-1L] + ybreaks[-length(ybreaks)]) / 2
      abline(h = ymid, col = col, lty = 3L)
      n_horizontal <- n_horizontal + length(ymid)
    }
  }
  invisible(list(n_vertical = n_vertical, n_horizontal = n_horizontal))
}

#' Draw a secondary axis on the opposite side
#'
#' @param which Axis to mirror.
#' @param trans Function mapping primary to secondary values.
#' @param inverse Inverse of `trans`, used to place the secondary ticks.
#' @param n Approximate number of ticks.
#' @param style Label style passed to [axis_label_format()].
#' @param title Optional title for the secondary axis.
#' @return Invisibly, the secondary tick values.
#' @export
draw_secondary_axis <- function(which = c("x", "y"), trans = identity, inverse = identity,
                                n = 5L, style = "plain", title = NULL) {
  which <- match.arg(which)
  if (!is.function(trans) || !is.function(inverse)) {
    stop("`trans` and `inverse` must be functions.", call. = FALSE)
  }
  usr <- par("usr")
  if (which == "x") {
    lim <- trans(usr[1:2])
    values <- pretty(lim, n = n)
    values <- values[values >= min(lim) & values <= max(lim)]
    labels <- axis_label_format(values, style = style)
    axis(3L, at = inverse(values), labels = labels, las = 1)
    if (!is.null(title)) {
      mtext(title, side = 3L, line = 2.5)
    }
  } else {
    lim <- trans(usr[3:4])
    values <- pretty(lim, n = n)
    values <- values[values >= min(lim) & values <= max(lim)]
    labels <- axis_label_format(values, style = style)
    axis(4L, at = inverse(values), labels = labels, las = 1)
    if (!is.null(title)) {
      mtext(title, side = 4L, line = 2.5)
    }
  }
  invisible(values)
}
