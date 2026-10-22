#' Evaluate plotting code with temporary graphical parameters
#'
#' Sets the given [graphics::par()] values, evaluates `code` and restores the
#' previous values afterwards, even if `code` fails.
#'
#' @param ... Graphical parameters passed to [graphics::par()].
#' @param code Plotting code to evaluate.
#' @return The value of `code`, invisibly.
#' @keywords internal
with_par = function(..., code) {
  old = graphics::par(...)
  on.exit(graphics::par(old), add = TRUE)
  invisible(force(code))
}

#' Plot several series as lines
#'
#' Draws one line per group with a shared y range and a legend, using base
#' graphics.
#'
#' @param x Numeric or date vector for the x axis.
#' @param y Numeric vector of values.
#' @param group Vector of group labels the same length as `y`.
#' @param cols Colours, one per group. Defaults to the harbour palette.
#' @param legend_pos Position keyword passed to [graphics::legend()].
#' @param ... Passed to [graphics::plot()].
#' @return A named character vector of the colours used, invisibly.
#' @export
plot_series = function(x, y, group, cols = NULL, legend_pos = "topleft", ...) {
  stopifnot(length(x) == length(y), length(y) == length(group))
  groups = unique(group[!is.na(group)])
  if (is.null(cols)) {
    cols = paletteGet("harbour", n = length(groups))
  }
  names(cols) = groups

  old = graphics::par(mar = c(4, 4, 2, 1), las = 1)
  on.exit(graphics::par(old), add = TRUE)

  # Empty frame first so every series shares the same axis ranges
  graphics::plot(x, y, type = "n", xlab = "", ylab = "", bty = "l", ...)
  graphics::grid(nx = NA, ny = NULL, col = "grey90", lty = 1)

  for (g in groups) {
    idx = which(group == g)
    ord = order(x[idx])
    graphics::lines(x[idx][ord], y[idx][ord], col = cols[[g]], lwd = 2)
  }

  graphics::legend(legend_pos, legend = groups, col = cols, lwd = 2, bty = "n",
                   cex = 0.85)
  invisible(cols)
}

#' Histogram with a density curve
#'
#' Draws a probability-scaled histogram of `x` and overlays a kernel
#' density estimate and a rug of the raw values.
#'
#' @param x Numeric vector of values.
#' @param breaks Breaks specification passed to [graphics::hist()].
#' @param col Fill colour of the bars.
#' @param main Plot title.
#' @return The [graphics::hist()] object, invisibly.
#' @export
plot_hist_density = function(x, breaks = "FD", col = "#7fb7be", main = NULL) {
  x = x[is.finite(x)]
  if (length(x) < 2) {
    stop("`x` needs at least two finite values.", call. = FALSE)
  }

  dens = stats::density(x)
  h = graphics::hist(x, breaks = breaks, plot = FALSE)

  old = graphics::par(mar = c(4, 4, 3, 1), las = 1)
  on.exit(graphics::par(old), add = TRUE)
  # Leave headroom so the density peak is not clipped by the bars' ylim
  graphics::plot(h, freq = FALSE, col = col, border = "white", main = main,
                 xlab = "", ylim = c(0, max(h$density, dens$y) * 1.05))
  graphics::lines(dens, lwd = 2, col = darkenColour(col, 0.5))
  graphics::rug(x, col = "grey40")
  invisible(h)
}

#' Paired bar chart for two measurements per category
#'
#' Draws side-by-side bars for two numeric vectors, for example before and
#' after values, and labels the bars with their values.
#'
#' @param first,second Numeric vectors of the same length.
#' @param labels Category labels, one per element.
#' @param legend_labels Labels for the two series.
#' @param cols Two fill colours.
#' @return The bar midpoints returned by [graphics::barplot()], invisibly.
#' @export
plot_paired_bars = function(first, second, labels,
                            legend_labels = c("Before", "After"),
                            cols = c("#a7b7c7", "#1f4e79")) {
  stopifnot(length(first) == length(second), length(first) == length(labels))
  heights = rbind(first, second)
  colnames(heights) = labels

  # xpd = NA lets the legend sit above the plot region
  old = graphics::par(mar = c(5, 4, 2, 1), las = 1, xpd = NA)
  on.exit(graphics::par(old), add = TRUE)
  mids = graphics::barplot(heights, beside = TRUE, col = cols, border = NA,
                           ylim = c(0, max(heights, na.rm = TRUE) * 1.15))

  graphics::text(mids, heights, labels = format(round(heights, 1)), pos = 3, cex = 0.75)
  graphics::legend("top", legend = legend_labels, fill = cols, border = NA,
                   horiz = TRUE, bty = "n", inset = c(0, -0.08))
  invisible(mids)
}

#' Draw a legend outside the plot region
#'
#' Places a legend to the right of the plotting region, temporarily
#' allowing drawing in the margin.
#'
#' @param legend Character vector of legend labels.
#' @param col Colours matching `legend`.
#' @param ... Passed to [graphics::legend()].
#' @keywords internal
legend_outside = function(legend, col, ...) {
  old = graphics::par(xpd = NA)
  on.exit(graphics::par(old), add = TRUE)

  # Anchor the legend's top-left corner at the plot's top-right corner
  usr = graphics::par("usr")
  res = graphics::legend(x = usr[2], y = usr[4], legend = legend, col = col,
                         xjust = 0, yjust = 1, bty = "n", ...)
  invisible(res)
}

#' Draw a list of panels in a grid
#'
#' Splits the device into a grid and calls `fun` on each element of `data`,
#' passing the element name as the panel title.
#'
#' @param data A named list of datasets, one per panel.
#' @param fun A function taking a dataset and a `main` argument.
#' @param ncol Number of columns in the grid.
#' @return `data`, invisibly.
#' @export
plot_panels = function(data, fun, ncol = 2) {
  n = length(data)
  if (n == 0) {
    stop("`data` must contain at least one panel.", call. = FALSE)
  }

  ncol = min(ncol, n)
  nrow = ceiling(n / ncol)
  old = graphics::par(mfrow = c(nrow, ncol), mar = c(3, 3, 2, 1), mgp = c(2, 0.6, 0))
  on.exit(graphics::par(old), add = TRUE)

  panel_names = names(data)
  if (is.null(panel_names)) {
    panel_names = paste("Panel", seq_len(n))
  }

  for (i in seq_len(n)) {
    fun(data[[i]], main = panel_names[i])
  }
  invisible(data)
}
