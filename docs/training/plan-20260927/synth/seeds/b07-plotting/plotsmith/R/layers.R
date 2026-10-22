#' Check that columns exist in a data frame
#'
#' Stops with an informative message listing every requested column that
#' is missing from `data`.
#'
#' @param data A data frame of observations to plot.
#' @param cols Character vector of column names, `NULL` entries are ignored.
#' @return `data`, invisibly.
#' @noRd
check_plot_columns <- function(data, cols) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  cols <- unlist(cols, use.names = FALSE)
  missing <- setdiff(cols, names(data))
  if (length(missing) > 0) {
    stop(
      "Column(s) not found in `data`: ",
      paste0("`", missing, "`", collapse = ", "),
      call. = FALSE
    )
  }
  invisible(data)
}

#' Scatter plot builder
#'
#' Builds a scatter plot from column names given as strings, with an
#' optional colour grouping.
#'
#' @param data A data frame of observations to plot.
#' @param x Name of the column mapped to the x axis.
#' @param y Name of the column mapped to the y axis.
#' @param colour Optional name of the column mapped to colour.
#' @param size Point size.
#' @param alpha Point transparency between 0 and 1.
#' @return A [ggplot2::ggplot()] object.
#' @family builders
#' @importFrom ggplot2 ggplot aes geom_point
#' @export
ps_scatter <- function(data, x, y, colour = NULL, size = 2, alpha = 0.8) {
  check_plot_columns(data, list(x, y, colour))
  if (is.null(colour)) {
    mapping <- aes(x = .data[[x]], y = .data[[y]])
  } else {
    mapping <- aes(x = .data[[x]], y = .data[[y]], colour = .data[[colour]])
  }
  ggplot(data, mapping) +
    geom_point(size = size, alpha = alpha) +
    theme_plotsmith()
}

#' Line plot builder
#'
#' Builds a line plot from column names given as strings, drawing one line
#' per level of an optional grouping column.
#'
#' @param data A data frame of observations to plot.
#' @param x Name of the column mapped to the x axis.
#' @param y Name of the column mapped to the y axis.
#' @param group Optional name of the column defining separate lines.
#' @param linewidth Line width.
#' @return A [ggplot2::ggplot()] object.
#' @family builders
#' @importFrom ggplot2 ggplot aes geom_line
#' @export
ps_line <- function(data, x, y, group = NULL, linewidth = 0.8) {
  check_plot_columns(data, list(x, y, group))
  if (is.null(group)) {
    mapping <- aes(x = .data[[x]], y = .data[[y]])
  } else {
    mapping <- aes(
      x = .data[[x]], y = .data[[y]],
      group = .data[[group]], colour = .data[[group]]
    )
  }
  ggplot(data, mapping) +
    geom_line(linewidth = linewidth) +
    theme_plotsmith()
}

#' Bar chart builder
#'
#' Builds a bar chart of pre-computed values, optionally sorting bars by
#' height and flipping them horizontally for long category labels.
#'
#' @param data A data frame of observations to plot.
#' @param x Name of the categorical column.
#' @param y Name of the column holding bar heights.
#' @param fill Optional name of the column mapped to fill.
#' @param sort Whether to order bars by decreasing height.
#' @param horizontal Whether to draw horizontal bars.
#' @return A [ggplot2::ggplot()] object.
#' @family builders
#' @importFrom ggplot2 ggplot aes geom_col coord_flip
#' @export
ps_bar <- function(data, x, y, fill = NULL, sort = TRUE, horizontal = FALSE) {
  check_plot_columns(data, list(x, y, fill))
  if (sort) {
    lvls <- data[[x]][order(data[[y]], decreasing = !horizontal)]
    data[[x]] <- factor(data[[x]], levels = unique(lvls))
  }
  mapping <- aes(x = .data[[x]], y = .data[[y]])
  p <- ggplot(data, mapping)
  if (is.null(fill)) {
    p <- p + geom_col(fill = "#3b6e8f", width = 0.7)
  } else {
    p <- p + geom_col(aes(fill = .data[[fill]]), width = 0.7)
  }
  if (horizontal) {
    p <- p + coord_flip() + theme_plotsmith(grid = "x")
  } else {
    p <- p + theme_plotsmith(grid = "y")
  }
  p
}

#' Add a fitted trend line
#'
#' Adds a smoothed trend to an existing plot, using the plot's own x and y
#' mappings.
#'
#' @param plot A [ggplot2::ggplot()] object.
#' @param method Smoothing method.
#' @param se Whether to draw the confidence band.
#' @param colour Line colour.
#' @return The modified plot.
#' @family layers
#' @importFrom ggplot2 geom_smooth
#' @export
add_trend_line <- function(plot, method = c("lm", "loess", "none"), se = TRUE,
                           colour = "grey20") {
  method <- match.arg(method)
  if (method == "none") {
    return(plot)
  }
  plot + geom_smooth(
    method = method, formula = y ~ x, se = se,
    colour = colour, linewidth = 0.7, fill = "grey80"
  )
}

#' Add a horizontal or vertical reference line
#'
#' Draws a dashed reference line at one or more positions, with an optional
#' text label at the end of each line.
#'
#' @param plot A [ggplot2::ggplot()] object.
#' @param value Numeric positions of the reference lines.
#' @param axis `"h"` for horizontal lines or `"v"` for vertical lines.
#' @param label Optional character labels, recycled to the length of `value`.
#' @return The modified plot.
#' @family layers
#' @importFrom ggplot2 geom_hline geom_vline annotate
#' @export
add_reference_line <- function(plot, value, axis = c("h", "v"), label = NULL) {
  axis <- match.arg(axis)
  if (axis == "h") {
    plot <- plot + geom_hline(yintercept = value, linetype = "dashed", colour = "grey40")
  } else {
    plot <- plot + geom_vline(xintercept = value, linetype = "dashed", colour = "grey40")
  }
  if (is.null(label)) {
    return(plot)
  }
  label <- rep_len(label, length(value))
  if (axis == "h") {
    plot + annotate("text", x = Inf, y = value, label = label,
                    hjust = 1.05, vjust = -0.4, size = 3, colour = "grey30")
  } else {
    plot + annotate("text", x = value, y = Inf, label = label,
                    hjust = -0.1, vjust = 1.5, size = 3, colour = "grey30")
  }
}

#' Add group means as larger points
#'
#' Overlays a summary point at the mean of y for each x value, which is
#' handy on top of jittered raw data.
#'
#' @param plot A [ggplot2::ggplot()] object.
#' @param size Size of the summary points.
#' @param shape Point shape of the summary points.
#' @param colour Colour of the summary points.
#' @return The modified plot.
#' @family layers
#' @importFrom ggplot2 stat_summary
#' @export
add_mean_points <- function(plot, size = 3.5, shape = 18, colour = "black") {
  plot + stat_summary(
    fun = mean, geom = "point",
    size = size, shape = shape, colour = colour
  )
}

#' Assemble a plot from a list of layers
#'
#' Starts from a base plot and adds each element of `layers` in order,
#' skipping `NULL` entries so callers can build the list conditionally.
#'
#' @param plot A [ggplot2::ggplot()] object.
#' @param layers A list of ggplot2 components (geoms, scales, themes, labels).
#' @return The modified plot.
#' @family layers
#' @export
build_layered_plot <- function(plot, layers) {
  if (!inherits(plot, "ggplot")) {
    stop("`plot` must be a ggplot object.", call. = FALSE)
  }
  if (!is.list(layers)) {
    layers <- list(layers)
  }
  for (i in seq_along(layers)) {
    layer <- layers[[i]]
    if (is.null(layer)) {
      next
    }
    plot <- plot + layer
  }
  plot
}
