#' Validate a base font size
#'
#' Checks that a base font size is a single positive number within a
#' range that renders legibly on screen and in print.
#'
#' @param base_size Base font size in points.
#' @return `base_size`, invisibly, if valid; otherwise an error is raised.
#' @keywords internal
#' @noRd
check_base_size <- function(base_size) {
  if (!is.numeric(base_size) || length(base_size) != 1L || is.na(base_size)) {
    stop("`base_size` must be a single number.", call. = FALSE)
  }
  if (base_size < 6 || base_size > 40) {
    stop("`base_size` must lie between 6 and 40 points.", call. = FALSE)
  }
  invisible(base_size)
}

#' House theme for plotsmith figures
#'
#' A clean theme built on [ggplot2::theme_minimal()] with a muted grid,
#' left-aligned titles and a bottom legend.
#'
#' @param base_size Base font size in points.
#' @param base_family Base font family.
#' @param grid Which major grid lines to keep: `"y"`, `"x"`, `"both"` or `"none"`.
#' @return A [ggplot2::theme()] object.
#' @family themes
#' @export
theme_plotsmith <- function(base_size = 11, base_family = "",
                            grid = c("y", "x", "both", "none")) {
  check_base_size(base_size)
  grid <- match.arg(grid)
  grid_colour <- "grey88"
  thm <- ggplot2::theme_minimal(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", hjust = 0, size = base_size * 1.3),
      plot.subtitle = ggplot2::element_text(colour = "grey35", hjust = 0),
      plot.caption = ggplot2::element_text(colour = "grey50", size = base_size * 0.8),
      plot.title.position = "plot",
      legend.position = "bottom",
      legend.title = ggplot2::element_text(size = base_size * 0.9),
      panel.grid.minor = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank()
    )
  # Drop the grid lines the caller did not ask for
  thm + switch(grid,
    y = ggplot2::theme(
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(colour = grid_colour)
    ),
    x = ggplot2::theme(
      panel.grid.major.y = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_line(colour = grid_colour)
    ),
    both = ggplot2::theme(
      panel.grid.major = ggplot2::element_line(colour = grid_colour)
    ),
    none = ggplot2::theme(
      panel.grid.major = ggplot2::element_blank()
    )
  )
}

#' Dark variant of the house theme
#'
#' Same layout as [theme_plotsmith()] but with a dark panel and light text,
#' intended for slide decks with dark backgrounds.
#'
#' @inheritParams theme_plotsmith
#' @param background Fill colour of the plot and panel background.
#' @return A [ggplot2::theme()] object.
#' @family themes
#' @export
theme_plotsmith_dark <- function(base_size = 11, base_family = "",
                                 grid = c("y", "x", "both", "none"),
                                 background = "#1e2228") {
  grid <- match.arg(grid)
  theme_plotsmith(base_size = base_size, base_family = base_family, grid = grid) +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = background, colour = NA),
      panel.background = ggplot2::element_rect(fill = background, colour = NA),
      text = ggplot2::element_text(colour = "grey90"),
      axis.text = ggplot2::element_text(colour = "grey75"),
      plot.subtitle = ggplot2::element_text(colour = "grey70"),
      panel.grid.major = ggplot2::element_line(colour = "grey30")
    )
}

#' Remove the x axis decorations
#'
#' Blanks the x axis title, text and ticks. Useful for stacked panels that
#' share an axis with the panel below.
#'
#' @param keep_title Whether to keep the axis title.
#' @return A [ggplot2::theme()] object.
#' @family themes
#' @export
theme_no_x_axis <- function(keep_title = FALSE) {
  thm <- ggplot2::theme(
    axis.text.x = ggplot2::element_blank(),
    axis.ticks.x = ggplot2::element_blank()
  )
  if (!keep_title) {
    thm <- thm + ggplot2::theme(axis.title.x = ggplot2::element_blank())
  }
  thm
}

#' Remove the y axis decorations
#'
#' Blanks the y axis title, text and ticks. Useful for side-by-side panels
#' that share an axis with the panel to their left.
#'
#' @param keep_title Whether to keep the axis title.
#' @return A [ggplot2::theme()] object.
#' @family themes
#' @export
theme_no_y_axis <- function(keep_title = FALSE) {
  thm <- ggplot2::theme(
    axis.text.y = ggplot2::element_blank(),
    axis.ticks.y = ggplot2::element_blank()
  )
  if (!keep_title) {
    thm <- thm + ggplot2::theme(axis.title.y = ggplot2::element_blank())
  }
  thm
}

#' Place the legend
#'
#' Moves the legend to one of the standard positions, or inside the panel
#' at the given relative coordinates.
#'
#' @param position Where the legend goes.
#' @param inside Numeric vector of length two with relative x and y
#'   coordinates, used only when `position = "inside"`.
#' @return A [ggplot2::theme()] object.
#' @family themes
#' @export
theme_legend_position <- function(position = c("bottom", "top", "right", "none", "inside"),
                                  inside = c(0.95, 0.95)) {
  position <- match.arg(position)
  switch(position,
    bottom = ggplot2::theme(legend.position = "bottom", legend.direction = "horizontal"),
    top = ggplot2::theme(legend.position = "top", legend.direction = "horizontal"),
    right = ggplot2::theme(legend.position = "right", legend.direction = "vertical"),
    none = ggplot2::theme(legend.position = "none"),
    inside = ggplot2::theme(
      legend.position = "inside",
      legend.position.inside = inside,
      legend.justification = inside,
      legend.background = ggplot2::element_rect(fill = "white", colour = "grey80")
    )
  )
}

#' Set the house theme as the session default
#'
#' Calls [ggplot2::theme_set()] with [theme_plotsmith()] and returns the
#' previous theme so it can be restored later.
#'
#' @param ... Passed to [theme_plotsmith()].
#' @return The previously active theme, invisibly.
#' @family themes
#' @export
use_plotsmith_theme <- function(...) {
  old <- ggplot2::theme_set(theme_plotsmith(...))
  invisible(old)
}
