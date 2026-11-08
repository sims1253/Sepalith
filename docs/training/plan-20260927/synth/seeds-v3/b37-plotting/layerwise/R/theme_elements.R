#' Text element spec
#'
#' @param size Font size in points.
#' @param colour Text colour.
#' @param face Font face.
#' @param angle Rotation in degrees.
#' @param hjust Horizontal justification between 0 and 1.
#' @return A `lw_element_text` list with elements `size`, `colour`, `face`,
#'   `angle` and `hjust`.
#' @rdname elements
#' @export
el_text <- function(size = 10, colour = "grey20", face = c("plain", "bold", "italic"),
                    angle = 0, hjust = 0.5) {
  face <- match.arg(face)
  if (!is.numeric(size) || length(size) != 1L || size <= 0) {
    stop("`size` must be a single positive number.", call. = FALSE)
  }
  if (length(colour) != 1L || is.na(colour) ||
      inherits(try(grDevices::col2rgb(colour), silent = TRUE), "try-error")) {
    stop("`colour` must be a single valid colour.", call. = FALSE)
  }
  if (!is.numeric(angle) || length(angle) != 1L || abs(angle) > 360) {
    stop("`angle` must be a single number between -360 and 360.", call. = FALSE)
  }
  if (!is.numeric(hjust) || length(hjust) != 1L || hjust < 0 || hjust > 1) {
    stop("`hjust` must be a single number between 0 and 1.", call. = FALSE)
  }

  out <- structure(
    list(size = size, colour = colour, face = face, angle = angle, hjust = hjust),
    class = c("lw_element_text", "lw_element")
  )
  return(out)
}

#' Line element spec
#'
#' @param colour Line colour.
#' @param linewidth Line width in mm.
#' @param linetype Line type.
#' @param lineend Line end style.
#' @return A `lw_element_line` list with elements `colour`, `linewidth`,
#'   `linetype` and `lineend`.
#' @rdname elements
#' @export
el_line <- function(colour = "grey80", linewidth = 0.4,
                    linetype = c("solid", "dashed", "dotted"),
                    lineend = c("butt", "round", "square")) {
  linetype <- match.arg(linetype)
  lineend <- match.arg(lineend)
  if (length(colour) != 1L || is.na(colour) ||
      inherits(try(grDevices::col2rgb(colour), silent = TRUE), "try-error")) {
    stop("`colour` must be a single valid colour.", call. = FALSE)
  }
  if (!is.numeric(linewidth) || length(linewidth) != 1L || linewidth <= 0) {
    stop("`linewidth` must be a single positive number.", call. = FALSE)
  }

  out <- structure(
    list(colour = colour, linewidth = linewidth, linetype = linetype, lineend = lineend),
    class = c("lw_element_line", "lw_element")
  )
  return(out)
}

#' Rectangle element spec
#'
#' @param fill Fill colour.
#' @param colour Border colour, or `NA` for no border.
#' @param linewidth Border width in mm.
#' @return A `lw_element_rect` list with elements `fill`, `colour` and
#'   `linewidth`.
#' @rdname elements
#' @export
el_rect <- function(fill = "white", colour = NA, linewidth = 0.4) {
  if (length(fill) != 1L || is.na(fill) ||
      inherits(try(grDevices::col2rgb(fill), silent = TRUE), "try-error")) {
    stop("`fill` must be a single valid colour.", call. = FALSE)
  }
  if (length(colour) != 1L || (!is.na(colour) &&
      inherits(try(grDevices::col2rgb(colour), silent = TRUE), "try-error"))) {
    stop("`colour` must be a single valid colour or NA.", call. = FALSE)
  }
  if (!is.numeric(linewidth) || length(linewidth) != 1L || linewidth <= 0) {
    stop("`linewidth` must be a single positive number.", call. = FALSE)
  }

  out <- structure(
    list(fill = fill, colour = colour, linewidth = linewidth),
    class = c("lw_element_rect", "lw_element")
  )
  return(out)
}

#' Store an element in a theme spec
#'
#' @description
#' Theme specs are plain named lists of `lw_element` objects. Each element
#' name accepts one kind: `axis.text`, `axis.title`, `plot.title` and
#' `legend.text` take text; `axis.line` and `panel.grid` take lines;
#' `panel.background` takes a rectangle. `NULL` removes the element.
#'
#' @param spec A theme spec list.
#' @param element Element name.
#' @param value An `lw_element` of the matching kind, or `NULL`.
#' @return The updated spec.
#' @keywords internal
set_element <- function(spec, element, value) {
  kinds <- c(
    axis.text = "text", axis.title = "text", plot.title = "text", legend.text = "text",
    axis.line = "line", panel.grid = "line", panel.background = "rect"
  )
  if (!element %in% names(kinds)) {
    stop(sprintf("Unknown theme element `%s`.", element), call. = FALSE)
  }
  expected <- paste0("lw_element_", kinds[[element]])
  if (!is.null(value) && !inherits(value, expected)) {
    stop(sprintf("`%s` needs a %s element.", element, kinds[[element]]), call. = FALSE)
  }
  spec[[element]] <- value
  return(spec)
}

#' Build a complete theme spec
#'
#' @description
#' Creates one element per theme slot from three house-style inputs. Text
#' sizes are `base_size` times the factors in `rel`; text uses `ink`, the
#' background uses `paper`, and lines mix the two.
#'
#' @param base_size Base font size in points.
#' @param ink Main text colour.
#' @param paper Background colour.
#' @param grid Whether to include grid lines.
#' @return A theme spec list with elements `axis.text`, `axis.title`,
#'   `plot.title`, `legend.text`, `axis.line`, `panel.grid` and
#'   `panel.background`.
#' @export
theme_spec <- function(base_size = 11, ink = "grey20", paper = "white", grid = TRUE) {
  rel <- c(axis.text = 0.8, axis.title = 1, plot.title = 1.2, legend.text = 0.8)
  line_colour <- grDevices::colorRampPalette(c(ink, paper))(5L)
  spec <- list()

  # text elements
  spec <- set_element(spec, "axis.text",
                      el_text(size = base_size * rel[["axis.text"]], colour = ink))
  spec <- set_element(spec, "axis.title",
                      el_text(size = base_size * rel[["axis.title"]], colour = ink))
  spec <- set_element(spec, "plot.title",
                      el_text(size = base_size * rel[["plot.title"]], colour = ink,
                              face = "bold", hjust = 0))
  spec <- set_element(spec, "legend.text",
                      el_text(size = base_size * rel[["legend.text"]], colour = ink))

  # line elements
  spec <- set_element(spec, "axis.line",
                      el_line(colour = line_colour[[2]], linewidth = base_size / 22))
  if (grid) {
    spec <- set_element(spec, "panel.grid",
                        el_line(colour = line_colour[[4]], linewidth = base_size / 44))
  }

  # background
  spec <- set_element(spec, "panel.background", el_rect(fill = paper))
  return(spec)
}

#' Set axis tick label text
#'
#' Updates only the fields that are not `NULL`, starting from the current
#' `axis.text` element of `spec` or from [el_text()] defaults.
#'
#' @param spec A theme spec list.
#' @param size Font size in points.
#' @param colour Text colour.
#' @param face Font face.
#' @param angle Rotation in degrees.
#' @return The updated spec.
#' @family element setters
#' @export
set_axis_text <- function(spec = list(), size = NULL, colour = NULL, face = NULL,
                          angle = NULL) {
  # start from the current element, or the defaults
  current <- spec[["axis.text"]]
  if (is.null(current)) {
    current <- el_text()
  }

  # override only the supplied fields
  if (!is.null(size)) {
    current$size <- size
  }
  if (!is.null(colour)) {
    current$colour <- colour
  }
  if (!is.null(face)) {
    current$face <- face
  }
  if (!is.null(angle)) {
    current$angle <- angle
  }

  # rebuild through the constructor so the new values are validated
  value <- do.call(el_text, unclass(current))
  spec <- set_element(spec, "axis.text", value)
  return(spec)
}

#' Set grid lines
#'
#' Updates only the fields that are not `NULL`, starting from the current
#' `panel.grid` element of `spec` or from [el_line()] defaults. With
#' `show = FALSE` the grid is removed.
#'
#' @param spec A theme spec list.
#' @param colour Line colour.
#' @param linewidth Line width in mm.
#' @param linetype Line type.
#' @param show Whether grid lines are drawn at all.
#' @return The updated spec.
#' @family element setters
#' @export
set_panel_grid <- function(spec = list(), colour = NULL, linewidth = NULL, linetype = NULL,
                           show = TRUE) {
  if (!show) {
    return(set_element(spec, "panel.grid", NULL))
  }
  # start from the current element, or the defaults
  current <- spec[["panel.grid"]]
  if (is.null(current)) {
    current <- el_line()
  }

  # override only the supplied fields
  if (!is.null(colour)) {
    current$colour <- colour
  }
  if (!is.null(linewidth)) {
    current$linewidth <- linewidth
  }
  if (!is.null(linetype)) {
    current$linetype <- linetype
  }

  # rebuild through the constructor so the new values are validated
  value <- do.call(el_line, unclass(current))
  spec <- set_element(spec, "panel.grid", value)
  return(spec)
}

#' Convert a theme spec to a ggplot2 theme
#'
#' @param spec A theme spec list built with [theme_spec()] or the `set_*()`
#'   helpers.
#' @return A ggplot2 theme object.
#' @export
as_ggtheme <- function(spec) {
  args <- lapply(spec, function(el) {
    fields <- unclass(el)
    switch(class(el)[[1]],
      lw_element_text = do.call(ggplot2::element_text, fields),
      lw_element_line = do.call(ggplot2::element_line, fields),
      lw_element_rect = do.call(ggplot2::element_rect, fields),
      stop(sprintf("Cannot convert element of class `%s`.", class(el)[[1]]), call. = FALSE)
    )
  })
  theme <- do.call(ggplot2::theme, args)
  return(theme)
}
