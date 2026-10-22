#' Choose a panel grid for a number of facets
#'
#' Picks the number of rows and columns for `n` panels, keeping the grid as
#' square as possible without exceeding `max_cols` columns.
#'
#' @param n Number of panels.
#' @param max_cols Maximum number of columns.
#' @return A named integer vector with elements `nrow` and `ncol`.
#' @keywords internal
choose_facet_layout <- function(n, max_cols = 4L) {
  if (n < 1) {
    stop("`n` must be at least 1.", call. = FALSE)
  }
  ncol <- min(max_cols, ceiling(sqrt(n)))
  nrow <- ceiling(n / ncol)
  c(nrow = as.integer(nrow), ncol = as.integer(ncol))
}

#' Facet a plot by row and column variables
#'
#' Wraps [ggplot2::facet_grid()] so that rows and columns can be given as
#' character vectors of column names.
#'
#' @param plot A [ggplot2::ggplot()] object.
#' @param rows Character vector of variables defining facet rows, or `NULL`.
#' @param cols Character vector of variables defining facet columns, or `NULL`.
#' @param scales Whether axis scales are shared across panels.
#' @param labeller A labeller function passed to [ggplot2::facet_grid()].
#' @return The modified plot.
#' @family facets
#' @export
facet_by <- function(plot, rows = NULL, cols = NULL,
                     scales = c("fixed", "free", "free_x", "free_y"),
                     labeller = "label_value") {
  scales <- match.arg(scales)
  if (is.null(rows) && is.null(cols)) {
    stop("Supply at least one of `rows` or `cols`.", call. = FALSE)
  }
  plot + ggplot2::facet_grid(
    rows = if (is.null(rows)) NULL else ggplot2::vars(!!!rlang::syms(rows)),
    cols = if (is.null(cols)) NULL else ggplot2::vars(!!!rlang::syms(cols)),
    scales = scales,
    labeller = labeller
  )
}

#' Wrap facets into a balanced grid
#'
#' Wraps a plot by one variable, sizing the grid with
#' [choose_facet_layout()] based on the number of distinct levels.
#'
#' @param plot A [ggplot2::ggplot()] object.
#' @param var Name of the faceting column.
#' @param max_cols Maximum number of columns.
#' @param scales Whether axis scales are shared across panels.
#' @return The modified plot.
#' @family facets
#' @export
facet_wrap_balanced <- function(plot, var, max_cols = 4L,
                                scales = c("fixed", "free", "free_x", "free_y")) {
  scales <- match.arg(scales)
  levels_n <- plot$data %>%
    dplyr::pull(!!rlang::sym(var)) %>%
    dplyr::n_distinct()
  layout <- choose_facet_layout(levels_n, max_cols = max_cols)
  plot + ggplot2::facet_wrap(
    ggplot2::vars(!!rlang::sym(var)),
    nrow = layout[["nrow"]],
    ncol = layout[["ncol"]],
    scales = scales
  )
}

#' Append group sizes to facet labels
#'
#' Adds a column of labels of the form `"level (n = 12)"` that can be used
#' as a faceting variable so that each strip shows its sample size.
#'
#' @param data A data frame of observations to plot.
#' @param var Name of the grouping column.
#' @param new_col Name of the label column to create.
#' @return `data` with an extra factor column `new_col`.
#' @family facets
#' @export
add_facet_counts <- function(data, var, new_col = paste0(var, "_label")) {
  counts <- data %>%
    dplyr::count(!!rlang::sym(var), name = ".n") %>%
    dplyr::mutate(
      !!new_col := paste0(.data[[var]], " (n = ", .data$.n, ")")
    )
  data %>%
    dplyr::left_join(dplyr::select(counts, -".n"), by = var) %>%
    dplyr::mutate(
      !!new_col := factor(.data[[new_col]], levels = counts[[new_col]])
    )
}

#' Labeller that wraps long strip text
#'
#' Returns a labeller that breaks facet strip labels onto several lines once
#' they exceed `width` characters.
#'
#' @param width Maximum number of characters per line.
#' @return A labeller function for use in facet calls.
#' @family facets
#' @export
label_wrapped <- function(width = 20) {
  ggplot2::as_labeller(function(x) {
    vapply(x, function(s) paste(strwrap(s, width = width), collapse = "\n"), character(1))
  })
}

#' Style facet strips as left-aligned headers
#'
#' Removes the strip background and left-aligns bold strip text, which reads
#' better than the default grey boxes in reports.
#'
#' @param size Relative size of the strip text.
#' @return A [ggplot2::theme()] object.
#' @family facets
#' @export
strip_theme_top <- function(size = 1) {
  ggplot2::theme(
    strip.background = ggplot2::element_blank(),
    strip.text.x = ggplot2::element_text(
      face = "bold", hjust = 0, size = ggplot2::rel(size)
    ),
    strip.placement = "outside"
  )
}

#' Style facet strips as horizontal row headers
#'
#' Removes the strip background and draws row strip text horizontally on
#' the left, so long row labels stay readable.
#'
#' @param size Relative size of the strip text.
#' @return A [ggplot2::theme()] object.
#' @family facets
#' @export
strip_theme_left <- function(size = 1) {
  ggplot2::theme(
    strip.background = ggplot2::element_blank(),
    strip.text.y.left = ggplot2::element_text(
      face = "bold", hjust = 1, angle = 0, size = ggplot2::rel(size)
    ),
    strip.placement = "outside"
  )
}
