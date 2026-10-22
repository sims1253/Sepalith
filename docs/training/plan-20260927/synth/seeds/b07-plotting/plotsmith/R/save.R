#' Output dimensions for a named preset
#'
#' Looks up width and height in inches for the size presets used across
#' plotsmith output functions.
#'
#' @param preset Name of the size preset.
#' @return A named numeric vector with elements `width` and `height`.
#' @family export
#' @export
plot_dimensions <- function(preset = c("slide", "a4", "square", "thumb")) {
  preset <- match.arg(preset)
  dims <- switch(preset,
    slide = c(width = 10, height = 5.625),
    a4 = c(width = 7.5, height = 10),
    square = c(width = 6, height = 6),
    thumb = c(width = 3, height = 2)
  )
  return(dims)
}

#' Build a safe file name for a plot
#'
#' Lower-cases `name`, replaces anything that is not a letter, digit or
#' hyphen with an underscore and appends the extension.
#'
#' @param name Base name of the file.
#' @param ext File extension without the leading dot.
#' @return A single file name.
#' @keywords internal
plot_file_name <- function(name, ext) {
  stem <- tolower(name)
  stem <- gsub("[^a-z0-9-]+", "_", stem)
  stem <- gsub("^_+|_+$", "", stem)
  if (!nzchar(stem)) {
    stop("`name` does not contain any usable characters.", call. = FALSE)
  }
  return(paste0(stem, ".", ext))
}

#' Save a plot as PNG
#'
#' Writes a plot to a PNG file using one of the size presets.
#'
#' @param plot A [ggplot2::ggplot()] object.
#' @param path Output file path.
#' @param preset Name of the size preset.
#' @param dpi Resolution in dots per inch.
#' @return `path`, invisibly.
#' @family export
#' @export
save_plot_png <- function(plot, path, preset = "slide", dpi = 300) {
  dims <- plot_dimensions(preset)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(
    filename = path, plot = plot, device = "png",
    width = dims[["width"]], height = dims[["height"]], units = "in",
    dpi = dpi, bg = "white"
  )
  return(invisible(path))
}

#' Save a plot as PDF
#'
#' Writes a plot to a PDF file using one of the size presets.
#'
#' @param plot A [ggplot2::ggplot()] object.
#' @param path Output file path.
#' @param preset Name of the size preset.
#' @param embed_fonts Whether to use the cairo device so fonts are embedded.
#' @return `path`, invisibly.
#' @family export
#' @export
save_plot_pdf <- function(plot, path, preset = "a4", embed_fonts = TRUE) {
  dims <- plot_dimensions(preset)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(
    filename = path, plot = plot,
    device = if (embed_fonts) grDevices::cairo_pdf else "pdf",
    width = dims[["width"]], height = dims[["height"]], units = "in"
  )
  return(invisible(path))
}

#' Save a plot in several formats at once
#'
#' Writes the same plot to every requested format, naming each file after
#' `name` inside `dir`.
#'
#' @param plot A [ggplot2::ggplot()] object.
#' @param name Base name of the output files.
#' @param dir Output directory.
#' @param formats Character vector of formats, any of `"png"` and `"pdf"`.
#' @param preset Name of the size preset.
#' @return A named character vector of written paths, invisibly.
#' @family export
#' @export
save_plot <- function(plot, name, dir = "figures", formats = c("png", "pdf"),
                      preset = "slide") {
  formats <- match.arg(formats, several.ok = TRUE)
  paths <- character(length(formats))
  names(paths) <- formats
  for (fmt in formats) {
    path <- file.path(dir, plot_file_name(name, fmt))
    paths[[fmt]] <- switch(fmt,
      png = save_plot_png(plot, path, preset = preset),
      pdf = save_plot_pdf(plot, path, preset = preset)
    )
  }
  return(invisible(paths))
}

#' Save a named list of plots
#'
#' Calls [save_plot()] on each element of a named list, using the list names
#' as file names, and reports how many files were written.
#'
#' @param plots A named list of [ggplot2::ggplot()] objects.
#' @param dir Output directory.
#' @param ... Passed to [save_plot()].
#' @return A character vector of all written paths, invisibly.
#' @family export
#' @export
save_plot_set <- function(plots, dir = "figures", ...) {
  if (is.null(names(plots)) || any(!nzchar(names(plots)))) {
    stop("`plots` must be a fully named list.", call. = FALSE)
  }
  written <- vector("list", length(plots))
  for (i in seq_along(plots)) {
    written[[i]] <- save_plot(plots[[i]], name = names(plots)[i], dir = dir, ...)
  }
  written <- unlist(written, use.names = FALSE)
  message("Wrote ", length(written), " file(s) to ", dir)
  return(invisible(written))
}
