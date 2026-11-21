#' Convert a plot size to inches
#'
#' @param width,height Plot size in `units`.
#' @param units Units of `width` and `height`.
#' @param res Resolution in pixels per inch, used when `units = "px"`.
#' @return A list with elements `width` and `height` in inches.
#' @keywords internal
device_size <- function(width, height, units = c("in", "cm", "mm", "px"), res = 96) {
  units <- match.arg(units)
  per_inch <- c(`in` = 1, cm = 2.54, mm = 25.4, px = res)
  if (!is.numeric(width) || length(width) != 1L || width <= 0) {
    stop("`width` must be a single positive number.", call. = FALSE)
  }
  if (!is.numeric(height) || length(height) != 1L || height <= 0) {
    stop("`height` must be a single positive number.", call. = FALSE)
  }
  list(width = width / per_inch[[units]], height = height / per_inch[[units]])
}

#' Guess the output format from a file name
#'
#' @param path Output file path.
#' @return A lower-case file extension such as `"pdf"` or `"png"`.
#' @keywords internal
format_from_path <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (identical(ext, "jpg")) {
    ext <- "jpeg"
  }
  if (!nzchar(ext)) {
    stop(sprintf("Cannot guess a format from `%s`.", basename(path)), call. = FALSE)
  }
  ext
}

#' Write a plot to a PDF file
#'
#' Opens the device, runs `plot_fn` and always closes the device again. A
#' failing `plot_fn` is reported as a warning so that batch exports go on.
#'
#' @param plot_fn A function without arguments that draws the plot.
#' @param path Output file path.
#' @param width,height Plot size in inches.
#' @param res Ignored for vector formats; kept so all writers share one
#'   signature.
#' @param bg Background colour.
#' @return Invisibly, `path` with attributes `format` and `ok`.
#' @family writers
#' @export
save_pdf <- function(plot_fn, path, width = 7, height = 5, res = 150, bg = "white") {
  if (!is.function(plot_fn)) {
    stop("`plot_fn` must be a function.", call. = FALSE)
  }
  if (!identical(format_from_path(path), "pdf")) {
    stop(sprintf("`path` must end in .%s.", "pdf"), call. = FALSE)
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

  grDevices::pdf(path, width = width, height = height, bg = bg)
  dev <- grDevices::dev.cur()
  on.exit(grDevices::dev.off(dev), add = TRUE)
  ok <- tryCatch({
    plot_fn()
    TRUE
  }, error = function(e) {
    warning(sprintf("Plot for `%s` failed: %s", basename(path), conditionMessage(e)),
            call. = FALSE)
    FALSE
  })
  invisible(structure(path, format = "pdf", ok = ok))
}

#' Write a plot to an SVG file
#'
#' @inheritParams save_pdf
#' @return Invisibly, `path` with attributes `format` and `ok`.
#' @family writers
#' @export
save_svg <- function(plot_fn, path, width = 7, height = 5, res = 150, bg = "white") {
  if (!is.function(plot_fn)) {
    stop("`plot_fn` must be a function.", call. = FALSE)
  }
  if (!identical(format_from_path(path), "svg")) {
    stop(sprintf("`path` must end in .%s.", "svg"), call. = FALSE)
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

  grDevices::svg(path, width = width, height = height, bg = bg)
  dev <- grDevices::dev.cur()
  on.exit(grDevices::dev.off(dev), add = TRUE)
  ok <- tryCatch({
    plot_fn()
    TRUE
  }, error = function(e) {
    warning(sprintf("Plot for `%s` failed: %s", basename(path), conditionMessage(e)),
            call. = FALSE)
    FALSE
  })
  invisible(structure(path, format = "svg", ok = ok))
}

#' Write a plot to a PNG file
#'
#' @inheritParams save_pdf
#' @param res Resolution in pixels per inch.
#' @return Invisibly, `path` with attributes `format` and `ok`.
#' @family writers
#' @export
save_png <- function(plot_fn, path, width = 7, height = 5, res = 150, bg = "white") {
  if (!is.function(plot_fn)) {
    stop("`plot_fn` must be a function.", call. = FALSE)
  }
  if (!identical(format_from_path(path), "png")) {
    stop(sprintf("`path` must end in .%s.", "png"), call. = FALSE)
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

  grDevices::png(path, width = width, height = height, units = "in", res = res, bg = bg)
  dev <- grDevices::dev.cur()
  on.exit(grDevices::dev.off(dev), add = TRUE)
  ok <- tryCatch({
    plot_fn()
    TRUE
  }, error = function(e) {
    warning(sprintf("Plot for `%s` failed: %s", basename(path), conditionMessage(e)),
            call. = FALSE)
    FALSE
  })
  invisible(structure(path, format = "png", ok = ok))
}

#' Write a plot to a JPEG file
#'
#' @inheritParams save_png
#' @return Invisibly, `path` with attributes `format` and `ok`.
#' @family writers
#' @export
save_jpeg <- function(plot_fn, path, width = 7, height = 5, res = 150, bg = "white") {
  if (!is.function(plot_fn)) {
    stop("`plot_fn` must be a function.", call. = FALSE)
  }
  if (!identical(format_from_path(path), "jpeg")) {
    stop(sprintf("`path` must end in .%s.", "jpeg"), call. = FALSE)
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

  grDevices::jpeg(path, width = width, height = height, units = "in", res = res, bg = bg)
  dev <- grDevices::dev.cur()
  on.exit(grDevices::dev.off(dev), add = TRUE)
  ok <- tryCatch({
    plot_fn()
    TRUE
  }, error = function(e) {
    warning(sprintf("Plot for `%s` failed: %s", basename(path), conditionMessage(e)),
            call. = FALSE)
    FALSE
  })
  invisible(structure(path, format = "jpeg", ok = ok))
}

#' Write a plot in the format implied by the file name
#'
#' Raster formats are limited to `max_px` pixels per side so that a size
#' given in the wrong units fails early instead of filling the disk.
#'
#' @param plot_fn A function without arguments that draws the plot.
#' @param path Output file path; its extension selects the writer.
#' @param width,height Plot size in `units`.
#' @param units Units of `width` and `height`, passed to [device_size()].
#' @param res Resolution in pixels per inch for raster formats.
#' @param max_px Largest raster width or height in pixels.
#' @return Invisibly, `path` with attributes `format` and `ok`.
#' @export
save_plot <- function(plot_fn, path, width = 7, height = 5, units = "in", res = 150,
                      max_px = 10000) {
  format <- format_from_path(path)
  size <- device_size(width, height, units = units, res = res)
  if (format %in% c("png", "jpeg")) {
    if (size$width * res > max_px) {
      stop(sprintf("`width` is too large: %.0f pixels at %d dpi.", size$width * res, res),
           call. = FALSE)
    }
    if (size$height * res > max_px) {
      stop(sprintf("`height` is too large: %.0f pixels at %d dpi.", size$height * res, res),
           call. = FALSE)
    }
  }

  switch(format,
    pdf = save_pdf(plot_fn, path, width = size$width, height = size$height, res = res),
    svg = save_svg(plot_fn, path, width = size$width, height = size$height, res = res),
    png = save_png(plot_fn, path, width = size$width, height = size$height, res = res),
    jpeg = save_jpeg(plot_fn, path, width = size$width, height = size$height, res = res),
    stop(sprintf("Unsupported format `%s`.", format), call. = FALSE)
  )
}

#' Write a named set of plots
#'
#' Writes each element of `plots` to `dir`, using its name as the file stem.
#' Existing files are kept unless `overwrite = TRUE`.
#'
#' @param plots A named list of plot functions.
#' @param dir Output directory, created if needed.
#' @param format Output format for all files.
#' @param overwrite Whether to replace existing files.
#' @param ... Passed to [save_plot()].
#' @return A data frame with columns `name`, `path`, `status` and `bytes`,
#'   where `status` is one of `"written"`, `"failed"` or `"skipped"`.
#' @export
save_plot_set <- function(plots, dir, format = "pdf", overwrite = FALSE, ...) {
  stems <- names(plots)
  if (is.null(stems) || any(!nzchar(stems)) || anyDuplicated(stems)) {
    stop("`plots` must be a list with unique, non-empty names.", call. = FALSE)
  }
  ext <- if (identical(format, "jpeg")) "jpg" else format
  paths <- file.path(dir, paste0(stems, ".", ext))
  status <- character(length(paths))

  for (i in seq_along(plots)) {
    if (file.exists(paths[[i]]) && !overwrite) {
      status[[i]] <- "skipped"
      next
    }
    res <- save_plot(plots[[i]], paths[[i]], ...)
    status[[i]] <- if (isTRUE(attr(res, "ok"))) "written" else "failed"
  }
  data.frame(
    name = stems,
    path = paths,
    status = status,
    bytes = file.size(paths),
    stringsAsFactors = FALSE
  )
}
