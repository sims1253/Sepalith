#' Recycle a legend attribute to the number of labels
#'
#' @param value Attribute values.
#' @param n Number of legend entries.
#' @param name Attribute name used in error messages.
#' @return `value` recycled to length `n`.
#' @noRd
recycle_legend_arg <- function(value, n, name) {
  if (length(value) == 1L) {
    return(rep(value, n))
  }
  if (length(value) != n) {
    stop(sprintf("`%s` must have length 1 or %d, not %d.", name, n, length(value)),
         call. = FALSE)
  }
  value
}

#' Legend entries for point marks
#'
#' @param labels Character vector of legend labels.
#' @param col Colours, recycled to the number of labels.
#' @param pch Point symbols, recycled to the number of labels.
#' @param cex Point size, recycled to the number of labels.
#' @param title Optional legend title.
#' @return A `plotloom_legend` list with elements `type`, `labels`, `col`,
#'   `pch`, `cex` and `title`.
#' @family legend specs
#' @export
legend_points <- function(labels, col = "black", pch = 19L, cex = 1, title = NULL) {
  labels <- as.character(labels)
  n <- length(labels)
  if (n == 0L) {
    stop("`labels` must contain at least one label.", call. = FALSE)
  }
  if (anyDuplicated(labels)) {
    stop("`labels` must be unique.", call. = FALSE)
  }

  col <- recycle_legend_arg(col, n, "col")
  pch <- recycle_legend_arg(pch, n, "pch")
  cex <- recycle_legend_arg(cex, n, "cex")
  if (any(!pch %in% 0:25)) {
    stop("`pch` must only contain codes from 0 to 25.", call. = FALSE)
  }
  if (any(cex <= 0)) {
    stop("`cex` must be positive.", call. = FALSE)
  }

  structure(
    list(type = "points", labels = labels, col = col, pch = pch, cex = cex, title = title),
    class = "plotloom_legend"
  )
}

#' Legend entries for line marks
#'
#' @param labels Character vector of legend labels.
#' @param col Colours, recycled to the number of labels.
#' @param lty Line types, recycled to the number of labels.
#' @param lwd Line widths, recycled to the number of labels.
#' @param title Optional legend title.
#' @return A `plotloom_legend` list with elements `type`, `labels`, `col`,
#'   `lty`, `lwd` and `title`.
#' @family legend specs
#' @export
legend_lines <- function(labels, col = "black", lty = 1L, lwd = 1, title = NULL) {
  labels <- as.character(labels)
  n <- length(labels)
  if (n == 0L) {
    stop("`labels` must contain at least one label.", call. = FALSE)
  }
  if (anyDuplicated(labels)) {
    stop("`labels` must be unique.", call. = FALSE)
  }

  col <- recycle_legend_arg(col, n, "col")
  lty <- recycle_legend_arg(lty, n, "lty")
  lwd <- recycle_legend_arg(lwd, n, "lwd")
  if (any(!lty %in% 0:6)) {
    stop("`lty` must only contain codes from 0 to 6.", call. = FALSE)
  }
  if (any(lwd <= 0)) {
    stop("`lwd` must be positive.", call. = FALSE)
  }

  structure(
    list(type = "lines", labels = labels, col = col, lty = lty, lwd = lwd, title = title),
    class = "plotloom_legend"
  )
}

#' Legend entries for filled marks
#'
#' @param labels Character vector of legend labels.
#' @param col Colours, recycled to the number of labels.
#' @param border Border colours, recycled to the number of labels.
#' @param density Shading line density, recycled to the number of labels;
#'   `NA` for solid fill.
#' @param title Optional legend title.
#' @return A `plotloom_legend` list with elements `type`, `labels`, `col`,
#'   `border`, `density` and `title`.
#' @family legend specs
#' @export
legend_fill <- function(labels, col = "grey70", border = "black", density = NA,
                        title = NULL) {
  labels <- as.character(labels)
  n <- length(labels)
  if (n == 0L) {
    stop("`labels` must contain at least one label.", call. = FALSE)
  }
  if (anyDuplicated(labels)) {
    stop("`labels` must be unique.", call. = FALSE)
  }

  col <- recycle_legend_arg(col, n, "col")
  border <- recycle_legend_arg(border, n, "border")
  density <- recycle_legend_arg(density, n, "density")
  if (any(is.na(col))) {
    stop("`col` must not be missing.", call. = FALSE)
  }
  if (any(density < 0, na.rm = TRUE)) {
    stop("`density` must be positive.", call. = FALSE)
  }

  structure(
    list(type = "fill", labels = labels, col = col, border = border, density = density,
         title = title),
    class = "plotloom_legend"
  )
}

#' Resolve a legend position
#'
#' @param where Named position. `"outside"` places the legend to the right
#'   of the plot region.
#' @param inset Inset from the plot border as a fraction of the region.
#' @return A list with elements `x`, `inset` and `xpd`, ready to splice into
#'   [graphics::legend()].
#' @keywords internal
legend_position <- function(where = c("topright", "topleft", "bottomright",
                                      "bottomleft", "top", "bottom", "outside"),
                            inset = 0.02) {
  where <- match.arg(where)
  switch(where,
    topright = list(x = "topright", inset = c(inset, inset), xpd = FALSE),
    topleft = list(x = "topleft", inset = c(inset, inset), xpd = FALSE),
    bottomright = list(x = "bottomright", inset = c(inset, inset), xpd = FALSE),
    bottomleft = list(x = "bottomleft", inset = c(inset, inset), xpd = FALSE),
    top = list(x = "top", inset = c(0, inset), xpd = FALSE),
    bottom = list(x = "bottom", inset = c(0, inset), xpd = FALSE),
    outside = list(x = "topright", inset = c(-0.25, inset), xpd = TRUE)
  )
}

#' Draw a legend from a spec
#'
#' @description
#' Translates a `plotloom_legend` spec into a call to
#' [graphics::legend()]: points become `pch`/`pt.cex`, lines become
#' `lty`/`lwd` and fills become `fill`/`border`/`density`. Legends at the
#' `top` or `bottom` are laid out horizontally.
#'
#' @param spec A `plotloom_legend` from [legend_points()], [legend_lines()]
#'   or [legend_fill()].
#' @param where Named position passed to [legend_position()].
#' @param title Legend title; defaults to the title stored in `spec`.
#' @return Invisibly, the value of [graphics::legend()].
#' @export
draw_legend <- function(spec, where = "topright", title = spec$title) {
  if (!inherits(spec, "plotloom_legend")) {
    stop("`spec` must be a plotloom_legend.", call. = FALSE)
  }
  horiz <- where %in% c("top", "bottom")
  args <- c(
    legend_position(where),
    list(legend = spec$labels, title = title, bty = "n", horiz = horiz)
  )

  args <- switch(spec$type,
    points = c(args, list(
      col = spec$col,
      pch = spec$pch,
      pt.cex = spec$cex
    )),
    lines = c(args, list(
      col = spec$col,
      lty = spec$lty,
      lwd = spec$lwd
    )),
    fill = c(args, list(
      fill = spec$col,
      border = spec$border,
      density = spec$density
    )),
    stop(sprintf("Unknown legend type `%s`.", spec$type), call. = FALSE)
  )
  invisible(do.call(legend, args))
}

#' Combine legend specs of the same type
#'
#' Entries are concatenated field by field in argument order; the title of
#' the first spec is kept.
#'
#' @param ... `plotloom_legend` objects.
#' @return A single `plotloom_legend`.
#' @export
legend_combine <- function(...) {
  specs <- list(...)
  if (length(specs) == 0L) {
    stop("Supply at least one legend spec.", call. = FALSE)
  }
  types <- vapply(specs, function(s) s$type, character(1))
  if (length(unique(types)) != 1L) {
    stop(sprintf("Cannot combine legend types: %s.", paste(unique(types), collapse = ", ")),
         call. = FALSE)
  }

  fields <- setdiff(names(specs[[1]]), c("type", "title"))
  out <- list(type = types[[1]])
  for (field in fields) {
    out[[field]] <- unlist(lapply(specs, `[[`, field), use.names = FALSE)
  }
  if (anyDuplicated(out$labels)) {
    stop("`labels` must be unique.", call. = FALSE)
  }
  out["title"] <- list(specs[[1]]$title)
  structure(out, class = "plotloom_legend")
}
