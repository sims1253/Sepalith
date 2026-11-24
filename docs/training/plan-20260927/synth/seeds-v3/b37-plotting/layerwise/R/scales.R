#' Validate palette arguments
#'
#' @param n Number of colours.
#' @param hue Base hue in degrees.
#' @param direction `1` for the natural order, `-1` to reverse.
#' @return `NULL`, invisibly, if all arguments are valid.
#' @keywords internal
check_palette_args <- function(n, hue, direction) {
  # n
  if (!is.numeric(n) || length(n) != 1L || is.na(n) || n < 1) {
    stop("`n` must be a single number of at least 1.", call. = FALSE)
  }
  # hue
  if (!is.numeric(hue) || length(hue) != 1L || is.na(hue) || hue < 0 || hue > 360) {
    stop("`hue` must be a single number between 0 and 360.", call. = FALSE)
  }
  # direction
  if (!identical(direction, 1) && !identical(direction, -1)) {
    stop("`direction` must be 1 or -1.", call. = FALSE)
  }
  invisible(NULL)
}

#' Colour palettes for layerwise scales
#'
#' @description
#' Builds `n` HCL colours of one family around `hue`. The chroma and
#' luminance ranges in `chroma` and `luminance` are shared by all families:
#' `sequential` runs light to dark at one hue; `diverging` runs from `hue`
#' through a light centre to the opposite hue; `qualitative` spaces hues
#' evenly at fixed chroma and luminance; `cyclic` does the same but returns
#' to the starting colour.
#'
#' @param family Palette family.
#' @param n Number of colours.
#' @param hue Base hue in degrees.
#' @param direction `1` for the natural order, `-1` to reverse.
#' @return A character vector of `n` hex colours.
#' @export
lw_palette <- function(family = c("sequential", "diverging", "qualitative", "cyclic"),
                       n = 7L, hue = 220, direction = 1) {
  family <- match.arg(family)
  check_palette_args(n, hue, direction)
  chroma <- c(low = 20, high = 70)
  luminance <- c(light = 90, dark = 30)
  mid <- (luminance[["light"]] + luminance[["dark"]]) / 2

  cols <- switch(family,
    sequential = {
      l <- seq(luminance[["light"]], luminance[["dark"]], length.out = n)
      chr <- seq(chroma[["low"]], chroma[["high"]], length.out = n)
      grDevices::hcl(h = hue, c = chr, l = l)
    },
    diverging = {
      pos <- seq(-1, 1, length.out = n)
      h <- ifelse(pos < 0, hue, (hue + 180) %% 360)
      l <- luminance[["light"]] - abs(pos) * (luminance[["light"]] - luminance[["dark"]])
      chr <- chroma[["low"]] + abs(pos) * (chroma[["high"]] - chroma[["low"]])
      grDevices::hcl(h = h, c = chr, l = l)
    },
    qualitative = {
      h <- (hue + seq(0, 360, length.out = n + 1L)[seq_len(n)]) %% 360
      grDevices::hcl(h = h, c = chroma[["high"]], l = mid)
    },
    cyclic = {
      h <- (hue + seq(0, 360, length.out = n)) %% 360
      grDevices::hcl(h = h, c = chroma[["high"]], l = mid)
    }
  )
  if (direction == -1) {
    cols <- rev(cols)
  }
  return(cols)
}

#' Draw a palette preview
#'
#' Draws one swatch per colour with its hex code, choosing black or white
#' text by the luminance of the swatch, and a legend naming the family.
#'
#' @inheritParams lw_palette
#' @param labels Whether to print the hex codes on the swatches.
#' @return Invisibly, the drawn colours.
#' @export
palette_preview <- function(family = "sequential", n = 7L, hue = 220, direction = 1,
                            labels = TRUE) {
  cols <- lw_palette(family, n = n, hue = hue, direction = direction)
  old <- graphics::par(mar = c(3, 1, 2, 1))
  on.exit(graphics::par(old), add = TRUE)
  left <- seq_len(n) - 1
  right <- seq_len(n)

  graphics::plot.new()
  graphics::plot.window(xlim = c(0, n), ylim = c(0, 1))
  graphics::rect(left, 0, right, 1, col = cols, border = "white")

  # hex codes in black or white, whichever reads better on the swatch
  graphics::axis(1, at = left + 0.5, labels = seq_len(n), tick = FALSE)
  if (labels) {
    lum <- colSums(grDevices::col2rgb(cols) * c(0.299, 0.587, 0.114)) / 255
    ink <- ifelse(lum > 0.5, "black", "white")
    graphics::text(left + 0.5, 0.5, labels = cols, col = ink, srt = 90, cex = 0.7)
  }

  graphics::title(main = sprintf("%s (hue %s)", family, format(hue)))
  graphics::legend("top", legend = family, fill = cols[[ceiling(n / 2)]], bty = "n",
                   inset = c(0, -0.1), xpd = TRUE)
  invisible(cols)
}

#' Sequential colour scale
#'
#' @inheritParams lw_palette
#' @param aesthetics Aesthetic the scale applies to.
#' @param na.value Colour for missing values.
#' @param limits Data range of the scale, or `NULL` for the data range.
#' @param ... Passed to [ggplot2::continuous_scale()].
#' @return A ggplot2 scale.
#' @family colour scales
#' @export
scale_lw_sequential <- function(aesthetics = c("colour", "fill"), hue = 220, direction = 1,
                                na.value = "grey85", limits = NULL, ...) {
  # validate arguments
  aesthetics <- match.arg(aesthetics)
  if (!is.character(na.value) || length(na.value) != 1L) {
    stop("`na.value` must be a single colour.", call. = FALSE)
  }
  if (!is.null(limits) && (!is.numeric(limits) || length(limits) != 2L)) {
    stop("`limits` must be NULL or two numbers.", call. = FALSE)
  }
  cols <- lw_palette("sequential", n = 9L, hue = hue, direction = direction)

  # build the scale
  scale <- ggplot2::continuous_scale(
    aesthetics = aesthetics,
    palette = scales::gradient_n_pal(cols),
    na.value = na.value,
    limits = limits,
    guide = "colourbar",
    ...
  )
  return(scale)
}

#' Diverging colour scale
#'
#' @inheritParams scale_lw_sequential
#' @param midpoint Data value mapped to the light centre colour.
#' @return A ggplot2 scale.
#' @family colour scales
#' @export
scale_lw_diverging <- function(aesthetics = c("colour", "fill"), hue = 220, direction = 1,
                               na.value = "grey85", limits = NULL, midpoint = 0, ...) {
  # validate arguments
  aesthetics <- match.arg(aesthetics)
  if (!is.character(na.value) || length(na.value) != 1L) {
    stop("`na.value` must be a single colour.", call. = FALSE)
  }
  if (!is.null(limits) && (!is.numeric(limits) || length(limits) != 2L)) {
    stop("`limits` must be NULL or two numbers.", call. = FALSE)
  }
  if (!is.numeric(midpoint) || length(midpoint) != 1L || !is.finite(midpoint)) {
    stop("`midpoint` must be a single finite number.", call. = FALSE)
  }
  cols <- lw_palette("diverging", n = 9L, hue = hue, direction = direction)

  # build the scale
  scale <- ggplot2::continuous_scale(
    aesthetics = aesthetics,
    palette = scales::gradient_n_pal(cols),
    na.value = na.value,
    limits = limits,
    guide = "colourbar",
    rescaler = function(x, to = c(0, 1), from = range(x, na.rm = TRUE)) {
      scales::rescale_mid(x, to = to, from = from, mid = midpoint)
    },
    ...
  )
  return(scale)
}

#' Binned colour scale
#'
#' @inheritParams scale_lw_sequential
#' @param n_bins Number of colour bins.
#' @return A ggplot2 scale.
#' @family colour scales
#' @export
scale_lw_binned <- function(aesthetics = c("colour", "fill"), hue = 220, direction = 1,
                            na.value = "grey85", limits = NULL, n_bins = 5L, ...) {
  # validate arguments
  aesthetics <- match.arg(aesthetics)
  if (!is.character(na.value) || length(na.value) != 1L) {
    stop("`na.value` must be a single colour.", call. = FALSE)
  }
  if (!is.null(limits) && (!is.numeric(limits) || length(limits) != 2L)) {
    stop("`limits` must be NULL or two numbers.", call. = FALSE)
  }
  if (!is.numeric(n_bins) || length(n_bins) != 1L || n_bins < 2) {
    stop("`n_bins` must be a single number of at least 2.", call. = FALSE)
  }
  cols <- lw_palette("sequential", n = n_bins, hue = hue, direction = direction)

  # build the scale
  scale <- ggplot2::binned_scale(
    aesthetics = aesthetics,
    palette = function(x) cols[pmax(1L, ceiling(x * n_bins))],
    na.value = na.value,
    limits = limits,
    n.breaks = n_bins - 1L,
    guide = "coloursteps",
    ...
  )
  return(scale)
}

#' Qualitative colour scale
#'
#' @inheritParams scale_lw_sequential
#' @param drop Whether to drop unused factor levels.
#' @param ... Passed to [ggplot2::discrete_scale()].
#' @return A ggplot2 scale.
#' @family colour scales
#' @export
scale_lw_qualitative <- function(aesthetics = c("colour", "fill"), hue = 220, direction = 1,
                                 na.value = "grey85", drop = TRUE, ...) {
  # validate arguments
  aesthetics <- match.arg(aesthetics)
  if (!is.character(na.value) || length(na.value) != 1L) {
    stop("`na.value` must be a single colour.", call. = FALSE)
  }
  if (!is.logical(drop) || length(drop) != 1L || is.na(drop)) {
    stop("`drop` must be TRUE or FALSE.", call. = FALSE)
  }
  check_palette_args(1L, hue, direction)

  # build the scale
  scale <- ggplot2::discrete_scale(
    aesthetics = aesthetics,
    palette = function(n) lw_palette("qualitative", n = n, hue = hue, direction = direction),
    na.value = na.value,
    drop = drop,
    guide = "legend",
    ...
  )
  return(scale)
}
