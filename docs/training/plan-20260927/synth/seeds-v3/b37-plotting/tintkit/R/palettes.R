#' Validate common palette arguments
#'
#' @param n Number of colours.
#' @param hue Hue in degrees.
#' @param direction `1` for the natural order, `-1` to reverse.
#' @param alpha Opacity between 0 and 1.
#' @return `TRUE`, invisibly.
#' @noRd
checkPaletteArgs = function(n, hue, direction, alpha) {
  if (!is.numeric(n) || length(n) != 1L || n < 1 || n != round(n)) {
    stop(sprintf("`%s` must be %s.", "n", "a single positive whole number"), call. = FALSE)
  }
  if (!is.numeric(hue) || length(hue) != 1L || hue < 0 || hue > 360) {
    stop(sprintf("`%s` must be %s.", "hue", "a single number between 0 and 360"), call. = FALSE)
  }
  if (!direction %in% c(-1, 1)) {
    stop(sprintf("`%s` must be %s.", "direction", "1 or -1"), call. = FALSE)
  }
  if (!is.numeric(alpha) || length(alpha) != 1L || alpha < 0 || alpha > 1) {
    stop(sprintf("`%s` must be %s.", "alpha", "a single number between 0 and 1"), call. = FALSE)
  }
  invisible(TRUE)
}

#' Finish a palette
#'
#' @param cols Hex colours.
#' @param direction `1` for the natural order, `-1` to reverse.
#' @param alpha Opacity between 0 and 1.
#' @param type Palette type stored as an attribute.
#' @return `cols`, reversed and made transparent as requested, with class
#'   `tint_palette`.
#' @noRd
finishPalette = function(cols, direction, alpha, type) {
  if (direction == -1) {
    cols = rev(cols)
  }
  if (alpha < 1) {
    cols = grDevices::adjustcolor(cols, alpha.f = alpha)
  }
  structure(cols, type = type, class = "tint_palette")
}

#' Sequential palette
#'
#' @param n Number of colours.
#' @param hue Hue in degrees.
#' @param lum Luminance of the first and last colour, between 0 and 100.
#' @param chroma Chroma of the first and last colour, non-negative.
#' @param direction `1` for the natural order, `-1` to reverse.
#' @param alpha Opacity between 0 and 1.
#' @return A `tint_palette` character vector of length `n`.
#' @family tint palettes
#' @export
tintSequential = function(n, hue = 240, lum = c(92, 25), chroma = c(15, 60), direction = 1,
                          alpha = 1) {
  checkPaletteArgs(n, hue, direction, alpha)

  # ramp-specific arguments
  if (!is.numeric(lum) || length(lum) != 2L || any(lum < 0 | lum > 100)) {
    stop(sprintf("`%s` must be %s.", "lum", "two numbers between 0 and 100"), call. = FALSE)
  }
  if (!is.numeric(chroma) || length(chroma) != 2L || any(chroma < 0)) {
    stop(sprintf("`%s` must be %s.", "chroma", "two non-negative numbers"), call. = FALSE)
  }

  # one colour sits halfway along the ramp
  if (n == 1) {
    lumSeq = mean(lum)
    chromaSeq = mean(chroma)
  } else {
    lumSeq = seq(lum[1], lum[2], length.out = n)
    chromaSeq = seq(chroma[1], chroma[2], length.out = n)
  }

  cols = grDevices::hcl(h = hue, c = chromaSeq, l = lumSeq)
  finishPalette(cols, direction, alpha, "sequential")
}

#' Diverging palette
#'
#' The low half uses `hue`, the high half the opposite hue; both halves run
#' from the outer `lum`/`chroma` values at the ends to the inner ones at the
#' centre.
#'
#' @inheritParams tintSequential
#' @param lum Luminance at the centre and at the ends, between 0 and 100.
#' @param chroma Chroma at the centre and at the ends, non-negative.
#' @return A `tint_palette` character vector of length `n`.
#' @family tint palettes
#' @export
tintDiverging = function(n, hue = 240, lum = c(92, 25), chroma = c(15, 60), direction = 1,
                         alpha = 1) {
  checkPaletteArgs(n, hue, direction, alpha)

  # ramp-specific arguments
  if (!is.numeric(lum) || length(lum) != 2L || any(lum < 0 | lum > 100)) {
    stop(sprintf("`%s` must be %s.", "lum", "two numbers between 0 and 100"), call. = FALSE)
  }
  if (!is.numeric(chroma) || length(chroma) != 2L || any(chroma < 0)) {
    stop(sprintf("`%s` must be %s.", "chroma", "two non-negative numbers"), call. = FALSE)
  }

  # distance from the centre drives luminance and chroma
  pos = if (n == 1) 0 else seq(-1, 1, length.out = n)
  hues = ifelse(pos < 0, hue, (hue + 180) %% 360)
  lumSeq = lum[1] + abs(pos) * (lum[2] - lum[1])
  chromaSeq = chroma[1] + abs(pos) * (chroma[2] - chroma[1])

  cols = grDevices::hcl(h = hues, c = chromaSeq, l = lumSeq)
  finishPalette(cols, direction, alpha, "diverging")
}

#' Qualitative palette
#'
#' Hues are spaced evenly around the colour wheel starting at `hue`, at
#' constant luminance and chroma.
#'
#' @inheritParams tintSequential
#' @param lum Luminance of every colour, between 0 and 100.
#' @param chroma Chroma of every colour, non-negative.
#' @return A `tint_palette` character vector of length `n`.
#' @family tint palettes
#' @export
tintQualitative = function(n, hue = 240, lum = 60, chroma = 60, direction = 1, alpha = 1) {
  checkPaletteArgs(n, hue, direction, alpha)

  # ramp-specific arguments
  if (!is.numeric(lum) || length(lum) != 1L || any(lum < 0 | lum > 100)) {
    stop(sprintf("`%s` must be %s.", "lum", "a single number between 0 and 100"), call. = FALSE)
  }
  if (!is.numeric(chroma) || length(chroma) != 1L || any(chroma < 0)) {
    stop(sprintf("`%s` must be %s.", "chroma", "a single non-negative number"), call. = FALSE)
  }

  # leave out the end point so the first and last hue differ
  hues = (hue + seq(0, 360, length.out = n + 1L)[seq_len(n)]) %% 360
  lumSeq = rep(lum, n)
  chromaSeq = rep(chroma, n)

  cols = grDevices::hcl(h = hues, c = chromaSeq, l = lumSeq)
  finishPalette(cols, direction, alpha, "qualitative")
}

#' Palette by type
#'
#' @param type Palette type.
#' @param n Number of colours.
#' @param ... Passed to the palette function for `type`.
#' @return A `tint_palette` character vector of length `n`.
#' @export
tintPalette = function(type = c("sequential", "diverging", "qualitative"), n, ...) {
  type = match.arg(type)
  switch(type,
    sequential = tintSequential(n, ...),
    diverging = tintDiverging(n, ...),
    qualitative = tintQualitative(n, ...)
  )
}

#' Tabulate several palettes
#'
#' Builds every palette in `types` with `n` colours and stacks them into one
#' long table, for documentation and swatch plots.
#'
#' @param types Palette types passed to [tintPalette()].
#' @param n Number of colours per palette.
#' @param hue Hue passed to every palette.
#' @return A data frame with columns `type`, `index` and `colour`.
#' @export
paletteTable = function(types = c("sequential", "diverging", "qualitative"), n = 5L,
                        hue = 240) {
  rows = vector("list", length(types))
  for (i in seq_along(types)) {
    cols = tintPalette(types[[i]], n = n, hue = hue)
    rows[[i]] = data.frame(
      type = rep(types[[i]], n),
      index = seq_len(n),
      colour = unclass(cols),
      stringsAsFactors = FALSE
    )
  }
  out = do.call(rbind, rows)
  rownames(out) = NULL
  out
}
