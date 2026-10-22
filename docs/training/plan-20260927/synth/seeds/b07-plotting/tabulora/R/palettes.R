#' Named colour palettes
#'
#' Returns the list of palettes shipped with tabulora. Each element is a
#' character vector of hex colours.
#'
#' @return A named list of character vectors.
#' @keywords internal
paletteRegistry <- function() {
  list(
    harbour = c("#1f4e79", "#2e86ab", "#7fb7be", "#d3d5a1", "#f2a541"),
    meadow = c("#2d5016", "#5a8f29", "#a3c45a", "#e8d77a", "#c97b2e"),
    slate = c("#22303c", "#3e5366", "#6b8299", "#a7b7c7", "#dfe6ec"),
    ember = c("#4a1c1c", "#8c2f2f", "#c8553d", "#f28f3b", "#ffd5a3")
  )
}

#' Get colours from a named palette
#'
#' Returns `n` colours from one of the built-in palettes, interpolating
#' when more colours are requested than the palette holds.
#'
#' @param name Name of the palette.
#' @param n Number of colours to return. Defaults to the palette length.
#' @param reverse Whether to reverse the colour order.
#' @return A character vector of hex colours.
#' @family palettes
#' @export
paletteGet <- function(name, n = NULL, reverse = FALSE) {
  registry <- paletteRegistry()
  if (!name %in% names(registry)) {
    stop("Unknown palette '", name, "'. Available: ",
         paste(names(registry), collapse = ", "), call. = FALSE)
  }
  cols <- registry[[name]]
  if (reverse) {
    cols <- rev(cols)
  }
  if (is.null(n)) {
    return(cols)
  }
  if (n <= length(cols)) {
    return(cols[seq_len(n)])
  }
  paletteInterpolate(cols, n)
}

#' Interpolate between colours
#'
#' Creates `n` evenly spaced colours along the path through `colours` in
#' Lab space.
#'
#' @param colours Character vector of colours.
#' @param n Number of colours to return.
#' @return A character vector of hex colours.
#' @family palettes
#' @export
paletteInterpolate <- function(colours, n) {
  if (length(colours) < 2L) {
    stop("Need at least two colours to interpolate.", call. = FALSE)
  }
  ramp <- grDevices::colorRampPalette(colours, space = "Lab")
  ramp(n)
}

#' Build a diverging palette
#'
#' Creates an odd-length or even-length palette that runs from `low`
#' through `mid` to `high`, with the midpoint colour at the centre.
#'
#' @param n Number of colours to return.
#' @param low Colour at the low end of the scale.
#' @param mid Colour at the centre of the scale.
#' @param high Colour at the high end of the scale.
#' @return A character vector of hex colours.
#' @family palettes
#' @export
paletteDiverging <- function(n, low = "#2e86ab", mid = "#f7f7f7", high = "#c8553d") {
  if (n < 3) {
    stop("A diverging palette needs at least three colours.", call. = FALSE)
  }
  toMid <- grDevices::colorRampPalette(c(low, mid), space = "Lab")
  fromMid <- grDevices::colorRampPalette(c(mid, high), space = "Lab")
  if (n %% 2 == 1) {
    k <- (n + 1) / 2
    cols <- c(toMid(k), fromMid(k)[-1])
  } else {
    k <- n / 2
    cols <- c(toMid(k + 1)[seq_len(k)], fromMid(k + 1)[-1])
  }
  cols
}

#' Lighten a colour
#'
#' Mixes each colour with white by the given amount.
#'
#' @param colour Character vector of colours.
#' @param amount Proportion of white to mix in, between 0 and 1.
#' @return A character vector of hex colours.
#' @family colourTools
#' @export
lightenColour <- function(colour, amount = 0.3) {
  stopifnot(amount >= 0, amount <= 1)
  rgb <- grDevices::col2rgb(colour)
  mixed <- rgb + (255 - rgb) * amount
  grDevices::rgb(mixed[1, ], mixed[2, ], mixed[3, ], maxColorValue = 255)
}

#' Darken a colour
#'
#' Mixes each colour with black by the given amount.
#'
#' @param colour Character vector of colours.
#' @param amount Proportion of black to mix in, between 0 and 1.
#' @return A character vector of hex colours.
#' @family colourTools
#' @export
darkenColour <- function(colour, amount = 0.3) {
  stopifnot(amount >= 0, amount <= 1)
  rgb <- grDevices::col2rgb(colour)
  mixed <- rgb * (1 - amount)
  grDevices::rgb(mixed[1, ], mixed[2, ], mixed[3, ], maxColorValue = 255)
}

#' Pick a readable text colour for a background
#'
#' Uses relative luminance to choose between a dark and a light text colour
#' for each background colour.
#'
#' @param colour Character vector of background colours.
#' @param dark Text colour used on light backgrounds.
#' @param light Text colour used on dark backgrounds.
#' @return A character vector the same length as `colour`.
#' @family colourTools
#' @export
contrastText <- function(colour, dark = "#111111", light = "#ffffff") {
  rgb <- grDevices::col2rgb(colour) / 255
  linear <- ifelse(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055)^2.4)
  luminance <- 0.2126 * linear[1, ] + 0.7152 * linear[2, ] + 0.0722 * linear[3, ]
  ifelse(luminance > 0.179, dark, light)
}

#' Show palette colours as swatches
#'
#' Draws each colour as a filled rectangle labelled with its hex code, using
#' base graphics. Graphical parameters are restored on exit.
#'
#' @param colours Character vector of colours.
#' @param title Optional title drawn above the swatches.
#' @return `colours`, invisibly.
#' @family palettes
#' @export
paletteShow <- function(colours, title = NULL) {
  n <- length(colours)
  oldPar <- graphics::par(mar = c(1, 1, if (is.null(title)) 1 else 3, 1))
  on.exit(graphics::par(oldPar), add = TRUE)
  graphics::plot.new()
  graphics::plot.window(xlim = c(0, n), ylim = c(0, 1))
  graphics::rect(seq_len(n) - 1, 0, seq_len(n), 1, col = colours, border = "white")
  graphics::text(seq_len(n) - 0.5, 0.5, labels = colours,
                 col = contrastText(colours), cex = 0.8)
  if (!is.null(title)) {
    graphics::title(main = title)
  }
  invisible(colours)
}
