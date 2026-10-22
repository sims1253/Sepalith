#' Format numbers compactly
#'
#' Turns large numbers into short labels with a K, M or B suffix, keeping
#' small numbers as they are.
#'
#' @param x Numeric vector of values to label.
#' @param digits Number of significant digits to keep.
#' @return A character vector the same length as `x`.
#' @keywords internal
label_compact <- function(x, digits = 3) {
  out <- character(length(x))
  ax <- abs(x)
  for (i in seq_along(x)) {
    if (is.na(x[i])) {
      out[i] <- NA_character_
    } else if (ax[i] >= 1e9) {
      out[i] <- paste0(signif(x[i] / 1e9, digits), "B")
    } else if (ax[i] >= 1e6) {
      out[i] <- paste0(signif(x[i] / 1e6, digits), "M")
    } else if (ax[i] >= 1e3) {
      out[i] <- paste0(signif(x[i] / 1e3, digits), "K")
    } else {
      out[i] <- as.character(signif(x[i], digits))
    }
  }
  out
}

#' Format proportions as percentages
#'
#' @param x Numeric vector of proportions, where 1 means 100 percent.
#' @param accuracy Number of decimal places to show.
#' @return A character vector the same length as `x`.
#' @keywords internal
label_percent_simple <- function(x, accuracy = 0) {
  out <- formatC(100 * x, format = "f", digits = accuracy)
  out[is.na(x)] <- NA_character_
  paste0(out, "%")
}

#' Percentage scale for the x axis
#'
#' Continuous x scale that labels proportions as percentages and does not
#' pad the axis below zero.
#'
#' @param accuracy Number of decimal places to show.
#' @param ... Passed to [ggplot2::scale_x_continuous()].
#' @return A ggplot2 scale.
#' @family scales
#' @export
scale_x_percent <- function(accuracy = 0, ...) {
  ggplot2::scale_x_continuous(
    labels = function(x) label_percent_simple(x, accuracy = accuracy),
    expand = ggplot2::expansion(mult = c(0, 0.05)),
    ...
  )
}

#' Percentage scale for the y axis
#'
#' Continuous y scale that labels proportions as percentages and does not
#' pad the axis below zero.
#'
#' @param accuracy Number of decimal places to show.
#' @param ... Passed to [ggplot2::scale_y_continuous()].
#' @return A ggplot2 scale.
#' @family scales
#' @export
scale_y_percent <- function(accuracy = 0, ...) {
  ggplot2::scale_y_continuous(
    labels = function(x) label_percent_simple(x, accuracy = accuracy),
    expand = ggplot2::expansion(mult = c(0, 0.05)),
    ...
  )
}

#' Compact number scale for the x axis
#'
#' Continuous x scale that labels thousands, millions and billions with a
#' short suffix.
#'
#' @param digits Number of significant digits to keep.
#' @param ... Passed to [ggplot2::scale_x_continuous()].
#' @return A ggplot2 scale.
#' @family scales
#' @export
scale_x_compact <- function(digits = 3, ...) {
  ggplot2::scale_x_continuous(
    labels = function(x) label_compact(x, digits = digits),
    ...
  )
}

#' Compact number scale for the y axis
#'
#' Continuous y scale that labels thousands, millions and billions with a
#' short suffix.
#'
#' @param digits Number of significant digits to keep.
#' @param ... Passed to [ggplot2::scale_y_continuous()].
#' @return A ggplot2 scale.
#' @family scales
#' @export
scale_y_compact <- function(digits = 3, ...) {
  ggplot2::scale_y_continuous(
    labels = function(x) label_compact(x, digits = digits),
    ...
  )
}

#' House qualitative colours
#'
#' @return A character vector of hex colours.
#' @keywords internal
plotsmith_colours <- function() {
  c("#3b6e8f", "#d1793a", "#5b9a68", "#b8484f", "#8a6fae", "#a38b5a", "#5f6b75")
}

#' Discrete house colour scale
#'
#' Maps a discrete variable to the plotsmith qualitative palette, failing
#' loudly when there are more levels than palette colours.
#'
#' @param reverse Whether to use the palette in reverse order.
#' @param ... Passed to [ggplot2::discrete_scale()].
#' @return A ggplot2 scale.
#' @family scales
#' @export
scale_colour_plotsmith <- function(reverse = FALSE, ...) {
  pal <- plotsmith_colours()
  if (reverse) {
    pal <- rev(pal)
  }
  ggplot2::discrete_scale(
    aesthetics = "colour",
    palette = function(n) {
      if (n > length(pal)) {
        stop("The plotsmith palette has only ", length(pal), " colours.", call. = FALSE)
      }
      pal[seq_len(n)]
    },
    ...
  )
}

#' Discrete house fill scale
#'
#' Maps a discrete variable to the plotsmith qualitative palette, failing
#' loudly when there are more levels than palette colours.
#'
#' @param reverse Whether to use the palette in reverse order.
#' @param ... Passed to [ggplot2::discrete_scale()].
#' @return A ggplot2 scale.
#' @family scales
#' @export
scale_fill_plotsmith <- function(reverse = FALSE, ...) {
  pal <- plotsmith_colours()
  if (reverse) {
    pal <- rev(pal)
  }
  ggplot2::discrete_scale(
    aesthetics = "fill",
    palette = function(n) {
      if (n > length(pal)) {
        stop("The plotsmith palette has only ", length(pal), " colours.", call. = FALSE)
      }
      pal[seq_len(n)]
    },
    ...
  )
}
