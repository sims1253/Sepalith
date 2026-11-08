#' Convert hex colours to RGB
#'
#' @description
#' Accepts `#RGB`, `#RRGGBB` and `#RRGGBBAA` forms. Short forms are expanded
#' by doubling each digit; a missing alpha channel means fully opaque.
#'
#' @param hex Character vector of hex colours.
#' @return A numeric matrix with columns `red`, `green`, `blue` and `alpha`
#'   on the 0-255 scale, one row per colour.
#' @export
hexToRgb = function(hex) {
  ok = grepl("^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$", hex)
  if (!all(ok)) {
    stop(sprintf("Invalid hex colour(s): %s.", paste(hex[!ok], collapse = ", ")), call. = FALSE)
  }

  # normalise to eight digits: expand #RGB, append opaque alpha
  digits = substring(hex, 2)
  short = nchar(digits) == 3L
  digits[short] = gsub("(.)", "\\1\\1", digits[short])
  opaque = nchar(digits) == 6L
  digits[opaque] = paste0(digits[opaque], "FF")

  # one byte per channel
  red = strtoi(substr(digits, 1, 2), base = 16L)
  green = strtoi(substr(digits, 3, 4), base = 16L)
  blue = strtoi(substr(digits, 5, 6), base = 16L)
  alpha = strtoi(substr(digits, 7, 8), base = 16L)
  cbind(red = red, green = green, blue = blue, alpha = alpha)
}

#' Convert RGB to hex colours
#'
#' @description
#' The inverse of [hexToRgb()]. Values are rounded and clipped to 0-255; an
#' `alpha` column is written only where it is below 255.
#'
#' @param rgb A numeric matrix with columns `red`, `green`, `blue` and
#'   optionally `alpha` on the 0-255 scale, one row per colour.
#' @return A character vector of `#RRGGBB` or `#RRGGBBAA` colours.
#' @export
rgbToHex = function(rgb) {
  needed = c("red", "green", "blue")
  if (!is.matrix(rgb) || !all(needed %in% colnames(rgb))) {
    stop("`rgb` must be a matrix with columns red, green, blue.", call. = FALSE)
  }
  if (!"alpha" %in% colnames(rgb)) {
    rgb = cbind(rgb, alpha = 255)
  }

  # round and clip each channel
  red = as.integer(pmin(pmax(round(rgb[, "red"]), 0), 255))
  green = as.integer(pmin(pmax(round(rgb[, "green"]), 0), 255))
  blue = as.integer(pmin(pmax(round(rgb[, "blue"]), 0), 255))
  alpha = as.integer(pmin(pmax(round(rgb[, "alpha"]), 0), 255))

  # append the alpha byte only where the colour is translucent
  out = sprintf("#%02X%02X%02X", red, green, blue)
  translucent = alpha < 255L
  out[translucent] = paste0(out[translucent], sprintf("%02X", alpha[translucent]))
  unname(out)
}

#' Mix two sets of colours
#'
#' @param a,b Hex colours, recycled to a common length.
#' @param weight Share of `b` in the mix, between 0 and 1.
#' @return A character vector of hex colours.
#' @export
mixColours = function(a, b, weight = 0.5) {
  if (!is.numeric(weight) || any(weight < 0 | weight > 1)) {
    stop("`weight` must be between 0 and 1.", call. = FALSE)
  }
  n = max(length(a), length(b))
  rgbA = hexToRgb(rep_len(a, n))
  rgbB = hexToRgb(rep_len(b, n))
  rgbToHex(rgbA * (1 - weight) + rgbB * weight)
}

#' Lighten colours
#'
#' Mixes each colour with white.
#'
#' @param col Hex colours.
#' @param amount Share of white in the mix, between 0 and 1.
#' @return A character vector of hex colours.
#' @family shade helpers
#' @export
lighten = function(col, amount = 0.2) {
  if (!is.numeric(amount) || length(amount) != 1L || amount < 0 || amount > 1) {
    stop("`amount` must be a single number between 0 and 1.", call. = FALSE)
  }
  mixColours(col, "#FFFFFF", weight = amount)
}

#' Darken colours
#'
#' Mixes each colour with black.
#'
#' @param col Hex colours.
#' @param amount Share of black in the mix, between 0 and 1.
#' @return A character vector of hex colours.
#' @family shade helpers
#' @export
darken = function(col, amount = 0.2) {
  if (!is.numeric(amount) || length(amount) != 1L || amount < 0 || amount > 1) {
    stop("`amount` must be a single number between 0 and 1.", call. = FALSE)
  }
  mixColours(col, "#000000", weight = amount)
}

#' Adjust single colour channels
#'
#' @description
#' Scales the `red`, `green` and `blue` channels independently. Each factor
#' multiplies its channel; results are clipped to 0-255 by [rgbToHex()].
#'
#' @param col Hex colours.
#' @param red,green,blue Non-negative multipliers for each channel.
#' @return A character vector of hex colours.
#' @export
adjustChannels = function(col, red = 1, green = 1, blue = 1) {
  rgb = hexToRgb(col)

  # red channel
  if (!is.numeric(red) || length(red) != 1L || red < 0) {
    stop(sprintf("`%s` must be a single non-negative number.", "red"), call. = FALSE)
  }
  rgb[, "red"] = rgb[, "red"] * red

  # green channel
  if (!is.numeric(green) || length(green) != 1L || green < 0) {
    stop(sprintf("`%s` must be a single non-negative number.", "green"), call. = FALSE)
  }
  rgb[, "green"] = rgb[, "green"] * green

  # blue channel
  if (!is.numeric(blue) || length(blue) != 1L || blue < 0) {
    stop(sprintf("`%s` must be a single non-negative number.", "blue"), call. = FALSE)
  }
  rgb[, "blue"] = rgb[, "blue"] * blue

  rgbToHex(rgb)
}

#' Relative luminance of colours
#'
#' Uses the WCAG definition: linearised sRGB channels weighted by
#' `weights`.
#'
#' @param col Hex colours.
#' @param weights Channel weights for `red`, `green` and `blue`.
#' @return A numeric vector between 0 and 1.
#' @export
relativeLuminance = function(col, weights = c(red = 0.2126, green = 0.7152, blue = 0.0722)) {
  rgb = hexToRgb(col) / 255

  # undo the sRGB gamma curve
  linear = ifelse(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055)^2.4)
  dim(linear) = dim(rgb)
  colnames(linear) = colnames(rgb)
  out = linear[, "red"] * weights[["red"]] +
    linear[, "green"] * weights[["green"]] +
    linear[, "blue"] * weights[["blue"]]
  unname(out)
}

#' Check text colours against contrast levels
#'
#' @description
#' Computes the contrast ratio `(lighter + 0.05) / (darker + 0.05)` of
#' the relative luminances for every pair and compares it with the WCAG
#' thresholds in `levels`: `AA_large` (large text), `AA` and `AAA`.
#'
#' @param foreground,background Hex colours, recycled to a common length.
#' @param levels Named numeric vector of minimum ratios.
#' @return A data frame with columns `foreground`, `background`, `ratio`,
#'   `AA_large`, `AA` and `AAA`.
#' @export
contrastReport = function(foreground, background = "#FFFFFF",
                          levels = c(AA_large = 3, AA = 4.5, AAA = 7)) {
  needed = c("AA_large", "AA", "AAA")
  if (!is.numeric(levels) || !all(needed %in% names(levels))) {
    stop("`levels` must be a numeric vector named AA_large, AA, AAA.", call. = FALSE)
  }
  n = max(length(foreground), length(background))
  foreground = rep_len(foreground, n)
  background = rep_len(background, n)

  # ratio of the lighter to the darker luminance
  lumFg = relativeLuminance(foreground)
  lumBg = relativeLuminance(background)
  lighter = pmax(lumFg, lumBg)
  darker = pmin(lumFg, lumBg)
  ratio = (lighter + 0.05) / (darker + 0.05)

  out = data.frame(
    foreground = foreground,
    background = background,
    ratio = round(ratio, 2),
    stringsAsFactors = FALSE
  )

  # one pass/fail column per level
  out$AA_large = ratio >= levels[["AA_large"]]
  out$AA = ratio >= levels[["AA"]]
  out$AAA = ratio >= levels[["AAA"]]
  out
}
