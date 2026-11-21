#' Regular expressions for the supported coordinate text formats
#'
#' @format A named character vector with elements `decimal`, `dms`, `dm`
#'   and `hemisphere`.
#' @keywords internal
coord.patterns <- c(
  decimal = "^([-+]?[0-9]*\\.?[0-9]+)$",
  dms = "^([0-9]+)D ([0-9]+)M ([0-9.]+)S ?([NSEW]?)$",
  dm = "^([0-9]+)D ([0-9.]+)M ?([NSEW]?)$",
  hemisphere = "^([0-9.]+) ?([NSEW])$"
)

#' @title Normalise coordinate text
#' @description Rewrites the many ways of writing degrees, minutes and
#'   seconds into one canonical spelling that [coord.patterns] matches.
#' @param x A character vector.
#' @keywords internal
#' @return A character vector of the same length.
coord.clean <- function(x) {
  x <- as.character(x)
  # 1. trim and upper-case
  x <- toupper(trimws(x))

  # 2. hemisphere words
  x <- gsub("NORTH", "N", x, fixed = TRUE)
  x <- gsub("SOUTH", "S", x, fixed = TRUE)
  x <- gsub("EAST", "E", x, fixed = TRUE)
  x <- gsub("WEST", "W", x, fixed = TRUE)

  # 3. degree marks
  x <- gsub("([0-9.])\\s*(DEGREES|DEGREE|DEG|D|O|\\^)\\s*", "\\1D ", x)

  # 4. second marks
  x <- gsub("([0-9.])\\s*(SECONDS|SECOND|SEC|\"|'')\\s*", "\\1S ", x)

  # 5. minute marks
  x <- gsub("([0-9.])\\s*(MINUTES|MINUTE|MIN|')\\s*", "\\1M ", x)

  # 6. collapse spaces
  x <- gsub("\\s+", " ", x)
  trimws(x)
}

#' @title Parse coordinate text in a known format
#' @description Converts text such as `"52D 31M 12S N"` to decimal degrees.
#' @param x A character vector, already passed through [coord.clean()].
#' @param format One of `"decimal"`, `"dms"`, `"dm"` or `"hemisphere"`.
#' @return A numeric vector of decimal degrees; `NA` where `x` does not
#'   match `format`.
#' @export
coord.parse <- function(x, format = c("decimal", "dms", "dm", "hemisphere")) {
  format <- match.arg(format)
  grab <- function(pattern, i) {
    out <- sub(pattern, paste0("\\", i), x)
    out[!grepl(pattern, x)] <- NA_character_
    out
  }
  negative <- function(hemi) ifelse(!is.na(hemi) & hemi %in% c("S", "W"), -1, 1)
  switch(format,
    decimal = {
      pattern <- coord.patterns[["decimal"]]
      value <- as.numeric(grab(pattern, 1))
      value
    },
    dms = {
      pattern <- coord.patterns[["dms"]]
      deg <- as.numeric(grab(pattern, 1))
      min <- as.numeric(grab(pattern, 2))
      sec <- as.numeric(grab(pattern, 3))
      value <- deg + min / 60 + sec / 3600
      value * negative(grab(pattern, 4))
    },
    dm = {
      pattern <- coord.patterns[["dm"]]
      deg <- as.numeric(grab(pattern, 1))
      min <- as.numeric(grab(pattern, 2))
      value <- deg + min / 60
      value * negative(grab(pattern, 3))
    },
    hemisphere = {
      pattern <- coord.patterns[["hemisphere"]]
      deg <- as.numeric(grab(pattern, 1))
      value <- deg
      value * negative(grab(pattern, 2))
    }
  )
}

#' @title Detect the format of coordinate text
#' @description Tries each entry of [coord.patterns] in turn.
#' @param x A character vector, already passed through [coord.clean()].
#' @return A character vector of format names, `NA` where nothing matches.
#' @export
coord.detect <- function(x) {
  out <- rep(NA_character_, length(x))

  # Try "decimal"
  hit <- is.na(out) & grepl(coord.patterns[["decimal"]], x)
  out[hit] <- "decimal"

  # Try "dms"
  hit <- is.na(out) & grepl(coord.patterns[["dms"]], x)
  out[hit] <- "dms"

  # Try "dm"
  hit <- is.na(out) & grepl(coord.patterns[["dm"]], x)
  out[hit] <- "dm"

  # Try "hemisphere"
  hit <- is.na(out) & grepl(coord.patterns[["hemisphere"]], x)
  out[hit] <- "hemisphere"
  out
}

#' @title Parse coordinate pairs
#' @description Splits each string at `sep`, detects the format of each half
#'   and parses it.
#' @param x A character vector of pairs such as `"52.5, 13.4"`.
#' @param order Whether pairs are written `"latlon"` or `"lonlat"`.
#' @param sep Regular expression that separates the two halves.
#' @export
#' @return A data frame with columns `lon` and `lat`.
coord.parse.pair <- function(x, order = c("latlon", "lonlat"), sep = "[,;]") {
  order <- match.arg(order)
  halves <- strsplit(x, sep)
  if (any(lengths(halves) != 2)) {
    stop("Every element of `x` must contain exactly one separator.", call. = FALSE)
  }
  values <- matrix(NA_real_, nrow = length(x), ncol = 2)

  # First half
  first <- coord.clean(vapply(halves, `[`, "", 1))
  fmt <- coord.detect(first)
  for (f in unique(fmt[!is.na(fmt)])) {
    values[fmt %in% f, 1] <- coord.parse(first[fmt %in% f], f)
  }

  # Second half
  second <- coord.clean(vapply(halves, `[`, "", 2))
  fmt <- coord.detect(second)
  for (f in unique(fmt[!is.na(fmt)])) {
    values[fmt %in% f, 2] <- coord.parse(second[fmt %in% f], f)
  }
  if (order == "latlon") {
    out <- data.frame(lon = values[, 2], lat = values[, 1])
  } else {
    out <- data.frame(lon = values[, 1], lat = values[, 2])
  }
  out$lon[abs(out$lon) > 180] <- NA
  out$lat[abs(out$lat) > 90] <- NA
  out
}

#' @title Report coordinate strings by format
#' @param x A character vector of raw coordinate text.
#' @return A data frame with one row per format (`decimal`, `dms`, `dm`,
#'   `hemisphere`, then `unparsed`) and columns `format`, `n`, `share` and
#'   `example` (the first raw string of that format).
#' @keywords internal
coord.format.report <- function(x) {
  fmt <- coord.detect(coord.clean(x))
  total <- max(length(x), 1L)
  rows <- list()

  # Format "decimal"
  hit <- which(fmt %in% "decimal")
  rows$decimal <- data.frame(format = "decimal", n = length(hit),
    share = length(hit) / total, example = x[hit][1])

  # Format "dms"
  hit <- which(fmt %in% "dms")
  rows$dms <- data.frame(format = "dms", n = length(hit),
    share = length(hit) / total, example = x[hit][1])

  # Format "dm"
  hit <- which(fmt %in% "dm")
  rows$dm <- data.frame(format = "dm", n = length(hit),
    share = length(hit) / total, example = x[hit][1])

  # Format "hemisphere"
  hit <- which(fmt %in% "hemisphere")
  rows$hemisphere <- data.frame(format = "hemisphere", n = length(hit),
    share = length(hit) / total, example = x[hit][1])

  # Format "unparsed"
  hit <- which(is.na(fmt))
  rows$unparsed <- data.frame(format = "unparsed", n = length(hit),
    share = length(hit) / total, example = x[hit][1])
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}
