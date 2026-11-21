#' Zoning rules
#'
#' One row per zoning prefix. A code is a prefix optionally followed by a
#' density digit, so `"R2"` is residential at density level 2.
#'
#' @return A data frame with columns `prefix`, `category`, `far`
#'   (floor-area ratio at density 1), `height_m`, `min_lot_m2`,
#'   `units_per_ha` (dwellings per hectare at density 1) and
#'   `parking_per_100m2` (parking spaces per 100 m2 of floor area).
#' @noRd
zoning_rules <- function() {
  return(data.frame(
    prefix = c("R", "C", "I", "A", "MX"),
    category = c("residential", "commercial", "industrial", "agricultural", "mixed"),
    far = c(0.5, 1.5, 1, 0.05, 2),
    height_m = c(10, 25, 20, 12, 30),
    min_lot_m2 = c(300, 500, 2000, 10000, 400),
    units_per_ha = c(25, 0, 0, 1, 60),
    parking_per_100m2 = c(1, 2.5, 0.5, 0, 1.5),
    stringsAsFactors = FALSE
  ))
}

#' Validate a ZoningCode
#'
#' @param object A `ZoningCode` object.
#' @return `TRUE`, or a character vector describing every problem found.
#' @noRd
valid_zoning_code <- function(object) {
  pattern <- "^([A-Z]{1,2})([1-9]?)$"
  ok <- grepl(pattern, object@code)
  if (!all(ok)) {
    return(sprintf("Malformed zoning code(s): %s.", toString(object@code[!ok])))
  }
  prefix <- sub(pattern, "\\1", object@code)
  unknown <- setdiff(prefix, zoning_rules()$prefix)
  if (length(unknown) > 0L) {
    return(sprintf("Unknown zoning prefix(es): %s.", toString(unknown)))
  }
  return(TRUE)
}

#' Zoning codes
#'
#' @slot code Normalised zoning codes such as `"R2"` or `"MX"`.
#' @export
setClass("ZoningCode", representation(code = "character"), validity = valid_zoning_code)

#' Create zoning codes
#'
#' Codes are upper-cased and stripped of spaces and dashes, so `"r-2"`
#' becomes `"R2"`. Legacy prefixes from older plans are translated:
#' `RES` to `R`, `COM` to `C`, `IND` to `I`, `AG` to `A` and `MU` to `MX`.
#'
#' @param code A character vector of zoning codes.
#' @return A `ZoningCode` object.
#' @export
zoning_code <- function(code) {
  if (!is.character(code) && !is.factor(code)) {
    stop("`code` must be a character vector.", call. = FALSE)
  }
  code <- toupper(gsub("[[:space:]-]", "", as.character(code)))

  # Legacy prefixes
  code <- sub("^RES", "R", code)
  code <- sub("^COM", "C", code)
  code <- sub("^IND", "I", code)
  code <- sub("^AG", "A", code)
  code <- sub("^MU", "MX", code)

  return(new("ZoningCode", code = code))
}

#' Look up zoning rules for codes
#'
#' @param x A `ZoningCode` object.
#' @return A data frame with columns `code`, `prefix`, `density` and every
#'   column of [zoning_rules()] except `prefix`, one row per code.
#' @export
zoning_lookup <- function(x) {
  pattern <- "^([A-Z]{1,2})([1-9]?)$"
  prefix <- sub(pattern, "\\1", x@code)
  density <- suppressWarnings(as.integer(sub(pattern, "\\2", x@code)))
  density[is.na(density)] <- 1L

  rules <- zoning_rules()
  idx <- match(prefix, rules$prefix)
  out <- data.frame(code = x@code, prefix = prefix, density = density, stringsAsFactors = FALSE)
  out$category <- rules$category[idx]
  out$far <- rules$far[idx]
  out$height_m <- rules$height_m[idx]
  out$min_lot_m2 <- rules$min_lot_m2[idx]
  out$units_per_ha <- rules$units_per_ha[idx]
  out$parking_per_100m2 <- rules$parking_per_100m2[idx]
  return(out)
}

#' Building limits for parcels under a zoning code
#'
#' The floor-area ratio, the height limit and the dwelling density all grow
#' linearly with the density digit of the code.
#'
#' @param x A `ZoningCode` object.
#' @param area_m2 Parcel areas in square metres, one per code.
#' @return A data frame with columns `code`, `max_floor_m2`, `max_height_m`,
#'   `max_units`, `parking` and `buildable`.
#' @export
zoning_limits <- function(x, area_m2) {
  if (!is.numeric(area_m2)) {
    stop("`area_m2` must be numeric.", call. = FALSE)
  }
  if (length(area_m2) != length(x@code)) {
    stop("`area_m2` must have one value per zoning code.", call. = FALSE)
  }
  if (any(area_m2 <= 0)) {
    stop("`area_m2` must be positive.", call. = FALSE)
  }
  info <- zoning_lookup(x)

  # Every limit scales with the density digit
  max_floor_m2 <- area_m2 * info$far * info$density
  max_height_m <- info$height_m * info$density
  max_units <- floor(area_m2 / 10000 * info$units_per_ha * info$density)
  parking <- ceiling(max_floor_m2 / 100 * info$parking_per_100m2)
  buildable <- area_m2 >= info$min_lot_m2

  # Undersized lots keep their limits for reference but cannot be built on
  max_units[!buildable] <- 0
  parking[!buildable] <- 0

  return(data.frame(
    code = info$code,
    max_floor_m2 = max_floor_m2,
    max_height_m = max_height_m,
    max_units = max_units,
    parking = parking,
    buildable = buildable,
    stringsAsFactors = FALSE
  ))
}

#' Convert zoning codes to base types
#'
#' @param x A `ZoningCode` object.
#' @param to Target type: the codes as `"character"`, a `"factor"` of
#'   categories, the `"integer"` density levels, a `"list"` with one
#'   element per category, or the rule table as a `"data.frame"`.
#' @return An object of the requested type.
#' @export
convert_zoning <- function(x, to = c("character", "factor", "integer", "list", "data.frame")) {
  to <- match.arg(to)
  info <- zoning_lookup(x)
  switch(to,
    character = {
      out <- info$code
    },
    factor = {
      out <- factor(info$category, levels = zoning_rules()$category)
    },
    integer = {
      out <- info$density
      names(out) <- info$code
    },
    list = {
      out <- split(info$code, factor(info$category, levels = zoning_rules()$category))
      out <- out[lengths(out) > 0L]
    },
    data.frame = {
      out <- info
      rownames(out) <- NULL
    }
  )
  return(out)
}

setAs("ZoningCode", "character", function(from) convert_zoning(from, "character"))
setAs("ZoningCode", "factor", function(from) convert_zoning(from, "factor"))
setAs("ZoningCode", "integer", function(from) convert_zoning(from, "integer"))
setAs("ZoningCode", "list", function(from) convert_zoning(from, "list"))
setAs("ZoningCode", "data.frame", function(from) convert_zoning(from, "data.frame"))
setAs("character", "ZoningCode", function(from) zoning_code(from))

#' Print zoning codes
#'
#' @param object A `ZoningCode` object.
#' @return `object`, invisibly.
#' @noRd
show_zoning <- function(object) {
  info <- zoning_lookup(object)
  counts <- table(factor(info$category, levels = zoning_rules()$category))
  counts <- counts[counts > 0L]
  cat(sprintf("<ZoningCode> %d codes\n", nrow(info)))
  for (nm in names(counts)) {
    cat(sprintf("  %-13s %d\n", nm, counts[[nm]]))
  }
  return(invisible(object))
}

#' @export
setMethod("show", "ZoningCode", function(object) show_zoning(object))
