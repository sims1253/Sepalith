#' Validate a Parcel table
#'
#' All slots must have the same length as `parcel_id`. The slots
#' `parcel_id`, `area_m2`, `zone`, `owner` and `assessed_value` are then
#' checked in turn.
#'
#' @param object A `Parcel` object.
#' @return `TRUE`, or a character vector describing every problem found.
#' @noRd
valid_parcel <- function(object) {
  fields <- c("parcel_id", "area_m2", "zone", "owner", "assessed_value")
  lens <- vapply(fields, function(f) length(slot(object, f)), integer(1))
  if (any(lens != lens[["parcel_id"]])) {
    return(sprintf("`%s` must have one value per parcel.", fields[lens != lens[["parcel_id"]]]))
  }
  problems <- character()

  if (anyNA(object@parcel_id) || anyDuplicated(object@parcel_id) > 0L) {
    problems <- c(problems, "`parcel_id` must be unique and not missing.")
  }
  if (anyNA(object@area_m2) || any(object@area_m2 <= 0)) {
    problems <- c(problems, "`area_m2` must be positive and not missing.")
  }
  if (anyNA(object@zone) || !all(grepl("^[A-Z]{1,2}[0-9]?$", object@zone))) {
    problems <- c(problems, "`zone` must be a zoning code such as \"R2\" or \"MX\".")
  }
  if (anyNA(object@owner)) {
    problems <- c(problems, "`owner` must not be missing.")
  }
  if (any(object@assessed_value < 0, na.rm = TRUE)) {
    problems <- c(problems, "`assessed_value` must not be negative.")
  }

  if (length(problems) == 0L) {
    return(TRUE)
  }
  return(problems)
}

#' A table of land parcels
#'
#' @slot parcel_id Unique parcel identifiers.
#' @slot area_m2 Parcel areas in square metres.
#' @slot zone Zoning codes.
#' @slot owner Registered owners.
#' @slot assessed_value Assessed values; `NA` when not yet assessed.
#' @export
setClass(
  "Parcel",
  representation(
    parcel_id = "character",
    area_m2 = "numeric",
    zone = "character",
    owner = "character",
    assessed_value = "numeric"
  ),
  validity = valid_parcel
)

#' Create a Parcel table
#'
#' @param parcel_id Unique parcel identifiers.
#' @param area_m2 Parcel areas in square metres.
#' @param zone Zoning codes; recycled to the number of parcels.
#' @param owner Registered owners; recycled to the number of parcels.
#' @param assessed_value Assessed values; recycled to the number of parcels.
#' @return A `Parcel` object.
#' @family parcel
#' @export
parcel <- function(parcel_id, area_m2, zone, owner = "unknown", assessed_value = NA_real_) {
  n <- length(parcel_id)
  if (length(area_m2) != n) {
    stop(sprintf("`%s` must have length %d.", "area_m2", n), call. = FALSE)
  }
  if (!length(zone) %in% c(1L, n)) {
    stop(sprintf("`%s` must have length 1 or %d.", "zone", n), call. = FALSE)
  }
  if (!length(owner) %in% c(1L, n)) {
    stop(sprintf("`%s` must have length 1 or %d.", "owner", n), call. = FALSE)
  }
  if (!length(assessed_value) %in% c(1L, n)) {
    stop(sprintf("`%s` must have length 1 or %d.", "assessed_value", n), call. = FALSE)
  }

  return(new(
    "Parcel",
    parcel_id = as.character(parcel_id),
    area_m2 = as.numeric(area_m2),
    zone = toupper(rep_len(as.character(zone), n)),
    owner = rep_len(as.character(owner), n),
    assessed_value = rep_len(as.numeric(assessed_value), n)
  ))
}

#' Subset and combine Parcel tables
#'
#' `parcel_subset()` keeps the parcels selected by `i`; `parcel_combine()`
#' appends the parcels of `y` to those of `x`.
#'
#' @param x,y `Parcel` objects.
#' @param i A logical, integer or character index; character values are
#'   matched against `parcel_id`.
#' @return A `Parcel` object.
#' @family parcel
#' @export
parcel_subset <- function(x, i) {
  if (is.character(i)) {
    i <- match(i, x@parcel_id)
    if (anyNA(i)) {
      stop("Unknown `parcel_id` in index.", call. = FALSE)
    }
  }
  x@parcel_id <- x@parcel_id[i]
  x@area_m2 <- x@area_m2[i]
  x@zone <- x@zone[i]
  x@owner <- x@owner[i]
  x@assessed_value <- x@assessed_value[i]
  validObject(x)
  return(x)
}

#' @rdname parcel_subset
#' @export
parcel_combine <- function(x, y) {
  dup <- intersect(x@parcel_id, y@parcel_id)
  if (length(dup) > 0L) {
    stop("Duplicated `parcel_id`: ", toString(dup), call. = FALSE)
  }
  x@parcel_id <- c(x@parcel_id, y@parcel_id)
  x@area_m2 <- c(x@area_m2, y@area_m2)
  x@zone <- c(x@zone, y@zone)
  x@owner <- c(x@owner, y@owner)
  x@assessed_value <- c(x@assessed_value, y@assessed_value)
  validObject(x)
  return(x)
}

#' @export
setMethod("length", "Parcel", function(x) length(x@parcel_id))

#' @export
setMethod("[", "Parcel", function(x, i, ...) parcel_subset(x, i))

#' Convert between Parcel tables and data frames
#'
#' @param from The object to convert.
#' @return A data frame with one column per slot, or a `Parcel` object.
#' @noRd
parcel_to_frame <- function(from) {
  return(data.frame(
    parcel_id = from@parcel_id,
    area_m2 = from@area_m2,
    zone = from@zone,
    owner = from@owner,
    assessed_value = from@assessed_value,
    stringsAsFactors = FALSE
  ))
}

#' @rdname parcel_to_frame
#' @noRd
frame_to_parcel <- function(from) {
  n <- nrow(from)

  # Column `parcel_id`: required, stored as text
  if (!"parcel_id" %in% names(from)) {
    stop(sprintf("Data frame lacks column `%s`.", "parcel_id"), call. = FALSE)
  }
  parcel_id <- as.character(from[["parcel_id"]])

  # Column `area_m2`: required, numeric
  if (!"area_m2" %in% names(from)) {
    stop(sprintf("Data frame lacks column `%s`.", "area_m2"), call. = FALSE)
  }
  area_m2 <- as.numeric(from[["area_m2"]])

  # Column `zone`: required, upper case
  if (!"zone" %in% names(from)) {
    stop(sprintf("Data frame lacks column `%s`.", "zone"), call. = FALSE)
  }
  zone <- toupper(as.character(from[["zone"]]))

  # Column `owner`: optional, defaults to "unknown"
  if (!"owner" %in% names(from)) {
    owner <- rep("unknown", n)
  } else {
    owner <- as.character(from[["owner"]])
  }

  # Column `assessed_value`: optional, defaults to NA
  if (!"assessed_value" %in% names(from)) {
    assessed_value <- rep(NA_real_, n)
  } else {
    assessed_value <- as.numeric(from[["assessed_value"]])
  }

  return(parcel(parcel_id, area_m2, zone, owner, assessed_value))
}

setAs("Parcel", "data.frame", function(from) parcel_to_frame(from))
setAs("data.frame", "Parcel", function(from) frame_to_parcel(from))

#' Summarise parcels by group
#'
#' @param x A `Parcel` object.
#' @param by Grouping slot, `"zone"` or `"owner"`.
#' @return A data frame with one row per group and columns `group`, `n`,
#'   `area_m2`, `mean_area_m2`, `min_area_m2`, `max_area_m2`, `n_assessed`,
#'   `assessed_value` and `value_per_m2`.
#' @family parcel
#' @export
parcel_summary <- function(x, by = c("zone", "owner")) {
  by <- match.arg(by)
  if (length(x@parcel_id) == 0L) {
    stop("Cannot summarise an empty Parcel table.", call. = FALSE)
  }
  key <- factor(slot(x, by))
  assessed <- !is.na(x@assessed_value)

  n <- tapply(x@parcel_id, key, length)
  area_m2 <- tapply(x@area_m2, key, sum)
  mean_area_m2 <- tapply(x@area_m2, key, mean)
  min_area_m2 <- tapply(x@area_m2, key, min)
  max_area_m2 <- tapply(x@area_m2, key, max)
  n_assessed <- tapply(assessed, key, sum)
  assessed_value <- tapply(x@assessed_value, key, sum, na.rm = TRUE)

  # Only assessed parcels contribute area to the value density
  assessed_area <- tapply(ifelse(assessed, x@area_m2, 0), key, sum)
  value_per_m2 <- ifelse(assessed_area > 0, assessed_value / assessed_area, NA_real_)

  out <- data.frame(
    group = levels(key),
    n = as.integer(n),
    area_m2 = as.numeric(area_m2),
    mean_area_m2 = as.numeric(mean_area_m2),
    min_area_m2 = as.numeric(min_area_m2),
    max_area_m2 = as.numeric(max_area_m2),
    n_assessed = as.integer(n_assessed),
    assessed_value = as.numeric(assessed_value),
    value_per_m2 = as.numeric(value_per_m2),
    stringsAsFactors = FALSE
  )
  return(out)
}

#' @export
setMethod("show", "Parcel", function(object) {
  cat(sprintf("<Parcel> %d parcels, %g m2\n", length(object), sum(object@area_m2)))
  print(utils::head(parcel_to_frame(object)), row.names = FALSE)
  invisible(object)
})
