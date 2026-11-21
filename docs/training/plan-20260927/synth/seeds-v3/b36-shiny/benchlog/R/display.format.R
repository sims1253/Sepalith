#' @title Format volumes
#' @description Volumes are stored in microlitres; `unit = "auto"` picks the
#'   largest unit in which the biggest value is at least 1.
#' @param x Numeric vector of volumes in microlitres.
#' @param unit Display unit: `"auto"`, `"uL"`, `"mL"` or `"L"`.
#' @param digits Number of decimal places.
#' @return A character vector the same length as `x`.
#' @family display formatters
#' @export
fmt.volume <- function(x, unit = c("auto", "uL", "mL", "L"), digits = 1) {
  unit <- match.arg(unit)
  scale <- c(uL = 1, mL = 1e3, L = 1e6)
  if (unit == "auto") {
    biggest <- suppressWarnings(max(abs(x), na.rm = TRUE))
    if (!is.finite(biggest) || biggest < scale[["mL"]]) {
      unit <- "uL"
    } else if (biggest < scale[["L"]]) {
      unit <- "mL"
    } else {
      unit <- "L"
    }
  }
  out <- paste(formatC(x / scale[[unit]], format = "f", digits = digits), unit)
  out[is.na(x)] <- "-"
  out
}

#' @title Format masses
#' @description Masses are stored in nanograms; `unit = "auto"` picks the
#'   largest unit in which the biggest value is at least 1.
#' @param x Numeric vector of masses in nanograms.
#' @param unit Display unit: `"auto"`, `"ng"`, `"ug"` or `"mg"`.
#' @param digits Number of decimal places.
#' @return A character vector the same length as `x`.
#' @family display formatters
#' @export
fmt.mass <- function(x, unit = c("auto", "ng", "ug", "mg"), digits = 1) {
  unit <- match.arg(unit)
  scale <- c(ng = 1, ug = 1e3, mg = 1e6)
  if (unit == "auto") {
    biggest <- suppressWarnings(max(abs(x), na.rm = TRUE))
    if (!is.finite(biggest) || biggest < scale[["ug"]]) {
      unit <- "ng"
    } else if (biggest < scale[["mg"]]) {
      unit <- "ug"
    } else {
      unit <- "mg"
    }
  }
  out <- paste(formatC(x / scale[[unit]], format = "f", digits = digits), unit)
  out[is.na(x)] <- "-"
  out
}

#' @title Format durations
#' @description Durations are stored in seconds; `unit = "auto"` picks the
#'   largest unit in which the biggest value is at least 1.
#' @param x Numeric vector of durations in seconds.
#' @param unit Display unit: `"auto"`, `"s"`, `"min"` or `"h"`.
#' @param digits Number of decimal places.
#' @return A character vector the same length as `x`.
#' @family display formatters
#' @export
fmt.duration <- function(x, unit = c("auto", "s", "min", "h"), digits = 1) {
  unit <- match.arg(unit)
  scale <- c(s = 1, min = 60, h = 3600)
  if (unit == "auto") {
    biggest <- suppressWarnings(max(abs(x), na.rm = TRUE))
    if (!is.finite(biggest) || biggest < scale[["min"]]) {
      unit <- "s"
    } else if (biggest < scale[["h"]]) {
      unit <- "min"
    } else {
      unit <- "h"
    }
  }
  out <- paste(formatC(x / scale[[unit]], format = "f", digits = digits), unit)
  out[is.na(x)] <- "-"
  out
}

#' @title Format a storage location
#' @description Builds a location code such as `F02/R05/B12/A3` from its
#'   parts; parts that are `NA` are left out.
#' @param freezer Freezer number.
#' @param rack Rack number within the freezer.
#' @param box Box number within the rack.
#' @param slot Slot label within the box, such as `"A3"`.
#' @return A single string, `"unassigned"` when no part is known.
#' @family display formatters
#' @export
fmt.storage.location <- function(freezer = NA, rack = NA, box = NA, slot = NA) {
  parts <- character(0)

  # Freezer part
  if (!is.na(freezer)) {
    parts <- c(parts, sprintf("F%02d", as.integer(freezer)))
  }

  # Rack part
  if (!is.na(rack)) {
    parts <- c(parts, sprintf("R%02d", as.integer(rack)))
  }

  # Box part
  if (!is.na(box)) {
    parts <- c(parts, sprintf("B%02d", as.integer(box)))
  }

  # Slot part
  if (!is.na(slot)) {
    parts <- c(parts, toupper(as.character(slot)))
  }

  if (length(parts) == 0L) {
    return("unassigned")
  }
  paste(parts, collapse = "/")
}

#' @title Describe a sample status as a badge
#' @param status One of `"received"`, `"stored"`, `"in_use"`, `"depleted"` or
#'   `"discarded"`.
#' @return A list with elements `label`, `class` and `icon`.
#' @family display formatters
#' @export
fmt.status.badge <- function(status = c("received", "stored", "in_use", "depleted",
                                        "discarded")) {
  status <- match.arg(status)
  switch(status,
    received = list(
      label = "Received",
      class = "badge bg-info",
      icon = "inbox"
    ),
    stored = list(
      label = "Stored",
      class = "badge bg-primary",
      icon = "snowflake"
    ),
    in_use = list(
      label = "In use",
      class = "badge bg-warning",
      icon = "flask"
    ),
    depleted = list(
      label = "Depleted",
      class = "badge bg-secondary",
      icon = "battery-empty"
    ),
    discarded = list(
      label = "Discarded",
      class = "badge bg-dark",
      icon = "trash"
    )
  )
}

#' @title Format a sample for its detail card
#' @param sample A one-row data frame or list with elements `sample_id`,
#'   `type`, `volume`, `concentration`, `freezer`, `rack`, `box`, `slot` and
#'   `received`.
#' @param today Date used to compute the age of the sample.
#' @return A list with elements `title`, `subtitle`, `volume`, `amount`,
#'   `location` and `age`.
#' @family display formatters
#' @export
fmt.sample.card <- function(sample, today = Sys.Date()) {
  # 1. title and subtitle
  title <- as.character(sample$sample_id)
  subtitle <- paste("Type:", sample$type)

  # 2. volume and total amount (volume times concentration)
  volume <- fmt.volume(sample$volume)
  amount <- fmt.mass(sample$volume * sample$concentration)

  # 3. location
  location <- fmt.storage.location(sample$freezer, sample$rack, sample$box, sample$slot)

  # 4. age in days since receipt
  days <- as.integer(as.Date(today) - as.Date(sample$received))
  age <- if (is.na(days)) "-" else paste(days, if (days == 1L) "day" else "days")
  list(
    title = title,
    subtitle = subtitle,
    volume = volume,
    amount = amount,
    location = location,
    age = age
  )
}

#' @title Format the sample table for display
#' @param samples A data frame with columns `sample_id`, `volume`,
#'   `concentration`, `temperature`, `status` and `updated`.
#' @param unit Display unit for volumes, passed to [fmt.volume()].
#' @return A data frame of character columns whose names are the headers
#'   `Sample`, `Volume`, `Concentration`, `Storage`, `Status` and `Updated`.
#' @family display formatters
#' @export
fmt.sample.table <- function(samples, unit = "auto") {
  headers <- c(sample_id = "Sample", volume = "Volume", concentration = "Concentration",
               temperature = "Storage", status = "Status", updated = "Updated")
  out <- samples[names(headers)]

  # Column `volume`
  out$volume <- fmt.volume(samples$volume, unit)

  # Column `concentration`
  out$concentration <- paste(formatC(samples$concentration, format = "f", digits = 1),
                             "ng/uL")
  out$concentration[is.na(samples$concentration)] <- "-"

  # Column `temperature`
  out$temperature <- paste(samples$temperature, "C")
  out$temperature[is.na(samples$temperature)] <- "-"

  # Column `status`
  out$status <- vapply(samples$status, function(s) fmt.status.badge(s)$label,
                       character(1), USE.NAMES = FALSE)

  # Column `updated`
  out$updated <- format(samples$updated, "%Y-%m-%d %H:%M")
  out$updated[is.na(samples$updated)] <- "-"
  names(out) <- unname(headers)
  out
}
