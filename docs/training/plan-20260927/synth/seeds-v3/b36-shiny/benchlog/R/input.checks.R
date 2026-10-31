#' @title Check a sample identifier
#' @description Sample identifiers look like `BL-2024-00042`: a lab prefix,
#'   the year and a five-digit running number.
#' @param x Value of the input.
#' @param name Name of the input, used in messages.
#' @return `NULL` when `x` is valid, otherwise a single message.
#' @rdname input.checks
#' @export
check.sample.id <- function(x, name = "sample_id") {
  if (length(x) != 1L || is.na(x) || !nzchar(trimws(x))) {
    return(paste0("`", name, "` is required."))
  }
  x <- trimws(x)
  if (!grepl("^[A-Z]{2}-[0-9]{4}-[0-9]{5}$", x)) {
    return(paste0("`", name, "` must look like BL-2024-00042."))
  }
  year <- as.integer(substr(x, 4, 7))
  this.year <- as.integer(format(Sys.Date(), "%Y"))
  if (year < 2000L || year > this.year) {
    return(paste0("`", name, "` must contain a year between 2000 and ", this.year, "."))
  }
  NULL
}

#' @title Check a sample volume
#' @description Volumes are entered in microlitres.
#' @param x Value of the input.
#' @param name Name of the input, used in messages.
#' @param max Largest accepted volume in microlitres.
#' @return `NULL` when `x` is valid, otherwise a single message.
#' @rdname input.checks
#' @export
check.volume <- function(x, name = "volume", max = 5000) {
  if (length(x) != 1L || is.na(x)) {
    return(paste0("`", name, "` is required."))
  }
  if (!is.numeric(x)) {
    return(paste0("`", name, "` must be a number."))
  }
  if (x <= 0) {
    return(paste0("`", name, "` must be positive."))
  }
  if (x > max) {
    return(paste0("`", name, "` must be at most ", max, "."))
  }
  NULL
}

#' @title Check a sample concentration
#' @description Concentrations are entered in ng/uL; zero is allowed for
#'   blanks.
#' @param x Value of the input.
#' @param name Name of the input, used in messages.
#' @param max Largest accepted concentration in ng/uL.
#' @return `NULL` when `x` is valid, otherwise a single message.
#' @rdname input.checks
#' @export
check.concentration <- function(x, name = "concentration", max = 1000) {
  if (length(x) != 1L || is.na(x)) {
    return(paste0("`", name, "` is required."))
  }
  if (!is.numeric(x)) {
    return(paste0("`", name, "` must be a number."))
  }
  if (x < 0) {
    return(paste0("`", name, "` must not be negative."))
  }
  if (x > max) {
    return(paste0("`", name, "` must be at most ", max, "."))
  }
  NULL
}

#' @title Check a storage temperature
#' @description Temperatures are entered in degrees Celsius and must match
#'   one of the freezer or fridge settings.
#' @param x Value of the input.
#' @param name Name of the input, used in messages.
#' @param allowed Accepted storage temperatures.
#' @return `NULL` when `x` is valid, otherwise a single message.
#' @rdname input.checks
#' @export
check.temperature <- function(x, name = "temperature", allowed = c(-196, -80, -20, 4)) {
  if (length(x) != 1L || is.na(x)) {
    return(paste0("`", name, "` is required."))
  }
  if (!is.numeric(x)) {
    return(paste0("`", name, "` must be a number."))
  }
  if (!x %in% allowed) {
    return(paste0("`", name, "` must be one of ", paste(allowed, collapse = ", "), "."))
  }
  NULL
}

#' @title Check the sample entry form
#' @description Runs every field check and keeps all messages so the form can
#'   highlight each failing field.
#' @param sample_id Value of the sample identifier input.
#' @param volume Value of the volume input.
#' @param concentration Value of the concentration input.
#' @param temperature Value of the storage temperature input.
#' @return A named character vector of messages, one per failing field;
#'   empty when the form is valid.
#' @export
check.sample.form <- function(sample_id, volume, concentration, temperature) {
  messages <- character(0)

  # Check `sample_id`
  msg <- check.sample.id(sample_id)
  if (!is.null(msg)) {
    messages[["sample_id"]] <- msg
  }

  # Check `volume`
  msg <- check.volume(volume)
  if (!is.null(msg)) {
    messages[["volume"]] <- msg
  }

  # Check `concentration`
  msg <- check.concentration(concentration)
  if (!is.null(msg)) {
    messages[["concentration"]] <- msg
  }

  # Check `temperature`
  msg <- check.temperature(temperature)
  if (!is.null(msg)) {
    messages[["temperature"]] <- msg
  }
  messages
}

#' @title Check a plate layout
#' @description A plate layout has one row per well with columns `well`,
#'   `sample_id`, `type` and `dilution`.
#' @param layout A data frame describing the plate.
#' @param wells Number of wells on the plate, 96 or 384.
#' @return A character vector of problems, empty when the layout is valid.
#' @export
check.plate.layout <- function(layout, wells = 96L) {
  columns <- c("well", "sample_id", "type", "dilution")
  missing <- setdiff(columns, names(layout))
  if (length(missing) > 0L) {
    return(paste0("`", missing, "` column is missing."))
  }
  rows <- if (wells == 384L) LETTERS[1:16] else LETTERS[1:8]
  cols <- if (wells == 384L) 1:24 else 1:12
  valid.wells <- paste0(rep(rows, each = length(cols)), cols)
  problems <- character(0)

  # Check `well`
  if (!all(layout$well %in% valid.wells) || anyDuplicated(layout$well) > 0L) {
    problems <- c(problems, "`well` must hold unique wells of the plate.")
  }

  # Check `sample_id`
  if (anyNA(layout$sample_id[layout$type == "sample"])) {
    problems <- c(problems, "`sample_id` must be set for every sample well.")
  }

  # Check `type`
  if (!all(layout$type %in% c("sample", "standard", "blank", "control"))) {
    problems <- c(problems, "`type` must be sample, standard, blank or control.")
  }

  # Check `dilution`
  if (!is.numeric(layout$dilution) || any(layout$dilution < 1, na.rm = TRUE)) {
    problems <- c(problems, "`dilution` must be numbers of at least 1.")
  }
  problems
}

#' @title Check the storage location form
#' @description Freezer, rack and box are numbered from 1 up to the limits
#'   in `limits`; slots are labelled `A1` to `I9`.
#' @param freezer Value of the freezer input.
#' @param rack Value of the rack input.
#' @param box Value of the box input.
#' @param slot Value of the slot input.
#' @return A named character vector of messages, one per failing field;
#'   empty when the form is valid.
#' @export
check.storage.form <- function(freezer, rack, box, slot) {
  limits <- c(freezer = 12, rack = 10, box = 25)
  messages <- character(0)

  # Check `freezer`
  if (length(freezer) != 1L || is.na(freezer)) {
    messages[["freezer"]] <- paste0("`", "freezer", "` is required.")
  } else if (freezer < 1 || freezer > limits[["freezer"]]) {
    messages[["freezer"]] <- paste0("`", "freezer", "` must be between 1 and ",
                                    limits[["freezer"]], ".")
  }

  # Check `rack`
  if (length(rack) != 1L || is.na(rack)) {
    messages[["rack"]] <- paste0("`", "rack", "` is required.")
  } else if (rack < 1 || rack > limits[["rack"]]) {
    messages[["rack"]] <- paste0("`", "rack", "` must be between 1 and ",
                                 limits[["rack"]], ".")
  }

  # Check `box`
  if (length(box) != 1L || is.na(box)) {
    messages[["box"]] <- paste0("`", "box", "` is required.")
  } else if (box < 1 || box > limits[["box"]]) {
    messages[["box"]] <- paste0("`", "box", "` must be between 1 and ",
                                limits[["box"]], ".")
  }

  # Check `slot`
  if (length(slot) != 1L || is.na(slot) || !nzchar(slot)) {
    messages[["slot"]] <- paste0("`", "slot", "` is required.")
  } else if (!grepl("^[A-I][1-9]$", toupper(slot))) {
    messages[["slot"]] <- paste0("`", "slot", "` must be between A1 and I9.")
  }
  messages
}
