#' @title Read a DESCRIPTION file into a named character vector
#' @description Parses a DCF-formatted DESCRIPTION file and returns its fields
#'   with continuation lines collapsed to single spaces.
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return A named character vector, one element per field.
#' @export
desc_read <- function(path = ".") {
  file <- desc_path(path)
  if (!file.exists(file)) {
    stop("No DESCRIPTION file found at '", file, "'.", call. = FALSE)
  }

  # Keep the line structure of free-text fields; everything else is folded.
  fields <- read.dcf(file, keep.white = c("Description", "Authors@R"))
  if (nrow(fields) != 1L) {
    stop("DESCRIPTION at '", file, "' must contain exactly one record.", call. = FALSE)
  }

  out <- fields[1L, , drop = TRUE]
  out <- out[!is.na(out)]
  return(trimws(out))
}

#' @title Write a named character vector back to a DESCRIPTION file
#' @description Fields are written in a canonical order: well-known fields
#'   first, then any remaining fields in the order they were supplied.
#' @param fields A named character vector of DESCRIPTION fields.
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return The path to the written file, invisibly.
#' @export
desc_write <- function(fields, path = ".") {
  if (is.null(names(fields)) || any(names(fields) == "")) {
    stop("`fields` must be a fully named character vector.", call. = FALSE)
  }
  file <- desc_path(path)

  # Mirrors the order most DESCRIPTION files use, so diffs stay small.
  known <- c(
    "Package", "Type", "Title", "Version", "Authors@R", "Description",
    "License", "URL", "BugReports", "Depends", "Imports", "Suggests",
    "LinkingTo", "Config/testthat/edition", "Encoding", "Roxygen", "RoxygenNote"
  )
  ord <- c(intersect(known, names(fields)), setdiff(names(fields), known))
  fields <- fields[ord]

  mat <- matrix(fields, nrow = 1L, dimnames = list(NULL, names(fields)))
  write.dcf(mat, file = file, width = 80L, keep.white = c("Description", "Authors@R"))
  return(invisible(file))
}

#' @title Resolve the DESCRIPTION path for a package
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @noRd
desc_path <- function(path) {
  if (basename(path) == "DESCRIPTION") {
    return(path)
  }
  return(file.path(path, "DESCRIPTION"))
}

#' @title Get or set a single DESCRIPTION field
#' @description `desc_set_field()` returns the previous value of the field,
#'   invisibly.
#' @param field Name of the field, e.g. `"Version"`.
#' @param default Value returned when the field is absent.
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return A single string, or `default` when the field is missing.
#' @export
desc_get_field <- function(field, default = NA_character_, path = ".") {
  fields <- desc_read(path)
  if (!field %in% names(fields)) {
    return(default)
  }
  return(unname(fields[[field]]))
}

#' @rdname desc_get_field
#' @param value New value for the field, a single string.
#' @export
desc_set_field <- function(field, value, path = ".") {
  if (!is.character(value) || length(value) != 1L || is.na(value)) {
    stop("`value` must be a single non-missing string.", call. = FALSE)
  }

  fields <- desc_read(path)
  old <- if (field %in% names(fields)) unname(fields[[field]]) else NA_character_
  fields[[field]] <- value
  desc_write(fields, path)
  return(invisible(old))
}

#' @title Remove a single DESCRIPTION field
#' @param field Name of the field, e.g. `"Version"`.
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return The removed value of the field, invisibly.
#' @export
desc_drop_field <- function(field, path = ".") {
  fields <- desc_read(path)
  if (!field %in% names(fields)) {
    message("Field '", field, "' is not present; nothing to drop.")
    return(invisible(NA_character_))
  }
  old <- unname(fields[[field]])
  fields <- fields[names(fields) != field]
  desc_write(fields, path)
  return(invisible(old))
}

#' @title Check a DESCRIPTION for common problems
#' @description Looks for missing required fields, a title ending in a period,
#'   a description that does not start with a capital letter and an unparseable
#'   version string.
#' @param path Path to the package root or to the DESCRIPTION file itself.
#' @return A character vector of problems, empty when none were found.
#' @export
desc_check <- function(path = ".") {
  fields <- desc_read(path)
  problems <- character()

  required <- c("Package", "Title", "Version", "Description", "License")
  missing <- setdiff(required, names(fields))
  if (length(missing) > 0L) {
    problems <- c(problems, paste0("Missing required field: ", missing))
  }

  # The 65-character limit and the trailing period follow CRAN policy.
  if ("Title" %in% names(fields)) {
    title <- fields[["Title"]]
    if (grepl("\\.$", title)) {
      problems <- c(problems, "Title should not end in a period.")
    }
    if (nchar(title) > 65L) {
      problems <- c(problems, "Title is longer than 65 characters.")
    }
  }
  if ("Description" %in% names(fields)) {
    if (!grepl("^[A-Z]", fields[["Description"]])) {
      problems <- c(problems, "Description should start with a capital letter.")
    }
  }
  if ("Version" %in% names(fields)) {
    # strict = FALSE turns an invalid version into NA instead of an error.
    ver <- package_version(fields[["Version"]], strict = FALSE)
    if (is.na(ver)) {
      problems <- c(problems, paste0("Version '", fields[["Version"]], "' is not valid."))
    }
  }

  return(problems)
}
