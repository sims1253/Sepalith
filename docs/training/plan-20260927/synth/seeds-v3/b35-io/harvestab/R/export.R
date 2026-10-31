#' Export the households table
#'
#' @description Writes `households.csv` with columns `hh_id`, `village`,
#'   `size` and `visited`, ordered by `hh_id`. The key `hh_id` must be unique
#'   and non-missing. Missing values in other columns are written as empty
#'   cells.
#'
#' @param households A data frame of household records.
#' @param dir Export directory.
#' @param overwrite Whether an existing export file may be replaced.
#' @rdname export.tables
#' @return The path of the written file.
#' @export
export.households <- function(households, dir, overwrite = FALSE) {
  cols <- c("hh_id", "village", "size", "visited")
  # Check required columns
  missing.cols <- setdiff(cols, names(households))
  if (length(missing.cols) > 0) {
    stop(sprintf("`%s` is missing column(s): %s", "households",
                 paste(missing.cols, collapse = ", ")), call. = FALSE)
  }
  # Check the key column
  if (anyNA(households$hh_id) || anyDuplicated(households$hh_id) > 0) {
    stop(sprintf("`%s` must be unique and non-missing in `%s`.", "hh_id", "households"),
         call. = FALSE)
  }
  # Check the destination
  path <- file.path(dir, "households.csv")
  if (file.exists(path) && !overwrite) {
    stop(sprintf("Export file `%s` already exists.", path), call. = FALSE)
  }
  # Write in key order
  out <- households[order(households$hh_id), cols, drop = FALSE]
  utils::write.csv(out, path, row.names = FALSE, na = "")
  path
}

#' Export the persons table
#'
#' @description Writes `persons.csv` with columns `person_id`, `hh_id`,
#'   `age` and `sex`, ordered by `hh_id` and `person_id`.
#'
#' @param persons A data frame of person records.
#' @param dir Export directory.
#' @param overwrite Whether an existing export file may be replaced.
#' @rdname export.tables
#' @return The path of the written file.
#' @export
export.persons <- function(persons, dir, overwrite = FALSE) {
  cols <- c("person_id", "hh_id", "age", "sex")
  # Check required columns
  missing.cols <- setdiff(cols, names(persons))
  if (length(missing.cols) > 0) {
    stop(sprintf("`%s` is missing column(s): %s", "persons",
                 paste(missing.cols, collapse = ", ")), call. = FALSE)
  }
  # Check the key column
  if (anyNA(persons$person_id) || anyDuplicated(persons$person_id) > 0) {
    stop(sprintf("`%s` must be unique and non-missing in `%s`.", "person_id", "persons"),
         call. = FALSE)
  }
  # Check the destination
  path <- file.path(dir, "persons.csv")
  if (file.exists(path) && !overwrite) {
    stop(sprintf("Export file `%s` already exists.", path), call. = FALSE)
  }
  # Write in key order
  out <- persons[order(persons$hh_id, persons$person_id), cols, drop = FALSE]
  utils::write.csv(out, path, row.names = FALSE, na = "")
  path
}

#' Export the visits table
#'
#' @description Writes `visits.csv` with columns `visit_id`, `hh_id`, `date`
#'   and `outcome`, ordered by `date` and `visit_id`.
#'
#' @param visits A data frame of visit records.
#' @param dir Export directory.
#' @param overwrite Whether an existing export file may be replaced.
#' @rdname export.tables
#' @return The path of the written file.
#' @export
export.visits <- function(visits, dir, overwrite = FALSE) {
  cols <- c("visit_id", "hh_id", "date", "outcome")
  # Check required columns
  missing.cols <- setdiff(cols, names(visits))
  if (length(missing.cols) > 0) {
    stop(sprintf("`%s` is missing column(s): %s", "visits",
                 paste(missing.cols, collapse = ", ")), call. = FALSE)
  }
  # Check the key column
  if (anyNA(visits$visit_id) || anyDuplicated(visits$visit_id) > 0) {
    stop(sprintf("`%s` must be unique and non-missing in `%s`.", "visit_id", "visits"),
         call. = FALSE)
  }
  # Check the destination
  path <- file.path(dir, "visits.csv")
  if (file.exists(path) && !overwrite) {
    stop(sprintf("Export file `%s` already exists.", path), call. = FALSE)
  }
  # Write in key order
  out <- visits[order(visits$date, visits$visit_id), cols, drop = FALSE]
  utils::write.csv(out, path, row.names = FALSE, na = "")
  path
}

#' Export all survey tables
#'
#' @description Runs the per-table export steps in order and then writes
#'   `manifest.csv` listing every exported file.
#'
#' @param tables A named list with elements `households`, `persons` and
#'   `visits`.
#' @param dir Export directory, created if needed.
#' @param overwrite Whether existing export files may be replaced.
#' @return The manifest: a data frame with columns `table`, `file`, `rows`
#'   and `md5`, one row per exported table.
#' @export
export.all <- function(tables, dir, overwrite = FALSE) {
  missing.tables <- setdiff(c("households", "persons", "visits"), names(tables))
  if (length(missing.tables) > 0) {
    stop(sprintf("`tables` is missing element(s): %s",
                 paste(missing.tables, collapse = ", ")), call. = FALSE)
  }
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  paths <- c(households = NA_character_, persons = NA_character_, visits = NA_character_)

  # Export `households`
  paths[["households"]] <- export.households(tables$households, dir, overwrite = overwrite)
  # Export `persons`
  paths[["persons"]] <- export.persons(tables$persons, dir, overwrite = overwrite)
  # Export `visits`
  paths[["visits"]] <- export.visits(tables$visits, dir, overwrite = overwrite)

  manifest <- data.frame(
    table = names(paths),
    file = basename(paths),
    rows = vapply(tables[names(paths)], nrow, integer(1)),
    md5 = unname(tools::md5sum(paths)),
    stringsAsFactors = FALSE
  )
  rownames(manifest) <- NULL
  utils::write.csv(manifest, file.path(dir, "manifest.csv"), row.names = FALSE)
  manifest
}

#' Remove an export directory's files
#'
#' @description Deletes the files listed in `manifest.csv` and the manifest
#'   itself, leaving any other files in `dir` alone.
#'
#' @param dir Export directory.
#' @return The names of the removed files, invisibly.
#' @keywords internal
export.clean <- function(dir) {
  manifest.path <- file.path(dir, "manifest.csv")
  if (!file.exists(manifest.path)) {
    return(invisible(character()))
  }
  manifest <- utils::read.csv(manifest.path, stringsAsFactors = FALSE)
  targets <- c(manifest$file, "manifest.csv")
  removed <- targets[file.exists(file.path(dir, targets))]
  unlink(file.path(dir, removed))
  invisible(removed)
}
