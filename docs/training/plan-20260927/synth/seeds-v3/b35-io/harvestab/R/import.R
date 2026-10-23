#' Import the households table
#'
#' @description Reads `households.csv` as written by `export.households()`
#'   and checks that `hh_id` is unique and non-missing.
#'
#' @param dir Export directory.
#' @rdname import.tables
#' @return A data frame with columns `hh_id`, `village`, `size` and
#'   `visited`.
#' @export
import.households <- function(dir) {
  # Check the file
  path <- file.path(dir, "households.csv")
  if (!file.exists(path)) {
    stop(sprintf("Export file `%s` does not exist.", path), call. = FALSE)
  }
  classes <- c(hh_id = "character", village = "character", size = "integer",
               visited = "logical")
  # Read with fixed column classes
  data <- utils::read.csv(path, colClasses = classes, na.strings = "",
                          stringsAsFactors = FALSE)
  # Check the header
  if (!identical(names(data), names(classes))) {
    stop(sprintf("`%s` has unexpected columns.", basename(path)), call. = FALSE)
  }
  # Check the key column
  if (anyNA(data$hh_id) || anyDuplicated(data$hh_id) > 0) {
    stop(sprintf("`%s` has missing or duplicated `%s` values.", basename(path), "hh_id"),
         call. = FALSE)
  }
  data
}

#' Import the persons table
#'
#' @description Reads `persons.csv` as written by `export.persons()` and
#'   checks that `person_id` is unique and non-missing.
#'
#' @param dir Export directory.
#' @rdname import.tables
#' @return A data frame with columns `person_id`, `hh_id`, `age` and `sex`.
#' @export
import.persons <- function(dir) {
  # Check the file
  path <- file.path(dir, "persons.csv")
  if (!file.exists(path)) {
    stop(sprintf("Export file `%s` does not exist.", path), call. = FALSE)
  }
  classes <- c(person_id = "character", hh_id = "character", age = "integer",
               sex = "character")
  # Read with fixed column classes
  data <- utils::read.csv(path, colClasses = classes, na.strings = "",
                          stringsAsFactors = FALSE)
  # Check the header
  if (!identical(names(data), names(classes))) {
    stop(sprintf("`%s` has unexpected columns.", basename(path)), call. = FALSE)
  }
  # Check the key column
  if (anyNA(data$person_id) || anyDuplicated(data$person_id) > 0) {
    stop(sprintf("`%s` has missing or duplicated `%s` values.", basename(path), "person_id"),
         call. = FALSE)
  }
  data
}

#' Import the visits table
#'
#' @description Reads `visits.csv` as written by `export.visits()` and
#'   checks that `visit_id` is unique and non-missing.
#'
#' @param dir Export directory.
#' @rdname import.tables
#' @return A data frame with columns `visit_id`, `hh_id`, `date` and
#'   `outcome`.
#' @export
import.visits <- function(dir) {
  # Check the file
  path <- file.path(dir, "visits.csv")
  if (!file.exists(path)) {
    stop(sprintf("Export file `%s` does not exist.", path), call. = FALSE)
  }
  classes <- c(visit_id = "character", hh_id = "character", date = "Date",
               outcome = "character")
  # Read with fixed column classes
  data <- utils::read.csv(path, colClasses = classes, na.strings = "",
                          stringsAsFactors = FALSE)
  # Check the header
  if (!identical(names(data), names(classes))) {
    stop(sprintf("`%s` has unexpected columns.", basename(path)), call. = FALSE)
  }
  # Check the key column
  if (anyNA(data$visit_id) || anyDuplicated(data$visit_id) > 0) {
    stop(sprintf("`%s` has missing or duplicated `%s` values.", basename(path), "visit_id"),
         call. = FALSE)
  }
  data
}

#' Check that exported files match their manifest
#'
#' @param dir Export directory containing `manifest.csv`.
#' @return The manifest with an added logical column `ok`; an error is raised
#'   naming the first file whose checksum differs.
#' @keywords internal
import.verify <- function(dir) {
  manifest.path <- file.path(dir, "manifest.csv")
  if (!file.exists(manifest.path)) {
    stop(sprintf("Export file `%s` does not exist.", manifest.path), call. = FALSE)
  }
  manifest <- utils::read.csv(manifest.path, stringsAsFactors = FALSE)

  actual <- unname(tools::md5sum(file.path(dir, manifest$file)))
  manifest$ok <- !is.na(actual) & actual == manifest$md5
  if (!all(manifest$ok)) {
    stop(sprintf("Checksum mismatch for `%s`.", manifest$file[!manifest$ok][1]), call. = FALSE)
  }
  manifest
}

#' Import all survey tables
#'
#' @description Verifies the manifest, runs the per-table import steps and
#'   checks that every `hh_id` in `persons` and `visits` refers to a
#'   household.
#'
#' @param dir Export directory.
#' @param verify Whether to check file checksums against `manifest.csv`.
#' @return A named list with elements `households`, `persons` and `visits`.
#' @export
import.all <- function(dir, verify = TRUE) {
  if (verify) {
    import.verify(dir)
  }
  tables <- list(households = NULL, persons = NULL, visits = NULL)

  # Import `households`
  tables$households <- import.households(dir)
  # Import `persons`
  tables$persons <- import.persons(dir)
  # Import `visits`
  tables$visits <- import.visits(dir)

  # Check `persons` keys
  orphans <- setdiff(tables$persons$hh_id, tables$households$hh_id)
  if (length(orphans) > 0) {
    stop(sprintf("`%s` refers to unknown household(s): %s", "persons",
                 paste(orphans, collapse = ", ")), call. = FALSE)
  }
  # Check `visits` keys
  orphans <- setdiff(tables$visits$hh_id, tables$households$hh_id)
  if (length(orphans) > 0) {
    stop(sprintf("`%s` refers to unknown household(s): %s", "visits",
                 paste(orphans, collapse = ", ")), call. = FALSE)
  }
  tables
}
