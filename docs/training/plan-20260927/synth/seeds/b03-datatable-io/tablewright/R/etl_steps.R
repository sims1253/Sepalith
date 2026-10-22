#' Check that a table has the required columns
#'
#' Stops with an informative message listing every missing column, so the
#' caller sees all problems at once instead of one per run.
#'
#' @param dt A data.table holding the records being transformed.
#' @param cols Character vector of column names that must be present.
#' @param step Short label of the calling step, used in error messages.
#' @return `dt`, invisibly, when all columns are present.
#' @keywords internal
check_has_cols <- function(dt, cols, step = "step") {
  missing <- setdiff(cols, names(dt))
  if (length(missing) > 0L) {
    stop(
      sprintf("[%s] missing column(s): %s", step, paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }
  invisible(dt)
}

#' Check that a set of key columns identifies rows uniquely
#'
#' Duplicated keys are reported with the first few offending values so
#' upstream extracts can be fixed quickly.
#'
#' @param dt A data.table holding the records being transformed.
#' @param key Character vector of columns that together form the key.
#' @param step Short label of the calling step, used in error messages.
#' @return `dt`, invisibly, when the key is unique.
#' @keywords internal
check_unique_key <- function(dt, key, step = "step") {
  check_has_cols(dt, key, step)
  dups <- dt[, .N, by = key][N > 1L]
  if (nrow(dups) > 0L) {
    shown <- utils::head(do.call(paste, c(dups[, ..key], sep = "/")), 5L)
    stop(
      sprintf(
        "[%s] %d duplicated key(s), e.g. %s",
        step, nrow(dups), paste(shown, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  invisible(dt)
}

#' Check that columns contain no missing values
#'
#' Counts `NA` values per column and fails if any column has at least one.
#'
#' @param dt A data.table holding the records being transformed.
#' @param cols Character vector of column names that must be complete.
#' @param step Short label of the calling step, used in error messages.
#' @return `dt`, invisibly, when all listed columns are complete.
#' @keywords internal
check_no_missing <- function(dt, cols, step = "step") {
  check_has_cols(dt, cols, step)
  n_na <- dt[, vapply(.SD, function(x) sum(is.na(x)), integer(1)), .SDcols = cols]
  bad <- n_na[n_na > 0L]
  if (length(bad) > 0L) {
    stop(
      sprintf(
        "[%s] missing values in: %s",
        step, paste0(names(bad), " (", bad, ")", collapse = ", ")
      ),
      call. = FALSE
    )
  }
  invisible(dt)
}

#' Rename columns in place
#'
#' A thin wrapper around [data.table::setnames()] that validates the mapping
#' first and records the step on the table.
#'
#' @param dt A data.table holding the records being transformed.
#' @param mapping Named character vector; names are new names, values are old.
#' @return `dt`, modified by reference and returned invisibly.
#' @importFrom data.table setnames
#' @export
step_rename <- function(dt, mapping) {
  check_has_cols(dt, unname(mapping), "rename")
  setnames(dt, old = unname(mapping), new = names(mapping))
  log_step(dt, "rename", length(mapping))
  invisible(dt)
}

#' Coerce columns to declared types in place
#'
#' Each column listed in `types` is converted with the matching `as.*()`
#' function; values that fail to convert become `NA` with a warning count.
#'
#' @param dt A data.table holding the records being transformed.
#' @param types Named character vector mapping column names to one of
#'   `"integer"`, `"numeric"`, `"character"`, `"logical"` or `"Date"`.
#' @return `dt`, modified by reference and returned invisibly.
#' @importFrom data.table set
#' @export
step_cast <- function(dt, types) {
  check_has_cols(dt, names(types), "cast")
  casters <- list(
    integer = as.integer,
    numeric = as.numeric,
    character = as.character,
    logical = as.logical,
    Date = as.Date
  )
  unknown <- setdiff(unique(types), names(casters))
  if (length(unknown) > 0L) {
    stop("Unsupported type(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  for (col in names(types)) {
    before <- sum(is.na(dt[[col]]))
    converted <- suppressWarnings(casters[[types[[col]]]](dt[[col]]))
    lost <- sum(is.na(converted)) - before
    if (lost > 0L) {
      warning(sprintf("[cast] %d value(s) in '%s' became NA", lost, col), call. = FALSE)
    }
    set(dt, j = col, value = converted)
  }
  log_step(dt, "cast", length(types))
  invisible(dt)
}

#' Derive new columns in place
#'
#' Evaluates each expression in `exprs` inside the table and assigns the result
#' by reference, optionally grouped by `by`.
#'
#' @param dt A data.table holding the records being transformed.
#' @param exprs A named list of quoted expressions, one per new column.
#' @param by Optional character vector of grouping columns.
#' @return `dt`, modified by reference and returned invisibly.
#' @export
step_derive <- function(dt, exprs, by = NULL) {
  if (is.null(names(exprs)) || any(names(exprs) == "")) {
    stop("All derived expressions must be named.", call. = FALSE)
  }
  if (!is.null(by)) {
    check_has_cols(dt, by, "derive")
  }
  for (nm in names(exprs)) {
    e <- exprs[[nm]]
    dt[, (nm) := eval(e), by = by]
  }
  log_step(dt, "derive", length(exprs))
  invisible(dt)
}

#' Drop rows that fail a condition
#'
#' Unlike the other steps this returns a new table, since data.table cannot
#' delete rows by reference.
#'
#' @param dt A data.table holding the records being transformed.
#' @param condition A quoted logical expression evaluated within `dt`.
#' @return A data.table containing only the rows where `condition` is `TRUE`.
#' @export
step_filter <- function(dt, condition) {
  keep <- dt[, eval(condition)]
  if (!is.logical(keep) || length(keep) != nrow(dt)) {
    stop("Filter condition must give one logical value per row.", call. = FALSE)
  }
  keep[is.na(keep)] <- FALSE
  out <- dt[keep]
  log_step(out, "filter", nrow(dt) - nrow(out))
  out
}

#' Append an entry to a table's step log
#'
#' The log is stored as an attribute so it travels with the table through
#' later in-place steps.
#'
#' @param dt A data.table holding the records being transformed.
#' @param step Short label of the calling step, used in error messages.
#' @param n Number of columns or rows the step touched.
#' @return `dt`, invisibly.
#' @importFrom data.table setattr
#' @noRd
log_step <- function(dt, step, n) {
  entry <- data.frame(step = step, n = as.integer(n), at = Sys.time())
  prev <- attr(dt, "tw_log")
  setattr(dt, "tw_log", rbind(prev, entry))
  invisible(dt)
}

#' Run a list of steps against a table
#'
#' Each element of `steps` is a list with a `type` (one of `"rename"`,
#' `"cast"`, `"derive"`, `"filter"`) and the arguments for that step. Errors
#' are re-raised with the index of the failing step.
#'
#' @param dt A data.table holding the records being transformed.
#' @param steps A list of step specifications.
#' @param required Optional character vector of columns checked before running.
#' @param verbose If `TRUE`, print the row and column count after each step.
#' @return The transformed data.table, carrying the step log attribute.
#' @export
run_steps <- function(dt, steps, required = NULL, verbose = FALSE) {
  if (!data.table::is.data.table(dt)) {
    dt <- data.table::as.data.table(dt)
  } else {
    dt <- data.table::copy(dt)
  }
  if (!is.null(required)) {
    check_has_cols(dt, required, "run_steps")
  }
  types <- vapply(steps, function(s) s$type %||% NA_character_, character(1))
  unknown <- setdiff(types, c("rename", "cast", "derive", "filter"))
  if (length(unknown) > 0L) {
    stop("Unknown step type(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  for (i in seq_along(steps)) {
    s <- steps[[i]]
    dt <- tryCatch(
      switch(s$type,
        rename = step_rename(dt, s$mapping),
        cast = step_cast(dt, s$types),
        derive = step_derive(dt, s$exprs, by = s$by),
        filter = step_filter(dt, s$condition)
      ),
      error = function(e) {
        stop(sprintf("Step %d (%s) failed: %s", i, s$type, conditionMessage(e)), call. = FALSE)
      }
    )
    if (verbose) {
      message(sprintf(
        "[%d/%d] %-7s %d rows x %d cols",
        i, length(steps), s$type, nrow(dt), ncol(dt)
      ))
    }
  }
  dt
}
