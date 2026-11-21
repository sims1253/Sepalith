#' Describe one schema column
#'
#' @param type Storage type: one of `"character"`, `"numeric"`, `"integer"`,
#'   `"logical"` or `"Date"`.
#' @param required Whether the column must be present.
#' @param nullable Whether the column may contain `NA`.
#' @return A list with elements `type`, `required` and `nullable`.
#' @family schema helpers
#' @export
schemaColumn <- function(type = c("character", "numeric", "integer", "logical", "Date"),
                         required = TRUE, nullable = TRUE) {
  type <- match.arg(type)
  if (!is.logical(required) || length(required) != 1L || is.na(required)) {
    stop("`required` must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.logical(nullable) || length(nullable) != 1L || is.na(nullable)) {
    stop("`nullable` must be TRUE or FALSE.", call. = FALSE)
  }
  list(type = type, required = required, nullable = nullable)
}

#' Create a table schema
#'
#' @param ... Named column descriptions created with [schemaColumn()].
#' @return An object of class `sieve_schema`: a named list of column
#'   descriptions.
#' @family schema helpers
#' @export
newSchema <- function(...) {
  columns <- list(...)
  if (length(columns) == 0L || is.null(names(columns)) || any(!nzchar(names(columns)))) {
    stop("Every column description must be named.", call. = FALSE)
  }
  if (anyDuplicated(names(columns))) {
    stop("Column names in a schema must be unique.", call. = FALSE)
  }
  fields <- c("type", "required", "nullable")
  valid <- vapply(columns, function(col) is.list(col) && all(fields %in% names(col)), logical(1))
  if (!all(valid)) {
    stop("Not created with schemaColumn(): ", paste(names(columns)[!valid], collapse = ", "),
         call. = FALSE)
  }
  structure(columns, class = "sieve_schema")
}

#' Test a column against a storage type
#'
#' @param x A column vector.
#' @param type Storage type as accepted by [schemaColumn()].
#' @return `TRUE` or `FALSE`.
#' @keywords internal
#' @noRd
columnHasType <- function(x, type) {
  switch(type,
    character = is.character(x),
    numeric = is.numeric(x),
    integer = is.integer(x) || (is.numeric(x) && all(x == round(x), na.rm = TRUE)),
    logical = is.logical(x),
    Date = inherits(x, "Date"),
    FALSE
  )
}

#' Infer a schema from a data frame
#'
#' @param data A data frame to describe.
#' @param required Whether inferred columns are required.
#' @return A `sieve_schema` with one column description per column of
#'   `data`; `nullable` is `TRUE` where the column contains `NA`.
#' @family schema helpers
#' @export
inferSchema <- function(data, required = TRUE) {
  columns <- lapply(data, function(x) {
    type <- if (inherits(x, "Date")) {
      "Date"
    } else if (is.integer(x)) {
      "integer"
    } else if (is.numeric(x)) {
      "numeric"
    } else if (is.logical(x)) {
      "logical"
    } else {
      "character"
    }
    schemaColumn(type, required = required, nullable = anyNA(x))
  })
  do.call(newSchema, columns)
}

#' Check a data frame against a schema
#'
#' @param data A data frame to check.
#' @param schema A `sieve_schema` created with [newSchema()].
#' @param allowExtra Whether columns not in `schema` are tolerated.
#' @return A list with elements `missing`, `unexpected`, `wrongType`,
#'   `nullViolations`, `empty` (character vectors of column names) and `ok`.
#' @family schema helpers
#' @export
checkSchema <- function(data, schema, allowExtra = FALSE) {
  if (!inherits(schema, "sieve_schema")) {
    stop("`schema` must be created with newSchema().", call. = FALSE)
  }
  required <- names(schema)[vapply(schema, `[[`, logical(1), "required")]

  # 1. Required columns that are absent
  missing <- setdiff(required, names(data))

  # 2. Columns the schema does not describe
  unexpected <- if (allowExtra) character() else setdiff(names(data), names(schema))

  # 3. Present columns with the wrong storage type
  present <- intersect(names(schema), names(data))
  typeOk <- vapply(present, function(col) columnHasType(data[[col]], schema[[col]]$type),
                   logical(1))
  wrongType <- present[!typeOk]

  # 4. Non-nullable columns that contain NA
  nullOk <- vapply(present, function(col) schema[[col]]$nullable || !anyNA(data[[col]]),
                   logical(1))
  nullViolations <- present[!nullOk]

  # 5. Required columns without a single observed value
  emptyCol <- vapply(present, function(col) nrow(data) > 0L && all(is.na(data[[col]])),
                     logical(1))
  empty <- intersect(present[emptyCol], required)

  list(
    missing = missing,
    unexpected = unexpected,
    wrongType = wrongType,
    nullViolations = nullViolations,
    empty = empty,
    ok = length(c(missing, unexpected, wrongType, nullViolations, empty)) == 0L
  )
}

#' Compare two schemas
#'
#' @param old,new `sieve_schema` objects.
#' @return A list with elements `added`, `removed` and `changed` (character
#'   vectors of column names) and `details` (a named character vector
#'   describing the first changed property of each changed column).
#' @family schema helpers
#' @export
schemaDiff <- function(old, new) {
  oldNames <- names(old)
  newNames <- names(new)
  added <- setdiff(newNames, oldNames)
  removed <- setdiff(oldNames, newNames)
  shared <- intersect(oldNames, newNames)
  details <- character()
  for (col in shared) {
    before <- old[[col]]
    after <- new[[col]]
    if (!identical(before$type, after$type)) {
      details[[col]] <- sprintf("type %s -> %s", before$type, after$type)
    } else if (!identical(before$required, after$required)) {
      details[[col]] <- sprintf("required %s -> %s", before$required, after$required)
    } else if (!identical(before$nullable, after$nullable)) {
      details[[col]] <- sprintf("nullable %s -> %s", before$nullable, after$nullable)
    }
  }
  list(added = added, removed = removed, changed = names(details), details = details)
}

#' Print a table schema
#'
#' @param x A `sieve_schema`.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.sieve_schema <- function(x, ...) {
  cat(sprintf("<sieve_schema> %d column(s)\n", length(x)))
  for (col in names(x)) {
    spec <- x[[col]]
    flags <- c(if (spec$required) "required", if (!spec$nullable) "not null")
    cat(sprintf("  %-12s %-10s %s\n", col, spec$type, paste(flags, collapse = ", ")))
  }
  invisible(x)
}

#' Coerce a data frame to a schema
#'
#' Converts every column described in `schema` to its storage type, adds
#' absent optional columns filled with `NA`, and orders the columns as in
#' the schema. Columns not in the schema are dropped.
#'
#' @param data A data frame to check.
#' @param schema A `sieve_schema` created with [newSchema()].
#' @return A data frame with exactly the columns of `schema`.
#' @family schema helpers
#' @export
coerceToSchema <- function(data, schema) {
  n <- nrow(data)
  out <- list()
  for (col in names(schema)) {
    spec <- schema[[col]]
    if (!col %in% names(data)) {
      if (spec$required) {
        stop(sprintf("Required column `%s` is absent.", col), call. = FALSE)
      }
      value <- rep(NA, n)
    } else {
      value <- data[[col]]
    }
    out[[col]] <- switch(spec$type,
      character = {
        as.character(value)
      },
      numeric = {
        suppressWarnings(as.numeric(as.character(value)))
      },
      integer = {
        suppressWarnings(as.integer(as.character(value)))
      },
      logical = {
        as.logical(value)
      },
      Date = {
        as.Date(as.character(value), optional = TRUE)
      }
    )
  }
  as.data.frame(out, stringsAsFactors = FALSE)
}
