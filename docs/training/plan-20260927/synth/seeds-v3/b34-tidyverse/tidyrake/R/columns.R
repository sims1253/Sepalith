#' Standardise column names
#'
#' Lower-cases names, replaces runs of non-alphanumeric characters with a
#' single underscore and strips leading and trailing underscores.
#'
#' @param x A character vector of column names.
#' @return A character vector of unique, snake_case names.
#' @export
standardise_names <- function(x) {
  x <- tolower(trimws(x))
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("^_+|_+$", "", x)
  x[x == ""] <- "column"
  make.unique(x, sep = "_")
}

#' Check that columns exist in a data frame
#'
#' @param data A data frame.
#' @param cols A character vector of column names.
#' @param arg The name of the argument that supplied `cols`, used in the
#'   error message.
#' @return `data`, invisibly.
#' @keywords internal
check_known_cols <- function(data, cols, arg) {
  unknown <- setdiff(cols, names(data))
  if (length(unknown) > 0L) {
    stop(
      sprintf("Unknown columns in `%s`: %s.", arg, toString(unknown)),
      call. = FALSE
    )
  }
  invisible(data)
}

#' Coerce columns to declared types
#'
#' Each argument names the columns that should end up with that type. The
#' columns are checked and converted in the order `integer`, `numeric`,
#' `logical`, `date`, `character`.
#'
#' @param data A data frame.
#' @param integer Columns to convert with [as.integer()].
#' @param numeric Columns to convert with [as.numeric()].
#' @param logical Columns to convert with [as.logical()].
#' @param date Columns to convert with [as.Date()].
#' @param character Columns to convert with [as.character()].
#' @param date_format Format string passed to [as.Date()].
#' @return `data` with the listed columns converted.
#' @export
coerce_columns <- function(data, integer = NULL, numeric = NULL,
                           logical = NULL, date = NULL, character = NULL,
                           date_format = "%Y-%m-%d") {
  # Coerce `integer` columns
  check_known_cols(data, integer, "integer")
  for (col in integer) {
    data[[col]] <- as.integer(data[[col]])
  }

  # Coerce `numeric` columns
  check_known_cols(data, numeric, "numeric")
  for (col in numeric) {
    data[[col]] <- as.numeric(data[[col]])
  }

  # Coerce `logical` columns
  check_known_cols(data, logical, "logical")
  for (col in logical) {
    data[[col]] <- as.logical(data[[col]])
  }

  # Coerce `date` columns
  check_known_cols(data, date, "date")
  for (col in date) {
    data[[col]] <- as.Date(data[[col]], format = date_format)
  }

  # Coerce `character` columns
  check_known_cols(data, character, "character")
  for (col in character) {
    data[[col]] <- as.character(data[[col]])
  }
  data
}

#' Tidy the text in character columns
#'
#' Applies the enabled steps in the order of the arguments: `trim`,
#' `squish`, `lower`, `na_strings` and `blank_as_na`.
#'
#' @param data A data frame.
#' @param cols Columns to clean. Defaults to every character column.
#' @param trim Remove leading and trailing whitespace?
#' @param squish Collapse internal runs of whitespace to one space?
#' @param lower Convert to lower case?
#' @param na_strings Strings that should become `NA`, compared after the
#'   earlier steps and ignoring case.
#' @param blank_as_na Should empty strings become `NA`?
#' @return `data` with the selected columns cleaned. The number of values
#'   changed per column is stored in the `"changed"` attribute.
#' @export
tidy_text_columns <- function(data, cols = NULL, trim = TRUE, squish = TRUE,
                              lower = FALSE, na_strings = c("na", "n/a", "null"),
                              blank_as_na = TRUE) {
  if (is.null(cols)) {
    cols <- names(data)[vapply(data, is.character, logical(1))]
  }
  check_known_cols(data, cols, "cols")
  changed <- stats::setNames(integer(length(cols)), cols)

  for (col in cols) {
    old <- as.character(data[[col]])
    value <- old
    # Step `trim`
    if (trim) {
      value <- trimws(value)
    }
    # Step `squish`
    if (squish) {
      value <- gsub("\\s+", " ", value)
    }
    # Step `lower`
    if (lower) {
      value <- tolower(value)
    }
    # Step `na_strings`
    if (length(na_strings) > 0L) {
      value[tolower(value) %in% tolower(na_strings)] <- NA_character_
    }
    # Step `blank_as_na`
    if (blank_as_na) {
      value[!is.na(value) & value == ""] <- NA_character_
    }
    differs <- ifelse(is.na(old) | is.na(value), is.na(old) != is.na(value), old != value)
    changed[[col]] <- sum(differs)
    data[[col]] <- value
  }

  attr(data, "changed") <- changed
  data
}

#' Describe every column of a data frame
#'
#' @param data A data frame.
#' @return A data frame with one row per column of `data` and the columns
#'   `column`, `type`, `n_missing`, `n_unique` and `example`.
#' @export
column_report <- function(data) {
  cols <- names(data)

  # One entry per column for every statistic in the report
  type <- vapply(cols, function(col) {
    class(data[[col]])[1L]
  }, character(1))
  n_missing <- vapply(cols, function(col) {
    sum(is.na(data[[col]]))
  }, integer(1))
  n_unique <- vapply(cols, function(col) {
    length(unique(data[[col]][!is.na(data[[col]])]))
  }, integer(1))
  example <- vapply(cols, function(col) {
    value <- data[[col]][!is.na(data[[col]])]
    if (length(value) == 0L) NA_character_ else as.character(value[1L])
  }, character(1))

  data.frame(
    column = cols,
    type = unname(type),
    n_missing = unname(n_missing),
    n_unique = unname(n_unique),
    example = unname(example),
    stringsAsFactors = FALSE
  )
}

#' Drop columns that match a rule
#'
#' @param data A data frame.
#' @param rule Which columns to drop: `"empty"` drops all-missing columns,
#'   `"constant"` drops columns with a single distinct value, `"duplicate"`
#'   drops columns identical to an earlier column and `"id_like"` drops
#'   character columns whose values are all distinct.
#' @param keep Columns that are never dropped.
#' @return `data` without the dropped columns. The dropped names are stored
#'   in the `"dropped"` attribute.
#' @export
drop_columns_by_rule <- function(data,
                                 rule = c("empty", "constant", "duplicate", "id_like"),
                                 keep = character()) {
  rule <- match.arg(rule)
  cols <- names(data)

  # TRUE for every column that matches the rule
  drop <- switch(rule,
    empty = {
      vapply(cols, function(col) {
        all(is.na(data[[col]]))
      }, logical(1))
    },
    constant = {
      vapply(cols, function(col) {
        length(unique(data[[col]][!is.na(data[[col]])])) <= 1L
      }, logical(1))
    },
    duplicate = {
      duplicated(lapply(data, function(value) value))
    },
    id_like = {
      vapply(cols, function(col) {
        is.character(data[[col]]) && !anyDuplicated(data[[col]])
      }, logical(1))
    }
  )

  drop <- unname(drop) & !(cols %in% keep)
  out <- data[, !drop, drop = FALSE]
  attr(out, "dropped") <- cols[drop]
  out
}

#' Fill missing values column by column
#'
#' @param data A data frame.
#' @param cols Columns to fill.
#' @param method How to fill: `"zero"`, `"mean"`, `"median"`, `"mode"` or
#'   `"locf"` (last observation carried forward).
#' @return `data` with missing values in `cols` replaced.
#' @export
fill_missing <- function(data, cols,
                         method = c("zero", "mean", "median", "mode", "locf")) {
  method <- match.arg(method)
  check_known_cols(data, cols, "cols")

  for (col in cols) {
    value <- data[[col]]
    gaps <- is.na(value)
    # Columns without gaps are left as they are
    if (!any(gaps)) {
      next
    }
    fill <- switch(method,
      zero = {
        0
      },
      mean = {
        mean(value, na.rm = TRUE)
      },
      median = {
        stats::median(value, na.rm = TRUE)
      },
      mode = {
        counts <- table(value)
        type.convert(names(counts)[which.max(counts)], as.is = TRUE)
      },
      locf = {
        index <- cummax(ifelse(gaps, 0L, seq_along(value)))
        value[replace(index, index == 0L, NA)][gaps]
      }
    )

    value[gaps] <- fill
    data[[col]] <- value
  }
  data
}
