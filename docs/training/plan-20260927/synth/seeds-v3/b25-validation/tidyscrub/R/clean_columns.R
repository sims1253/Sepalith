#' Strings treated as missing
#'
#' @return A character vector of tokens that the cleaners turn into `NA`.
#' @keywords internal
default.na.tokens <- function() {
  c("", "NA", "N/A", "n/a", "NULL", "null", "-", "?")
}

#' Clean a column according to its type
#'
#' @description
#' `scrub()` is a generic with methods for numeric, character and factor
#' columns; other columns are returned unchanged. Every method returns a
#' vector of the same length and class as its input.
#'
#' @param x A column vector.
#' @param ... Passed on to methods.
#' @return The cleaned vector.
#' @export
scrub <- function(x, ...) {
  UseMethod("scrub")
}

#' @param na.values Sentinel numbers that mean "missing".
#' @rdname scrub
#' @export
scrub.numeric <- function(x, na.values = c(-99, -999, 9999), ...) {
  x[x %in% na.values] <- NA
  x[is.nan(x)] <- NA
  x[is.infinite(x)] <- NA
  x
}

#' @param na.tokens Strings that mean "missing".
#' @param squish Whether to trim and collapse inner whitespace.
#' @rdname scrub
#' @export
scrub.character <- function(x, na.tokens = default.na.tokens(), squish = TRUE, ...) {
  if (squish) {
    x <- gsub("[[:space:]]+", " ", trimws(x))
  }
  x[x %in% na.tokens] <- NA
  x
}

#' @rdname scrub
#' @export
scrub.factor <- function(x, na.tokens = default.na.tokens(), squish = TRUE, ...) {
  lev <- scrub.character(levels(x), na.tokens = na.tokens, squish = squish)
  values <- lev[as.integer(x)]
  factor(values, levels = unique(lev[!is.na(lev)]))
}

#' @rdname scrub
#' @export
scrub.default <- function(x, ...) {
  x
}

#' Coerce a character column to a target type
#'
#' @param x A character vector.
#' @param to Target type: one of `"numeric"`, `"integer"`, `"logical"`,
#'   `"date"`, `"factor"` or `"character"`.
#' @param dec Decimal mark used in `x`.
#' @details Values that cannot be converted become `NA`; the positions are
#'   stored in the attribute `"failed"`.
#' @return A vector of the target type.
#' @export
coerce.column <- function(x,
                          to = c("numeric", "integer", "logical", "date", "factor",
                                 "character"),
                          dec = ".") {
  to <- match.arg(to)
  x <- as.character(x)
  out <- switch(to,
    numeric = {
      digits <- if (dec == ",") gsub(".", "", x, fixed = TRUE) else gsub(",", "", x)
      suppressWarnings(as.numeric(sub(dec, ".", digits, fixed = TRUE)))
    },
    integer = {
      num <- suppressWarnings(as.numeric(gsub(",", "", x)))
      num[!is.na(num) & num != round(num)] <- NA
      as.integer(num)
    },
    logical = {
      key <- tolower(trimws(x))
      ifelse(key %in% c("true", "t", "yes", "y", "1"), TRUE,
             ifelse(key %in% c("false", "f", "no", "n", "0"), FALSE, NA))
    },
    date = {
      parsed <- as.Date(x, format = "%Y-%m-%d")
      retry <- is.na(parsed) & !is.na(x)
      parsed[retry] <- as.Date(x[retry], format = "%d.%m.%Y")
      parsed
    },
    factor = {
      factor(x, levels = unique(x[!is.na(x)]))
    },
    character = {
      x
    }
  )
  attr(out, "failed") <- which(is.na(out) & !is.na(x))
  out
}

#' Clean every column of a data frame
#'
#' @description
#' Runs the column cleaning pipeline: scrub each column by type, coerce the
#' columns listed in `types`, and drop rows that end up entirely missing.
#'
#' @param data A data frame to clean.
#' @param types Named character vector mapping column names to target types
#'   for [coerce.column()].
#' @param drop.empty Whether to drop rows in which every value is missing.
#' @return A list with elements `data` (the cleaned data frame), `changed`
#'   (number of values changed per column), `failed` (positions that failed
#'   coercion, per coerced column) and `dropped` (indices of dropped rows).
#' @export
clean.frame <- function(data, types = character(), drop.empty = TRUE) {
  unknown <- setdiff(names(types), names(data))
  if (length(unknown) > 0L) {
    stop("`types` names unknown column(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  before <- data

  # 1. Scrub every column by its type
  data[] <- lapply(data, function(column) scrub(column))

  # 2. Coerce the columns listed in `types`
  failed <- list()
  for (col in names(types)) {
    converted <- coerce.column(data[[col]], to = types[[col]])
    failed[[col]] <- attr(converted, "failed")
    attr(converted, "failed") <- NULL
    data[[col]] <- converted
  }

  # 3. Count changed values per column
  changed <- vapply(names(data), function(col) {
    old <- as.character(before[[col]])
    new <- as.character(data[[col]])
    sum(xor(is.na(old), is.na(new)) | (!is.na(old) & !is.na(new) & old != new))
  }, integer(1))

  # 4. Drop rows that are entirely missing
  dropped <- integer()
  if (drop.empty && nrow(data) > 0L) {
    dropped <- which(rowSums(!is.na(data)) == 0L)
    if (length(dropped) > 0L) {
      data <- data[-dropped, , drop = FALSE]
    }
  }
  list(data = data, changed = changed, failed = failed, dropped = dropped)
}
