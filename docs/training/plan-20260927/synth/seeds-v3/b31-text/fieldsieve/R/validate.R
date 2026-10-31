#' Check field values against their expected shape
#'
#' Values are expected to be normalised already, see `normalise_field()`.
#'
#' @param x A character vector of normalised field values.
#' @param field The field the values belong to.
#' @return A logical vector the same length as `x`: `TRUE` for well-formed
#'   values, `FALSE` for malformed ones and `NA` for missing ones.
#' @export
is_valid <- function(x, field = c("email", "url", "phone", "postcode", "birth_date",
                                 "country")) {
  field <- match.arg(field)
  if (!is.character(x)) {
    x <- as.character(x)
  }

  # Each arm anchors its pattern to the whole value
  ok <- switch(field,
    email = {
      pattern <- "^[[:alnum:]._%+-]+@[[:alnum:]-]+(\\.[[:alnum:]-]+)*\\.[[:alpha:]]{2,}$"
      grepl(pattern, x, perl = TRUE)
    },
    url = {
      pattern <- "^https?://[[:alnum:]-]+(\\.[[:alnum:]-]+)*(:[0-9]+)?(/[^[:space:]]*)?$"
      grepl(pattern, x, perl = TRUE)
    },
    phone = {
      pattern <- "^\\+?[0-9]{6,15}$"
      grepl(pattern, x, perl = TRUE)
    },
    postcode = {
      pattern <- "^[A-Z0-9]{3,8}$"
      grepl(pattern, x, perl = TRUE)
    },
    birth_date = {
      pattern <- "^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$"
      grepl(pattern, x, perl = TRUE) & !is.na(as.Date(x, format = "%Y-%m-%d"))
    },
    country = {
      pattern <- "^[A-Z]{2}$"
      grepl(pattern, x, perl = TRUE)
    }
  )

  ok[is.na(x)] <- NA
  return(ok)
}

#' List invalid contact fields
#'
#' Checks the columns `email`, `phone`, `postcode`, `birth_date` and
#' `country` of a normalised contact table. Missing values are not reported.
#'
#' @param data A data frame of contact records.
#' @return A data frame with columns `row`, `field` and `value`, one row per
#'   invalid value, ordered by field and then by row.
#' @export
find_invalid <- function(data) {
  stopifnot(is.data.frame(data))
  # One block per field: find malformed values and record them
  issues <- list()
  if ("email" %in% names(data)) {
    bad <- which(is_valid(data[["email"]], "email") %in% FALSE)
    issues$email <- data.frame(row = bad, field = rep("email", length(bad)),
                               value = data[["email"]][bad])
  }

  if ("phone" %in% names(data)) {
    bad <- which(is_valid(data[["phone"]], "phone") %in% FALSE)
    issues$phone <- data.frame(row = bad, field = rep("phone", length(bad)),
                               value = data[["phone"]][bad])
  }

  if ("postcode" %in% names(data)) {
    bad <- which(is_valid(data[["postcode"]], "postcode") %in% FALSE)
    issues$postcode <- data.frame(row = bad, field = rep("postcode", length(bad)),
                                  value = data[["postcode"]][bad])
  }

  if ("birth_date" %in% names(data)) {
    bad <- which(is_valid(data[["birth_date"]], "birth_date") %in% FALSE)
    issues$birth_date <- data.frame(row = bad, field = rep("birth_date", length(bad)),
                                    value = data[["birth_date"]][bad])
  }

  if ("country" %in% names(data)) {
    bad <- which(is_valid(data[["country"]], "country") %in% FALSE)
    issues$country <- data.frame(row = bad, field = rep("country", length(bad)),
                                 value = data[["country"]][bad])
  }

  if (length(issues) == 0L) {
    return(data.frame(row = integer(), field = character(), value = character()))
  }
  out <- do.call(rbind, unname(issues))
  rownames(out) <- NULL
  return(out)
}

#' Summarise field validity
#'
#' @param data A data frame of contact records.
#' @param fields The fields to summarise; fields missing from `data` are
#'   skipped.
#' @return A data frame with one row per field and columns `field`, `n`,
#'   `n_missing`, `n_invalid` and `share_valid`.
#' @export
validity_summary <- function(data, fields = c("email", "phone", "postcode", "birth_date",
                                               "country")) {
  fields <- intersect(fields, names(data))
  n <- integer(length(fields))
  n_missing <- integer(length(fields))
  n_invalid <- integer(length(fields))

  for (j in seq_along(fields)) {
    ok <- is_valid(data[[fields[j]]], fields[j])
    n[j] <- length(ok)
    n_missing[j] <- sum(is.na(ok))
    n_invalid[j] <- sum(ok %in% FALSE)
  }

  checked <- n - n_missing
  share_valid <- ifelse(checked > 0L, (checked - n_invalid) / checked, NA_real_)
  return(data.frame(
    field = fields,
    n = n,
    n_missing = n_missing,
    n_invalid = n_invalid,
    share_valid = share_valid
  ))
}

#' Describe invalid fields in plain text
#'
#' @param issues A data frame as returned by [find_invalid()].
#' @param max_lines Maximum number of lines to report; the rest are
#'   summarised in a final line.
#' @return A character vector of messages, one per reported issue.
#' @export
describe_invalid <- function(issues, max_lines = 10L) {
  if (nrow(issues) == 0L) {
    return(character())
  }
  msgs <- sprintf("Row %d: `%s` is not a valid %s.", issues$row, issues$value,
                  gsub("_", " ", issues$field, fixed = TRUE))
  if (length(msgs) > max_lines) {
    extra <- length(msgs) - max_lines
    msgs <- c(msgs[seq_len(max_lines)], sprintf("... and %d more.", extra))
  }
  return(msgs)
}

#' Check the arguments of a text window request
#'
#' @param x A character vector of documents.
#' @param min_chars Smallest accepted window width in characters.
#' @param max_chars Largest accepted window width in characters.
#' @param before Number of characters kept before a match.
#' @param after Number of characters kept after a match.
#' @return `TRUE`, invisibly, when every argument is valid; otherwise an
#'   error naming the first offending argument.
#' @keywords internal
check_window_args <- function(x, min_chars, max_chars, before, after) {
  if (!is.character(x)) {
    stop("`x` must be a character vector.", call. = FALSE)
  }

  # Validate `min_chars`
  if (!is.numeric(min_chars) || length(min_chars) != 1L || is.na(min_chars)) {
    stop("`min_chars` must be a single number.", call. = FALSE)
  }
  if (min_chars < 0) {
    stop("`min_chars` must not be negative.", call. = FALSE)
  }

  # Validate `max_chars`
  if (!is.numeric(max_chars) || length(max_chars) != 1L || is.na(max_chars)) {
    stop("`max_chars` must be a single number.", call. = FALSE)
  }
  if (max_chars < min_chars) {
    stop("`max_chars` must not be smaller than `min_chars`.", call. = FALSE)
  }

  # Validate `before`
  if (!is.numeric(before) || length(before) != 1L || is.na(before)) {
    stop("`before` must be a single number.", call. = FALSE)
  }
  if (before < 0) {
    stop("`before` must not be negative.", call. = FALSE)
  }

  # Validate `after`
  if (!is.numeric(after) || length(after) != 1L || is.na(after)) {
    stop("`after` must be a single number.", call. = FALSE)
  }
  if (after < 0) {
    stop("`after` must not be negative.", call. = FALSE)
  }
  return(invisible(TRUE))
}

#' Cut a context window around the first match
#'
#' @param x A character vector of documents.
#' @param pattern A regular expression.
#' @param before Number of characters kept before a match.
#' @param after Number of characters kept after a match.
#' @return A character vector the same length as `x`, `NA` where `pattern`
#'   does not match.
#' @export
match_window <- function(x, pattern, before = 10L, after = 10L) {
  check_window_args(x, 0, before + after, before, after)
  pos <- regexpr(pattern, x, perl = TRUE)
  len <- attr(pos, "match.length")
  out <- rep(NA_character_, length(x))
  hit <- !is.na(pos) & pos > 0L

  # Left edge
  left <- pmax(1L, pos[hit] - before)
  # Right edge
  right <- pmin(nchar(x[hit]), pos[hit] + len[hit] - 1L + after)

  out[hit] <- substr(x[hit], left, right)
  return(out)
}
