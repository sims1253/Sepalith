#' Normalise white space
#'
#' @param x A character vector.
#' @param keep_newlines Keep line breaks, collapsing only other white space?
#' @return `x` with leading and trailing white space removed and inner runs
#'   collapsed to a single space (or a single line break).
#' @keywords internal
squish <- function(x, keep_newlines = FALSE) {
  if (keep_newlines) {
    x <- gsub("[ \t\r\f\v]+", " ", x)
    x <- gsub(" ?\n[[:space:]]*", "\n", x)
  } else {
    x <- gsub("[[:space:]]+", " ", x)
  }
  return(gsub("^[[:space:]]+|[[:space:]]+$", "", x))
}

#' Count values changed by a normaliser
#'
#' @param before,after Character vectors of equal length.
#' @return The number of positions where `before` and `after` differ, treating
#'   `NA` as a value of its own.
#' @keywords internal
count_changed <- function(before, after) {
  stopifnot(length(before) == length(after))
  if (length(before) == 0L) {
    return(0L)
  }
  same <- (is.na(before) & is.na(after)) |
    (!is.na(before) & !is.na(after) & before == after)
  return(sum(!same))
}

#' Normalise a contact field
#'
#' @param x A character vector of raw field values.
#' @param field The field the values belong to.
#' @param na_empty Turn empty results into `NA`?
#' @return A character vector the same length as `x`.
#' @export
normalise_field <- function(x, field = c("name", "email", "phone", "postcode", "country"),
                            na_empty = TRUE) {
  field <- match.arg(field)
  if (!is.character(x)) {
    x <- as.character(x)
  }
  labels <- names(x)
  x <- squish(x)

  out <- switch(field,
    name = {
      # Title case, keeping apostrophes and hyphens inside names
      value <- tolower(x)
      value <- gsub("(^|[ '-])([[:alpha:]])", "\\1\\U\\2", value, perl = TRUE)
      value
    },
    email = {
      # Lower case, without a mailto: scheme
      value <- tolower(x)
      value <- sub("^mailto:", "", value)
      value
    },
    phone = {
      # Digits only, international prefix as +
      value <- gsub("[^0-9+]", "", x)
      value <- sub("^00", "+", value)
      value
    },
    postcode = {
      # Upper case, without spaces
      value <- toupper(x)
      value <- gsub(" ", "", value, fixed = TRUE)
      value
    },
    country = {
      # Upper case, letters only
      value <- toupper(x)
      value <- gsub("[^A-Z]", "", value)
      value
    }
  )

  if (na_empty) {
    out[!is.na(out) & out == ""] <- NA_character_
  }
  names(out) <- labels
  return(out)
}

#' Normalise the contact fields of a table
#'
#' Applies [normalise_field()] to each of the columns `name`, `email`,
#' `phone`, `postcode` and `country` that is present in `data`, and records
#' how many values each field lost or changed.
#'
#' @param data A data frame of contact records.
#' @param na_empty Turn empty results into `NA`?
#' @return `data` with normalised columns and an attribute `changed`, a named
#'   integer vector with one count per normalised field.
#' @export
normalise_contacts <- function(data, na_empty = TRUE) {
  stopifnot(is.data.frame(data))
  changed <- integer()

  # Normalise `name`
  if ("name" %in% names(data)) {
    before <- data[["name"]]
    data[["name"]] <- normalise_field(before, "name", na_empty = na_empty)
    changed[["name"]] <- count_changed(before, data[["name"]])
  }

  # Normalise `email`
  if ("email" %in% names(data)) {
    before <- data[["email"]]
    data[["email"]] <- normalise_field(before, "email", na_empty = na_empty)
    changed[["email"]] <- count_changed(before, data[["email"]])
  }

  # Normalise `phone`
  if ("phone" %in% names(data)) {
    before <- data[["phone"]]
    data[["phone"]] <- normalise_field(before, "phone", na_empty = na_empty)
    changed[["phone"]] <- count_changed(before, data[["phone"]])
  }

  # Normalise `postcode`
  if ("postcode" %in% names(data)) {
    before <- data[["postcode"]]
    data[["postcode"]] <- normalise_field(before, "postcode", na_empty = na_empty)
    changed[["postcode"]] <- count_changed(before, data[["postcode"]])
  }

  # Normalise `country`
  if ("country" %in% names(data)) {
    before <- data[["country"]]
    data[["country"]] <- normalise_field(before, "country", na_empty = na_empty)
    changed[["country"]] <- count_changed(before, data[["country"]])
  }

  attr(data, "changed") <- changed
  return(data)
}

#' Split full names into given and family names
#'
#' Accepts both `"Given Family"` and `"Family, Given"`. Names with a single
#' word are treated as family names.
#'
#' @param x A character vector of full names.
#' @return A data frame with columns `given` and `family`.
#' @export
split_names <- function(x) {
  x <- squish(x)
  given <- rep(NA_character_, length(x))
  family <- rep(NA_character_, length(x))
  comma <- !is.na(x) & grepl(",", x, fixed = TRUE)
  plain <- !is.na(x) & !comma & x != ""

  # "Family, Given"
  family[comma] <- squish(sub(",.*$", "", x[comma]))
  given[comma] <- squish(sub("^[^,]*,", "", x[comma]))

  # "Given Family"
  family[plain] <- sub("^.* ", "", x[plain])
  given[plain] <- ifelse(grepl(" ", x[plain], fixed = TRUE),
                         sub(" [^ ]*$", "", x[plain]), NA_character_)

  given[!is.na(given) & given == ""] <- NA_character_
  family[!is.na(family) & family == ""] <- NA_character_
  return(data.frame(given = given, family = family, stringsAsFactors = FALSE))
}

#' Normalise dates written in several layouts
#'
#' Each layout in `formats` is tried in turn; a value keeps the first
#' successful parse.
#'
#' @param x A character vector of dates.
#' @param formats Candidate [strptime()] formats, in order of preference.
#' @return A character vector of ISO 8601 dates (`YYYY-MM-DD`), `NA` where no
#'   format matched. The attribute `format_used` holds the matching format.
#' @export
normalise_dates <- function(x, formats = c("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y%m%d")) {
  x <- squish(x)
  out <- rep(NA_character_, length(x))
  used <- rep(NA_character_, length(x))

  for (fmt in formats) {
    todo <- is.na(out) & !is.na(x)
    if (!any(todo)) {
      break
    }
    parsed <- as.Date(x[todo], format = fmt)
    # Reject partial parses such as "2024-01-0123"
    exact <- !is.na(parsed) & format(parsed, fmt) == x[todo]
    idx <- which(todo)[exact]
    out[idx] <- format(parsed[exact], "%Y-%m-%d")
    used[idx] <- fmt
  }
  attr(out, "format_used") <- used
  return(out)
}
