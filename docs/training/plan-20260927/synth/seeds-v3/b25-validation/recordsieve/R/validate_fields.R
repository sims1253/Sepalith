#' Validate e-mail addresses
#'
#' @param x Character vector of values.
#' @param allowMissing Whether `NA` counts as valid.
#' @return A logical vector the same length as `x`.
#' @family field validators
#' @export
isValidEmail <- function(x, allowMissing = FALSE) {
  x <- trimws(as.character(x))
  ok <- grepl("^[^@[:space:]]+@[^@[:space:]]+\\.[A-Za-z]{2,}$", x)
  ok[is.na(x)] <- allowMissing
  ok
}

#' Validate phone numbers
#'
#' @param x Character vector of values.
#' @param allowMissing Whether `NA` counts as valid.
#' @return A logical vector the same length as `x`.
#' @family field validators
#' @export
isValidPhone <- function(x, allowMissing = FALSE) {
  x <- gsub("[[:space:]()/-]", "", as.character(x))
  ok <- grepl("^\\+?[0-9]{7,15}$", x)
  ok[is.na(x)] <- allowMissing
  ok
}

#' Validate ISO dates
#'
#' @param x Character vector of values.
#' @param allowMissing Whether `NA` counts as valid.
#' @return A logical vector the same length as `x`.
#' @family field validators
#' @export
isValidIsoDate <- function(x, allowMissing = FALSE) {
  x <- trimws(as.character(x))
  ok <- grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", x)
  ok[ok] <- !is.na(as.Date(x[ok], format = "%Y-%m-%d"))
  ok[is.na(x)] <- allowMissing
  ok
}

#' Validate postcodes
#'
#' @param x Character vector of values.
#' @param country Country whose postcode format applies: one of `"DE"`,
#'   `"NL"`, `"UK"` or `"US"`.
#' @param allowMissing Whether `NA` counts as valid.
#' @return A logical vector the same length as `x`.
#' @family field validators
#' @export
isValidPostcode <- function(x, country = c("DE", "NL", "UK", "US"), allowMissing = FALSE) {
  country <- match.arg(country)
  x <- toupper(trimws(as.character(x)))
  pattern <- switch(country,
    DE = "^[0-9]{5}$",
    NL = "^[1-9][0-9]{3} ?[A-Z]{2}$",
    UK = "^[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}$",
    US = "^[0-9]{5}(-[0-9]{4})?$"
  )
  ok <- grepl(pattern, x)
  ok[is.na(x)] <- allowMissing
  ok
}

#' Collect issues for one field
#'
#' @param field Field name.
#' @param values Field values.
#' @param ok Logical vector, `FALSE` where a value is invalid.
#' @param problem Description of the problem.
#' @return A data frame with columns `row`, `field`, `value` and `problem`,
#'   one row per invalid value.
#' @keywords internal
#' @noRd
fieldIssues <- function(field, values, ok, problem) {
  bad <- which(!ok)
  data.frame(
    row = bad,
    field = rep(field, length(bad)),
    value = as.character(values[bad]),
    problem = rep(problem, length(bad)),
    stringsAsFactors = FALSE
  )
}

#' Validate a table of contact records
#'
#' Checks each of the contact fields `id`, `email`, `phone`, `postcode` and
#' `birthDate` with its field validator and stacks the issues.
#'
#' @param data A data frame of contact records.
#' @param country Country whose postcode format applies.
#' @param optional Fields that may be missing.
#' @return A data frame with columns `row`, `field`, `value` and `problem`,
#'   ordered by `row`.
#' @family record validators
#' @export
validateContacts <- function(data, country = "DE", optional = c("phone", "birthDate")) {
  fields <- c("id", "email", "phone", "postcode", "birthDate")
  absent <- setdiff(fields, names(data))
  if (length(absent) > 0L) {
    stop("`data` lacks column(s): ", paste(absent, collapse = ", "), call. = FALSE)
  }

  issues <- list()

  # Field `id`
  okId <- !is.na(data$id) & !duplicated(data$id)
  issues[["id"]] <- fieldIssues("id", data$id, okId, "missing or duplicated")

  # Field `email`
  okEmail <- isValidEmail(data$email, allowMissing = "email" %in% optional)
  issues[["email"]] <- fieldIssues("email", data$email, okEmail, "invalid format")

  # Field `phone`
  okPhone <- isValidPhone(data$phone, allowMissing = "phone" %in% optional)
  issues[["phone"]] <- fieldIssues("phone", data$phone, okPhone, "invalid format")

  # Field `postcode`
  okPostcode <- isValidPostcode(data$postcode, country, allowMissing = "postcode" %in% optional)
  issues[["postcode"]] <- fieldIssues("postcode", data$postcode, okPostcode, "invalid format")

  # Field `birthDate`
  okBirthDate <- isValidIsoDate(data$birthDate, allowMissing = "birthDate" %in% optional)
  issues[["birthDate"]] <- fieldIssues("birthDate", data$birthDate, okBirthDate, "invalid format")

  out <- do.call(rbind, unname(issues))
  out <- out[order(out$row, match(out$field, fields)), , drop = FALSE]
  rownames(out) <- NULL
  out
}

#' Validate one contact record
#'
#' Checks the fields `id`, `email`, `phone`, `postcode` and `birthDate` of a
#' single record, as entered in a form, and explains every failure.
#'
#' @param record A named list holding one contact record.
#' @param country Country whose postcode format applies.
#' @return A list with elements `ok` (a named logical vector with one
#'   element per field) and `problems` (a named character vector with one
#'   message per invalid field).
#' @family record validators
#' @export
validateContact <- function(record, country = "DE") {
  get <- function(field) if (is.null(record[[field]])) NA_character_ else record[[field]]
  ok <- c(id = TRUE, email = TRUE, phone = TRUE, postcode = TRUE, birthDate = TRUE)
  problems <- character()

  # Field `id`
  if (is.na(get("id")) || !nzchar(get("id"))) {
    ok[["id"]] <- FALSE
    problems[["id"]] <- "`id` is required."
  }

  # Field `email`
  if (!isValidEmail(get("email"))) {
    ok[["email"]] <- FALSE
    problems[["email"]] <- sprintf("`email` has an invalid format: %s", get("email"))
  }

  # Field `phone`
  if (!isValidPhone(get("phone"), allowMissing = TRUE)) {
    ok[["phone"]] <- FALSE
    problems[["phone"]] <- sprintf("`phone` has an invalid format: %s", get("phone"))
  }

  # Field `postcode`
  if (!isValidPostcode(get("postcode"), country)) {
    ok[["postcode"]] <- FALSE
    problems[["postcode"]] <- sprintf("`postcode` has an invalid format: %s", get("postcode"))
  }

  # Field `birthDate`
  if (!isValidIsoDate(get("birthDate"), allowMissing = TRUE)) {
    ok[["birthDate"]] <- FALSE
    problems[["birthDate"]] <- sprintf("`birthDate` has an invalid format: %s",
                                       get("birthDate"))
  }
  list(ok = ok, problems = problems)
}
