#' Validate a required string field
#'
#' Checks that a request value is a single non-empty string no longer than
#' `maxChars` characters.
#'
#' @param value The raw value taken from the request.
#' @param name Field name used in error messages.
#' @param maxChars Maximum number of characters allowed.
#' @return A list with the cleaned `value` and an `error` string or `NULL`.
#' @noRd
requireString = function(value, name, maxChars = 200L) {
  if (is.null(value) || length(value) != 1 || is.na(value)) {
    return(list(value = NULL, error = sprintf("`%s` is required", name)))
  }
  value = trimws(as.character(value))
  if (!nzchar(value)) {
    return(list(value = NULL, error = sprintf("`%s` must not be empty", name)))
  }
  if (nchar(value) > maxChars) {
    return(list(value = NULL, error = sprintf("`%s` exceeds %d characters", name, maxChars)))
  }
  list(value = value, error = NULL)
}

#' Validate a required integer field
#'
#' Checks that a request value can be read as a whole number within the
#' closed range from `min` to `max`.
#'
#' @param value The raw value taken from the request.
#' @param name Field name used in error messages.
#' @param min,max Inclusive bounds for the value.
#' @return A list with the cleaned `value` and an `error` string or `NULL`.
#' @noRd
requireInteger = function(value, name, min = 1L, max = .Machine$integer.max) {
  if (is.null(value) || length(value) != 1 || is.na(value)) {
    return(list(value = NULL, error = sprintf("`%s` is required", name)))
  }
  number = suppressWarnings(as.numeric(value))
  if (is.na(number) || number != round(number)) {
    return(list(value = NULL, error = sprintf("`%s` must be a whole number", name)))
  }
  if (number < min || number > max) {
    return(list(value = NULL, error = sprintf("`%s` must be between %d and %d", name, min, max)))
  }
  list(value = as.integer(number), error = NULL)
}

#' Validate a required choice field
#'
#' Checks that a request value is one of a fixed set of choices, ignoring
#' case and surrounding whitespace.
#'
#' @param value The raw value taken from the request.
#' @param name Field name used in error messages.
#' @param choices Character vector of allowed values.
#' @return A list with the cleaned `value` and an `error` string or `NULL`.
#' @noRd
requireChoice = function(value, name, choices) {
  if (is.null(value) || length(value) != 1 || is.na(value)) {
    return(list(value = NULL, error = sprintf("`%s` is required", name)))
  }
  value = tolower(trimws(as.character(value)))
  if (!value %in% choices) {
    allowed = paste(choices, collapse = ", ")
    return(list(value = NULL, error = sprintf("`%s` must be one of: %s", name, allowed)))
  }
  list(value = value, error = NULL)
}

#' Validate a required email field
#'
#' Applies a deliberately loose pattern check: one `@`, a dot in the domain
#' and no whitespace.
#'
#' @param value The raw value taken from the request.
#' @param name Field name used in error messages.
#' @return A list with the cleaned `value` and an `error` string or `NULL`.
#' @noRd
requireEmail = function(value, name) {
  checked = requireString(value, name, maxChars = 254L)
  if (!is.null(checked$error)) {
    return(checked)
  }
  value = tolower(checked$value)
  if (!grepl("^[^@[:space:]]+@[^@[:space:]]+\\.[^@[:space:]]+$", value)) {
    return(list(value = NULL, error = sprintf("`%s` is not a valid email address", name)))
  }
  list(value = value, error = NULL)
}

#' Make an optional field validator
#'
#' Wraps a `require*()` validator so that a missing or empty value yields
#' the default instead of an error.
#'
#' @param validator One of the `require*()` validators.
#' @param default Value used when the field is absent.
#' @return A function with the same signature as `validator`.
#' @noRd
optionalField = function(validator, default = NULL) {
  force(validator)
  function(value, name, ...) {
    if (is.null(value) || length(value) == 0 || identical(value, "")) {
      return(list(value = default, error = NULL))
    }
    validator(value, name, ...)
  }
}

#' Validate the body of a new ticket request
#'
#' Runs every field validator, collecting all error messages rather than
#' stopping at the first one, so clients can fix everything in one go.
#'
#' @param body The parsed JSON body as a named list.
#' @param priorities Character vector of allowed priority levels.
#' @return A list with `ok`, `errors` (named character vector) and `data`.
#' @noRd
validateTicketInput = function(body, priorities = c("low", "normal", "high", "urgent")) {
  if (!is.list(body)) {
    return(list(ok = FALSE, errors = c(body = "request body must be a JSON object"), data = NULL))
  }
  optionalString = optionalField(requireString, default = "")
  optionalChoice = optionalField(requireChoice, default = "normal")
  checks = list(
    title = requireString(body$title, "title", maxChars = 120L),
    description = optionalString(body$description, "description", maxChars = 5000L),
    reporter = requireEmail(body$reporter, "reporter"),
    priority = optionalChoice(body$priority, "priority", choices = priorities),
    queue = requireString(body$queue, "queue", maxChars = 40L)
  )
  errors = character()
  data = list()
  for (field in names(checks)) {
    check = checks[[field]]
    if (is.null(check$error)) {
      data[[field]] = check$value
    } else {
      errors[[field]] = check$error
    }
  }
  # tags are optional but must be short slugs when given
  tags = unlist(body$tags %||% character())
  badTags = tags[!grepl("^[a-z0-9][a-z0-9-]{0,29}$", tags)]
  if (length(badTags) > 0) {
    errors[["tags"]] = sprintf("invalid tags: %s", paste(badTags, collapse = ", "))
  } else {
    data$tags = unique(tags)
  }
  list(ok = length(errors) == 0, errors = errors, data = if (length(errors) == 0) data)
}

#' Validate paging parameters
#'
#' Reads `page` and `perPage` query parameters, applying defaults and an
#' upper bound on the page size.
#'
#' @param page The raw `page` query parameter.
#' @param perPage The raw `perPage` query parameter.
#' @param maxPerPage Largest page size a client may request.
#' @return A list with `ok`, `errors` (named character vector) and `data`.
#' @noRd
validatePaging = function(page, perPage, maxPerPage = 100L) {
  optionalInteger = optionalField(requireInteger, default = NULL)
  checks = list(
    page = optionalInteger(page, "page", min = 1L),
    perPage = optionalInteger(perPage, "perPage", min = 1L, max = maxPerPage)
  )
  errors = character()
  data = list(page = 1L, perPage = 25L)
  for (field in names(checks)) {
    check = checks[[field]]
    if (!is.null(check$error)) {
      errors[[field]] = check$error
    } else if (!is.null(check$value)) {
      data[[field]] = check$value
    }
  }
  list(ok = length(errors) == 0, errors = errors, data = if (length(errors) == 0) data)
}
