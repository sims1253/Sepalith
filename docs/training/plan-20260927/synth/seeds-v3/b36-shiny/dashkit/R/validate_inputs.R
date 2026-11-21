#' Check a date-range filter
#'
#' Validates both ends of a date-range input against the dates that are
#' available in the sales data.
#'
#' @param start First day selected in the date-range input.
#' @param end Last day selected in the date-range input.
#' @param min_date Earliest date available in the data.
#' @param max_date Latest date available in the data.
#' @return `NULL` when the range is valid, otherwise a single string
#'   describing the first problem found.
#' @family input checks
#' @export
check_date_range <- function(start, end, min_date, max_date) {
  # Validate `start`
  if (length(start) != 1L || is.na(start)) {
    return(sprintf("`%s` must be a single date.", "start"))
  }
  start <- as.Date(start)
  if (start < min_date) {
    return(sprintf("`%s` must not be before %s.", "start", format(min_date)))
  }

  # Validate `end`
  if (length(end) != 1L || is.na(end)) {
    return(sprintf("`%s` must be a single date.", "end"))
  }
  end <- as.Date(end)
  if (end > max_date) {
    return(sprintf("`%s` must not be after %s.", "end", format(max_date)))
  }

  if (start > end) {
    return("`start` must not be after `end`.")
  }
  NULL
}

#' Check a two-handled slider value
#'
#' @param value Numeric vector of length two, as returned by a range slider.
#' @param name Name of the input, used in messages.
#' @param lower Smallest value the slider may take.
#' @param upper Largest value the slider may take.
#' @return `NULL` when the value is valid, otherwise a single string
#'   describing the first problem found.
#' @family input checks
#' @export
check_slider_range <- function(value, name, lower, upper) {
  if (!is.numeric(value) || length(value) != 2L || anyNA(value)) {
    return(sprintf("`%s` must be two numbers.", name))
  }
  low <- value[[1]]
  high <- value[[2]]

  # Lower handle
  if (low < lower) {
    return(sprintf("The lower end of `%s` must be at least %s.", name, lower))
  }
  if (low > upper) {
    return(sprintf("The lower end of `%s` must be at most %s.", name, upper))
  }

  # Upper handle
  if (high < lower) {
    return(sprintf("The upper end of `%s` must be at least %s.", name, lower))
  }
  if (high > upper) {
    return(sprintf("The upper end of `%s` must be at most %s.", name, upper))
  }

  if (low > high) {
    return(sprintf("The ends of `%s` are in the wrong order.", name))
  }
  NULL
}

#' Check the categorical filter inputs
#'
#' Each filter must have at least one selection, and every selection must be
#' one of the offered choices.
#'
#' @param region Selected regions.
#' @param channel Selected sales channels.
#' @param segment Selected customer segments.
#' @param product Selected product lines.
#' @param choices A named list of allowed values with elements `region`,
#'   `channel`, `segment` and `product`.
#' @return A named character vector with one message per failing filter;
#'   empty when every filter is valid.
#' @family input checks
#' @export
check_filter_inputs <- function(region, channel, segment, product, choices) {
  problems <- character(0)

  # Validate `region`
  if (length(region) == 0L) {
    problems[["region"]] <- sprintf("Select at least one `%s`.", "region")
  } else if (length(unknown <- setdiff(region, choices$region)) > 0L) {
    problems[["region"]] <- sprintf("Unknown `%s`: %s.", "region",
                                    paste(unknown, collapse = ", "))
  }

  # Validate `channel`
  if (length(channel) == 0L) {
    problems[["channel"]] <- sprintf("Select at least one `%s`.", "channel")
  } else if (length(unknown <- setdiff(channel, choices$channel)) > 0L) {
    problems[["channel"]] <- sprintf("Unknown `%s`: %s.", "channel",
                                     paste(unknown, collapse = ", "))
  }

  # Validate `segment`
  if (length(segment) == 0L) {
    problems[["segment"]] <- sprintf("Select at least one `%s`.", "segment")
  } else if (length(unknown <- setdiff(segment, choices$segment)) > 0L) {
    problems[["segment"]] <- sprintf("Unknown `%s`: %s.", "segment",
                                     paste(unknown, collapse = ", "))
  }

  # Validate `product`
  if (length(product) == 0L) {
    problems[["product"]] <- sprintf("Select at least one `%s`.", "product")
  } else if (length(unknown <- setdiff(product, choices$product)) > 0L) {
    problems[["product"]] <- sprintf("Unknown `%s`: %s.", "product",
                                     paste(unknown, collapse = ", "))
  }
  problems
}

#' Check the numeric planning inputs
#'
#' @param target Revenue target for the period.
#' @param budget Marketing budget for the period.
#' @param threshold Alert threshold as a share between 0 and 1.
#' @param horizon Forecast horizon in weeks.
#' @return A named character vector with one message per failing input;
#'   empty when every input is valid.
#' @family input checks
#' @export
check_numeric_inputs <- function(target, budget, threshold, horizon) {
  limits <- list(
    target = c(0, 1e9),
    budget = c(0, 1e7),
    threshold = c(0, 1),
    horizon = c(1, 52)
  )
  problems <- character(0)

  # Validate `target`
  if (!is.numeric(target) || length(target) != 1L || is.na(target)) {
    problems[["target"]] <- sprintf("`%s` must be a single number.", "target")
  } else if (target < limits$target[1] || target > limits$target[2]) {
    problems[["target"]] <- sprintf("`%s` must lie between %s and %s.", "target",
                                    limits$target[1], limits$target[2])
  }

  # Validate `budget`
  if (!is.numeric(budget) || length(budget) != 1L || is.na(budget)) {
    problems[["budget"]] <- sprintf("`%s` must be a single number.", "budget")
  } else if (budget < limits$budget[1] || budget > limits$budget[2]) {
    problems[["budget"]] <- sprintf("`%s` must lie between %s and %s.", "budget",
                                    limits$budget[1], limits$budget[2])
  }

  # Validate `threshold`
  if (!is.numeric(threshold) || length(threshold) != 1L || is.na(threshold)) {
    problems[["threshold"]] <- sprintf("`%s` must be a single number.", "threshold")
  } else if (threshold < limits$threshold[1] || threshold > limits$threshold[2]) {
    problems[["threshold"]] <- sprintf("`%s` must lie between %s and %s.", "threshold",
                                       limits$threshold[1], limits$threshold[2])
  }

  # Validate `horizon`
  if (!is.numeric(horizon) || length(horizon) != 1L || is.na(horizon)) {
    problems[["horizon"]] <- sprintf("`%s` must be a single number.", "horizon")
  } else if (horizon < limits$horizon[1] || horizon > limits$horizon[2]) {
    problems[["horizon"]] <- sprintf("`%s` must lie between %s and %s.", "horizon",
                                     limits$horizon[1], limits$horizon[2])
  }
  problems
}

#' Check an uploaded file
#'
#' Inspects the data frame that a file input returns for a single upload.
#'
#' @param file_info A one-row data frame with columns `name`, `size`, `type`
#'   and `datapath`.
#' @param max_mb Largest accepted file size in megabytes.
#' @param extensions Accepted file extensions, without the dot.
#' @return A character vector of problems, empty when the upload is usable.
#' @family input checks
#' @export
check_upload <- function(file_info, max_mb = 5, extensions = c("csv", "tsv")) {
  if (is.null(file_info) || nrow(file_info) == 0L) {
    return("No file was uploaded.")
  }
  problems <- character(0)

  # Check `name`
  ext <- tolower(tools::file_ext(file_info$name))
  if (!ext %in% extensions) {
    problems <- c(problems, sprintf("`%s` must end in one of: %s.", "name",
                                    paste(extensions, collapse = ", ")))
  }

  # Check `size`
  size_mb <- file_info$size / 1024^2
  if (is.na(size_mb) || size_mb > max_mb) {
    problems <- c(problems, sprintf("`%s` must be at most %s MB.", "size", max_mb))
  }

  # Check `type`
  if (!is.na(file_info$type) && !grepl("^text/", file_info$type)) {
    problems <- c(problems, sprintf("`%s` must be a text format.", "type"))
  }

  # Check `datapath`
  if (!file.exists(file_info$datapath)) {
    problems <- c(problems, sprintf("`%s` does not point to a file.", "datapath"))
  }
  problems
}

#' Check the free-text inputs of the report form
#'
#' @param title Report title.
#' @param note Note shown under the report title.
#' @param email Address the report is sent to.
#' @param tags Comma-separated list of tags.
#' @param max_chars Largest number of characters accepted per input.
#' @return A named character vector with one message per failing input;
#'   empty when every input is valid.
#' @family input checks
#' @export
check_text_inputs <- function(title, note, email, tags, max_chars = 200L) {
  problems <- character(0)

  # Validate `title`
  title <- trimws(title)
  if (length(title) != 1L || is.na(title) || !nzchar(title)) {
    problems[["title"]] <- sprintf("`%s` must not be empty.", "title")
  } else if (nchar(title) > max_chars) {
    problems[["title"]] <- sprintf("`%s` must have at most %d characters.", "title", max_chars)
  }

  # Validate `note`
  note <- trimws(note)
  if (length(note) != 1L || is.na(note)) {
    problems[["note"]] <- sprintf("`%s` must be a single string.", "note")
  } else if (nchar(note) > max_chars) {
    problems[["note"]] <- sprintf("`%s` must have at most %d characters.", "note", max_chars)
  }

  # Validate `email`
  email <- trimws(email)
  if (length(email) != 1L || is.na(email) || !nzchar(email)) {
    problems[["email"]] <- sprintf("`%s` must not be empty.", "email")
  } else if (!grepl("^[^@[:space:]]+@[^@[:space:]]+\\.[a-z]+$", email)) {
    problems[["email"]] <- sprintf("`%s` must be an e-mail address.", "email")
  }

  # Validate `tags`
  tags <- trimws(unlist(strsplit(tags, ",", fixed = TRUE)))
  if (any(!nzchar(tags))) {
    problems[["tags"]] <- sprintf("`%s` must not contain empty entries.", "tags")
  } else if (sum(nchar(tags)) > max_chars) {
    problems[["tags"]] <- sprintf("`%s` must have at most %d characters.", "tags", max_chars)
  }
  problems
}

#' Combine check results into one message
#'
#' @param ... Results of the `check_*()` functions: `NULL` or character
#'   vectors of messages.
#' @param sep Separator placed between messages.
#' @return `NULL` when there are no messages, otherwise a single string.
#' @keywords internal
collect_problems <- function(..., sep = "\n") {
  messages <- unlist(list(...), use.names = FALSE)
  messages <- messages[!is.na(messages) & nzchar(messages)]
  if (length(messages) == 0L) {
    return(NULL)
  }
  paste(unique(messages), collapse = sep)
}
