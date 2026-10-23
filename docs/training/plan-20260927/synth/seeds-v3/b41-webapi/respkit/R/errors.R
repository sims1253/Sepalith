#' Parse an error response body
#'
#' @description
#' Normalises the error formats we meet in practice into one shape.
#'
#' @param body Parsed error body (a nested list), or a string for `"plain"`.
#' @param style One of `"problem"` (RFC 9457 problem details), `"jsonapi"`
#'   (`errors[[1]]`), `"google"` (`error` object), `"graphql"`
#'   (`errors[[1]]` with `extensions`) or `"plain"`.
#' @return A list with elements `code`, `message` and `detail`; missing
#'   parts are `NA`.
#' @export
parse.error.body <- function(body,
                             style = c("problem", "jsonapi", "google", "graphql", "plain")) {
  style <- match.arg(style)
  chr <- function(x) if (is.null(x)) NA_character_ else as.character(x)[1]

  switch(style,
    problem = {
      list(
        code = chr(body[["type"]]),
        message = chr(body[["title"]]),
        detail = chr(body[["detail"]])
      )
    },
    jsonapi = {
      err <- if (length(body[["errors"]]) > 0L) body[["errors"]][[1]] else list()
      list(
        code = chr(err[["code"]]),
        message = chr(err[["title"]]),
        detail = chr(err[["detail"]])
      )
    },
    google = {
      err <- if (is.null(body[["error"]])) list() else body[["error"]]
      list(
        code = chr(err[["status"]]),
        message = chr(err[["message"]]),
        detail = chr(err[["details"]][[1]][["reason"]])
      )
    },
    graphql = {
      err <- if (length(body[["errors"]]) > 0L) body[["errors"]][[1]] else list()
      list(
        code = chr(err[["extensions"]][["code"]]),
        message = chr(err[["message"]]),
        detail = chr(paste(unlist(err[["path"]]), collapse = "."))
      )
    },
    plain = {
      text <- trimws(paste(as.character(body), collapse = " "))
      list(
        code = NA_character_,
        message = if (nzchar(text)) text else NA_character_,
        detail = NA_character_
      )
    }
  )
}

#' Field-level validation errors
#'
#' @param body Parsed error body (a nested list).
#' @param style One of `"jsonapi"` (`errors[[i]]$source$pointer`),
#'   `"rails"` (`errors$<field>` as message vectors), `"django"`
#'   (`<field>` as message vectors at the top level) or `"google"`
#'   (`error$details[[i]]$fieldViolations`).
#' @return A data frame with columns `field` and `message`, one row per error.
#' @export
validation.errors <- function(body, style = c("jsonapi", "rails", "django", "google")) {
  style <- match.arg(style)
  field <- character(0)
  message <- character(0)

  switch(style,
    jsonapi = {
      for (err in body[["errors"]]) {
        field <- c(field, sub("^/data/attributes/", "", err[["source"]][["pointer"]]))
        message <- c(message, err[["detail"]])
      }
    },
    rails = {
      for (nm in names(body[["errors"]])) {
        field <- c(field, rep(nm, length(body[["errors"]][[nm]])))
        message <- c(message, unlist(body[["errors"]][[nm]]))
      }
    },
    django = {
      for (nm in setdiff(names(body), "non_field_errors")) {
        field <- c(field, rep(nm, length(body[[nm]])))
        message <- c(message, unlist(body[[nm]]))
      }
    },
    google = {
      for (d in body[["error"]][["details"]]) {
        for (v in d[["fieldViolations"]]) {
          field <- c(field, v[["field"]])
          message <- c(message, v[["description"]])
        }
      }
    }
  )
  data.frame(field = field, message = message)
}

#' Turn an error response into a classed condition
#'
#' @description
#' The condition class follows the status: `respkit_bad_request` (400),
#' `respkit_unauthorized` (401), `respkit_forbidden` (403),
#' `respkit_not_found` (404), `respkit_conflict` (409),
#' `respkit_unprocessable` (422), `respkit_rate_limited` (429) and
#' `respkit_server_error` (5xx); other codes get `respkit_http_error` only.
#'
#' @param status HTTP status code.
#' @param error A parsed error from [parse.error.body()].
#' @param url Request URL, used in the message.
#' @return A condition object; signal it with [stop()].
#' @export
error.condition <- function(status, error, url = NA_character_) {
  classes <- c(
    "400" = "bad_request", "401" = "unauthorized", "403" = "forbidden",
    "404" = "not_found", "409" = "conflict", "422" = "unprocessable",
    "429" = "rate_limited"
  )
  key <- as.character(status)
  specific <- if (key %in% names(classes)) {
    paste0("respkit_", classes[[key]])
  } else if (status >= 500) {
    "respkit_server_error"
  } else {
    character(0)
  }
  text <- if (is.na(error$message)) "no message" else error$message
  if (!is.na(error$detail)) {
    text <- paste0(text, " (", error$detail, ")")
  }
  if (!is.na(url)) {
    text <- paste0(text, " [", url, "]")
  }

  structure(
    class = c(specific, "respkit_http_error", "error", "condition"),
    list(
      message = sprintf("HTTP %d: %s", status, text),
      call = NULL,
      status = status,
      code = error$code,
      url = url
    )
  )
}

#' Run the standard checks on a response
#'
#' @param resp A list with elements `status`, `content_type` and `body`
#'   (a raw vector).
#' @param status Accepted status codes.
#' @param type Accepted media type, or `NULL` for any.
#' @param max.bytes Largest accepted body size.
#' @param empty.ok Whether an empty body is accepted.
#' @return A named logical vector with one element per check: `status`,
#'   `type`, `size` and `empty`. Failing checks carry a `problems`
#'   attribute with one message each.
#' @export
check.response <- function(resp, status = 200:299, type = "application/json",
                           max.bytes = 1e7, empty.ok = FALSE) {
  checks <- c(status = TRUE, type = TRUE, size = TRUE, empty = TRUE)
  problems <- character(0)
  # Check `status`
  if (!resp$status %in% status) {
    checks[["status"]] <- FALSE
    problems <- c(problems, sprintf("Check `%s` failed: got %s.", "status", resp$status))
  }

  # Check `type`
  if (!is.null(type) && !identical(resp$content_type, type)) {
    checks[["type"]] <- FALSE
    problems <- c(problems, sprintf("Check `%s` failed: got %s.", "type", resp$content_type))
  }

  # Check `size`
  if (length(resp$body) > max.bytes) {
    checks[["size"]] <- FALSE
    problems <- c(problems, sprintf("Check `%s` failed: got %s.", "size", length(resp$body)))
  }

  # Check `empty`
  if (!empty.ok && length(resp$body) == 0L) {
    checks[["empty"]] <- FALSE
    problems <- c(problems, sprintf("Check `%s` failed: got %s.", "empty", length(resp$body)))
  }

  attr(checks, "problems") <- problems
  checks
}

#' Summarise a batch of failed requests
#'
#' @param status Integer vector of HTTP status codes.
#' @param url Character vector of request URLs, the same length as `status`.
#' @return A data frame with one row per status family (`client_auth`,
#'   `client_missing`, `client_other`, `rate_limited`, `server`) and columns
#'   `family`, `n` and `example` (one URL from that family).
#' @export
error.families <- function(status, url) {
  families <- c("client_auth", "client_missing", "client_other", "rate_limited", "server")
  n <- integer(length(families))
  example <- rep(NA_character_, length(families))
  names(n) <- names(example) <- families
  # Family `client_auth`
  hit <- status %in% c(401L, 403L)
  n[["client_auth"]] <- sum(hit)
  if (any(hit)) example[["client_auth"]] <- url[hit][1]
  # Family `client_missing`
  hit <- status %in% c(404L, 410L)
  n[["client_missing"]] <- sum(hit)
  if (any(hit)) example[["client_missing"]] <- url[hit][1]
  # Family `client_other`
  hit <- status >= 400L & status < 500L & !status %in% c(401L, 403L, 404L, 410L, 429L)
  n[["client_other"]] <- sum(hit)
  if (any(hit)) example[["client_other"]] <- url[hit][1]
  # Family `rate_limited`
  hit <- status == 429L
  n[["rate_limited"]] <- sum(hit)
  if (any(hit)) example[["rate_limited"]] <- url[hit][1]
  # Family `server`
  hit <- status >= 500L
  n[["server"]] <- sum(hit)
  if (any(hit)) example[["server"]] <- url[hit][1]
  data.frame(family = families, n = unname(n), example = unname(example))
}
