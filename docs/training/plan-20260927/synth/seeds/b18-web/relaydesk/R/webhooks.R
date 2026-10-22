#' Compute an HMAC signature for a webhook payload
#'
#' Hashes the raw request body with the shared secret and returns the
#' lower-case hex digest.
#'
#' @param payload The raw request body as a single string.
#' @param secret The shared signing secret.
#' @param algo Hash algorithm, either `"sha256"` or `"sha1"`.
#' @return A single hex string.
#' @noRd
computeSignature <- function(payload, secret, algo = c("sha256", "sha1")) {
  algo <- match.arg(algo)
  key <- charToRaw(secret)
  body <- charToRaw(enc2utf8(payload))
  digest <- switch(algo,
    sha256 = openssl::sha256(body, key = key),
    sha1 = openssl::sha1(body, key = key)
  )
  tolower(as.character(digest))
}

#' Compare two strings in constant time
#'
#' Avoids leaking how many leading characters match, which an early-exit
#' comparison would reveal through timing.
#'
#' @param a,b Single strings to compare.
#' @return `TRUE` when the strings are identical.
#' @noRd
safeEquals <- function(a, b) {
  if (!is.character(a) || !is.character(b) || nchar(a) != nchar(b)) {
    return(FALSE)
  }
  x <- as.integer(charToRaw(a))
  y <- as.integer(charToRaw(b))
  diff <- 0L
  for (i in seq_along(x)) {
    diff <- bitwOr(diff, bitwXor(x[i], y[i]))
  }
  diff == 0L
}

#' Verify the signature header of a webhook request
#'
#' Strips an optional algorithm prefix such as `sha256=` from the header
#' and compares it with the expected signature.
#'
#' @param payload The raw request body as a single string.
#' @param header The value of the signature header, or `NULL`.
#' @param secret The shared signing secret.
#' @param algo Hash algorithm, either `"sha256"` or `"sha1"`.
#' @return `TRUE` when the signature is valid.
#' @noRd
verifySignature <- function(payload, header, secret, algo = c("sha256", "sha1")) {
  algo <- match.arg(algo)
  if (is.null(header) || !nzchar(header) || !nzchar(secret)) {
    return(FALSE)
  }
  given <- sub(paste0("^", algo, "="), "", tolower(trimws(header)))
  expected <- computeSignature(payload, secret, algo)
  safeEquals(given, expected)
}

#' Check that a webhook timestamp is recent
#'
#' Rejects deliveries whose timestamp is too far from the current time,
#' which limits the window for replaying captured requests.
#'
#' @param timestamp Unix time in seconds, as a string or number.
#' @param tolerance Allowed clock difference in seconds.
#' @param now Current Unix time in seconds.
#' @return `TRUE` when the timestamp is within the tolerance.
#' @noRd
verifyTimestamp <- function(timestamp, tolerance = 300, now = as.numeric(Sys.time())) {
  if (is.null(timestamp) || length(timestamp) != 1) {
    return(FALSE)
  }
  sent <- suppressWarnings(as.numeric(timestamp))
  if (is.na(sent)) {
    return(FALSE)
  }
  abs(now - sent) <= tolerance
}

#' Normalize a git host event
#'
#' Maps an issue or pull request event from the git host to the common
#' event shape used by the ticket store.
#'
#' @param payload The parsed webhook payload as a named list.
#' @param eventType The event name taken from the request headers.
#' @return A list with `source`, `kind`, `externalId`, `title`, `actor` and
#'   `url`, or `NULL` for events that are ignored.
#' @noRd
normalizeGitEvent <- function(payload, eventType) {
  item <- switch(eventType,
    issues = payload$issue,
    pull_request = payload$pull_request,
    NULL
  )
  if (is.null(item) || !identical(payload$action, "opened")) {
    return(NULL)
  }
  list(
    source = "git",
    kind = if (eventType == "issues") "issue" else "review",
    externalId = as.character(item$id),
    title = item$title,
    actor = item$user$login,
    url = item$html_url
  )
}

#' Normalize a contact form event
#'
#' Maps a submission from the hosted contact form to the common event
#' shape used by the ticket store.
#'
#' @param payload The parsed webhook payload as a named list.
#' @param eventType The event name taken from the request headers.
#' @return A list with `source`, `kind`, `externalId`, `title`, `actor` and
#'   `url`, or `NULL` for events that are ignored.
#' @noRd
normalizeFormEvent <- function(payload, eventType) {
  if (!identical(eventType, "submission.created")) {
    return(NULL)
  }
  fields <- payload$data$fields
  if (is.null(fields$subject) || is.null(fields$email)) {
    return(NULL)
  }
  list(
    source = "form",
    kind = "request",
    externalId = as.character(payload$data$submission_id),
    title = fields$subject,
    actor = tolower(fields$email),
    url = payload$data$admin_url
  )
}

#' Normalize a monitoring alert event
#'
#' Maps a firing alert from the monitoring service to the common event
#' shape used by the ticket store. Resolved alerts are ignored.
#'
#' @param payload The parsed webhook payload as a named list.
#' @param eventType The event name taken from the request headers.
#' @return A list with `source`, `kind`, `externalId`, `title`, `actor` and
#'   `url`, or `NULL` for events that are ignored.
#' @noRd
normalizeAlertEvent <- function(payload, eventType) {
  if (!identical(eventType, "alert") || !identical(payload$state, "firing")) {
    return(NULL)
  }
  severity <- payload$labels$severity %||% "warning"
  list(
    source = "alert",
    kind = if (severity == "critical") "incident" else "alert",
    externalId = as.character(payload$fingerprint),
    title = sprintf("[%s] %s", toupper(severity), payload$annotations$summary),
    actor = payload$receiver %||% "monitoring",
    url = payload$generatorURL
  )
}

#' Parse and dispatch an incoming webhook
#'
#' Verifies the signature, parses the JSON body and hands it to the
#' normalizer registered for the source.
#'
#' @param source One of `"git"`, `"form"` or `"alert"`.
#' @param req The plumber request object.
#' @param secrets Named list of signing secrets, one per source.
#' @return A list with `status` (`"accepted"`, `"ignored"` or `"rejected"`),
#'   `reason` and `event`.
#' @noRd
handleWebhook <- function(source = c("git", "form", "alert"), req, secrets) {
  source <- match.arg(source)
  raw <- req$postBody %||% ""
  headers <- req$HEADERS
  signature <- switch(source,
    git = headers[["x-hub-signature-256"]],
    form = headers[["x-form-signature"]],
    alert = headers[["x-alert-signature"]]
  )
  if (!verifySignature(raw, signature, secrets[[source]] %||% "")) {
    return(list(status = "rejected", reason = "bad signature", event = NULL))
  }
  if (source == "form" && !verifyTimestamp(headers[["x-form-timestamp"]])) {
    return(list(status = "rejected", reason = "stale timestamp", event = NULL))
  }
  payload <- tryCatch(
    jsonlite::fromJSON(raw, simplifyVector = FALSE),
    error = function(e) NULL
  )
  if (is.null(payload)) {
    return(list(status = "rejected", reason = "invalid JSON", event = NULL))
  }
  eventType <- switch(source,
    git = headers[["x-github-event"]] %||% "",
    form = payload$type %||% "",
    alert = "alert"
  )
  event <- switch(source,
    git = normalizeGitEvent(payload, eventType),
    form = normalizeFormEvent(payload, eventType),
    alert = normalizeAlertEvent(payload, eventType)
  )
  if (is.null(event)) {
    return(list(status = "ignored", reason = paste("unhandled event", eventType), event = NULL))
  }
  list(status = "accepted", reason = NULL, event = event)
}
