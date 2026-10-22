#' @title Split a recipient string into addresses
#'
#' @description
#' Registry entries store recipients as one string separated by commas or
#' semicolons. Whitespace and duplicates are dropped.
#'
#' @param recipients A single string of addresses.
#' @noRd
splitRecipients <- function(recipients) {
  if (length(recipients) == 0L || is.na(recipients) || !nzchar(recipients)) {
    return(character())
  }
  addresses <- trimws(unlist(strsplit(recipients, "[,;]")))
  unique(addresses[nzchar(addresses)])
}

#' @title Validate email addresses
#'
#' @description
#' A deliberately loose check that catches typos such as missing `@` or a
#' missing domain part, without trying to implement the full RFC.
#'
#' @param addresses A character vector of addresses.
#' @param strict If `TRUE`, invalid addresses raise an error; otherwise they
#'   are dropped with a warning.
#' @return The valid addresses.
#' @noRd
validateAddresses <- function(addresses, strict = FALSE) {
  ok <- grepl("^[^@[:space:]]+@[^@[:space:]]+\\.[A-Za-z]{2,}$", addresses)
  if (all(ok)) {
    return(addresses)
  }

  bad <- paste(addresses[!ok], collapse = ", ")
  if (strict) {
    stop("Invalid email address(es): ", bad, call. = FALSE)
  }
  warning("Dropping invalid email address(es): ", bad, call. = FALSE)
  addresses[ok]
}

#' @title Format the subject line for a report email
#'
#' @param reportId The registry id of the report.
#' @param status Either `"ok"` or `"error"`.
#' @param runTime The time the report was rendered.
#' @noRd
formatSubject <- function(reportId, status = c("ok", "error"), runTime = Sys.time()) {
  status <- match.arg(status)
  prefix <- switch(status, ok = "[report]", error = "[report FAILED]")
  sprintf("%s %s - %s", prefix, reportId, format(runTime, "%Y-%m-%d %H:%M"))
}

#' @title Format the Markdown body for a report email
#'
#' @param reportId The registry id of the report.
#' @param status Either `"ok"` or `"error"`.
#' @param details A one-row data frame returned by [renderDocument()].
#' @noRd
formatBody <- function(reportId, status = c("ok", "error"), details) {
  status <- match.arg(status)

  # Both variants share the engine line so the mails look alike in a list
  lines <- switch(status,
    ok = c(
      sprintf("The **%s** report rendered successfully.", reportId),
      "",
      sprintf("- Engine: %s", details$engine),
      sprintf("- Render time: %.1f s", details$seconds),
      "",
      "The rendered file is attached."
    ),
    error = c(
      sprintf("The **%s** report failed to render.", reportId),
      "",
      sprintf("- Engine: %s", details$engine),
      sprintf("- Error: `%s`", details$message),
      "",
      "No file was attached. The previous version is still available."
    )
  )
  paste(lines, collapse = "\n")
}

#' @title Compose a report email
#'
#' @description
#' Builds a blastula email from a Markdown body and attaches any files that
#' exist on disk.
#'
#' @param body A Markdown string, usually from [formatBody()].
#' @param attachments Paths of files to attach.
#' @return A `blastula_message` object.
#' @keywords internal
composeReportEmail <- function(body, attachments = character()) {
  email <- blastula::compose_email(
    body = blastula::md(body),
    footer = blastula::md(sprintf("Sent by rendershed on %s", Sys.info()[["nodename"]]))
  )

  # Missing files (e.g. NA output of a failed render) are skipped quietly
  for (path in attachments[file.exists(attachments)]) {
    email <- blastula::add_attachment(email, file = path)
  }
  email
}

#' @title Send a composed report email
#'
#' @description
#' Credentials are read from environment variables so that scheduled jobs
#' never hold passwords in their configuration.
#'
#' @param email A message created by [composeReportEmail()].
#' @param to A character vector of recipient addresses.
#' @param subject The subject line.
#' @param from The sender address.
#' @param dryRun If `TRUE`, print what would be sent instead of sending.
#' @return `TRUE` if the message was sent or simulated, invisibly.
#' @keywords internal
sendReportEmail <- function(email, to, subject,
                            from = Sys.getenv("RENDERSHED_FROM", "reports@example.org"),
                            dryRun = FALSE) {
  to <- validateAddresses(to)
  if (length(to) == 0L) {
    warning("No valid recipients; email '", subject, "' not sent.", call. = FALSE)
    return(invisible(FALSE))
  }
  if (dryRun) {
    message(sprintf("[dry run] '%s' to %s", subject, paste(to, collapse = ", ")))
    return(invisible(TRUE))
  }

  # The password stays in its env var; blastula reads it at send time
  creds <- blastula::creds_envvar(
    user = Sys.getenv("RENDERSHED_SMTP_USER"),
    pass_envvar = "RENDERSHED_SMTP_PASS",
    host = Sys.getenv("RENDERSHED_SMTP_HOST"),
    port = as.integer(Sys.getenv("RENDERSHED_SMTP_PORT", "587"))
  )
  blastula::smtp_send(email, to = to, from = from, subject = subject, credentials = creds)
  invisible(TRUE)
}

#' @title Notify recipients that a report succeeded
#'
#' @param reportId The registry id of the report.
#' @param details A one-row data frame returned by [renderDocument()].
#' @param recipients A single string of addresses.
#' @param dryRun If `TRUE`, print what would be sent instead of sending.
#' @return `TRUE` if the message was sent or simulated, invisibly.
#' @rdname notify
#' @export
notifySuccess <- function(reportId, details, recipients, dryRun = FALSE) {
  to <- splitRecipients(recipients)
  subject <- formatSubject(reportId, "ok")
  body <- formatBody(reportId, "ok", details)
  email <- composeReportEmail(body, attachments = details$output)
  sendReportEmail(email, to = to, subject = subject, dryRun = dryRun)
}

#' @title Notify recipients that a report failed
#'
#' @param reportId The registry id of the report.
#' @param details A one-row data frame returned by [renderDocument()].
#' @param recipients A single string of addresses.
#' @param dryRun If `TRUE`, print what would be sent instead of sending.
#' @rdname notify
#' @export
notifyFailure <- function(reportId, details, recipients, dryRun = FALSE) {
  to <- splitRecipients(recipients)
  subject <- formatSubject(reportId, "error")
  body <- formatBody(reportId, "error", details)
  email <- composeReportEmail(body)
  sendReportEmail(email, to = to, subject = subject, dryRun = dryRun)
}
