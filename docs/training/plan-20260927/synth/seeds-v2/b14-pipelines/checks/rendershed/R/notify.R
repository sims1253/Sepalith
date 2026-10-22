# skipped: needs blastula (composeReportEmail, sendReportEmail, notifySuccess,
#   notifyFailure)

# splitRecipients()
stopifnot(
  identical(splitRecipients("a@x.org; b@x.org, a@x.org ,"), c("a@x.org", "b@x.org")),
  identical(splitRecipients(""), character()),
  identical(splitRecipients(NA_character_), character()),
  identical(splitRecipients(NULL), character())
)

# validateAddresses()
good <- c("ops@example.org", "lead@harbour.example.de")
stopifnot(
  identical(validateAddresses(good), good),
  identical(suppressWarnings(validateAddresses(c(good, "nobody"))), good),
  inherits(try(validateAddresses("x@y", strict = TRUE), silent = TRUE), "try-error")
)

# formatSubject()
t0 <- as.POSIXct("2026-03-02 08:15:00", tz = "UTC")
stopifnot(
  identical(formatSubject("berths", "ok", t0), "[report] berths - 2026-03-02 08:15"),
  identical(formatSubject("berths", "error", t0), "[report FAILED] berths - 2026-03-02 08:15")
)

# formatBody()
details <- data.frame(engine = "quarto", seconds = 12.34, message = "pandoc failed")
ok_body <- formatBody("berths", "ok", details)
err_body <- formatBody("berths", "error", details)
stopifnot(
  grepl("- Render time: 12.3 s", ok_body, fixed = TRUE),
  grepl("- Error: `pandoc failed`", err_body, fixed = TRUE),
  grepl("^The \\*\\*berths\\*\\* report failed", err_body),
  inherits(try(formatBody("berths", "maybe", details), silent = TRUE), "try-error")
)
