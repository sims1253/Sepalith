# skipped: needs openssl (computeSignature, verifySignature) and jsonlite
# (handleWebhook)
stopifnot(
  isTRUE(safeEquals("abc123", "abc123")),
  isFALSE(safeEquals("abc123", "abc124")),
  isFALSE(safeEquals("\u00e9", "a")),
  isFALSE(safeEquals("\u00e9", "ab")),
  isFALSE(safeEquals(NA_character_, "a")),
  isFALSE(safeEquals("a", NULL))
)

stopifnot(
  isTRUE(verifyTimestamp("1000", tolerance = 300, now = 1200)),
  isFALSE(verifyTimestamp(1000, tolerance = 300, now = 1400)),
  isFALSE(verifyTimestamp("yesterday", now = 1000)),
  isFALSE(verifyTimestamp(NULL))
)

issue <- list(id = 7, title = "Crash", user = list(login = "ana"), html_url = "https://g.ex/7")
git <- normalizeGitEvent(list(action = "opened", issue = issue), "issues")
stopifnot(
  identical(git$kind, "issue"),
  identical(git$externalId, "7"),
  is.null(normalizeGitEvent(list(action = "closed", issue = list(id = 7)), "issues")),
  is.null(normalizeGitEvent(list(action = "opened"), "push"))
)

form <- normalizeFormEvent(
  list(data = list(submission_id = 12, fields = list(subject = "Hi", email = "A@B.org"))),
  "submission.created"
)
stopifnot(
  identical(form$actor, "a@b.org"),
  is.null(normalizeFormEvent(list(data = list(fields = list(subject = "Hi"))), 
                             "submission.created")),
  is.null(normalizeFormEvent(list(), "submission.deleted"))
)

alert <- normalizeAlertEvent(
  list(state = "firing", fingerprint = "f1", labels = list(severity = "critical"),
       annotations = list(summary = "disk full")),
  "alert"
)
bare <- normalizeAlertEvent(list(state = "firing", fingerprint = "f2"), "alert")
stopifnot(
  identical(alert$kind, "incident"),
  identical(alert$title, "[CRITICAL] disk full"),
  identical(bare$title, "[WARNING] alert"),
  identical(bare$actor, "monitoring"),
  is.null(normalizeAlertEvent(list(state = "resolved"), "alert"))
)
