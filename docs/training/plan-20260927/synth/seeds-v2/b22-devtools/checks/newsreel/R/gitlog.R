# skipped: git_log, git_tags, git_log_since_tag and git_diff_stat need a git
# repository with history

# git_run fails loudly outside a repository (or without git at all)
stopifnot(inherits(try(git_run("status", repo = tempfile()), silent = TRUE), "try-error"))

log <- data.frame(
  hash = c("a1", "b2", "c3"),
  author = c("Ana", "Ben", "Ana"),
  email = c("ana@x.org", "ben@x.org", "ana@x.org"),
  date = as.POSIXct(c("2026-01-05 10:00", "2026-01-06 09:00", "2026-01-14 12:00"), tz = "UTC"),
  subject = c("feat: one", "fix: two", "docs: three"),
  stringsAsFactors = FALSE
)

# summarise_by_author
a <- summarise_by_author(log)
stopifnot(identical(a$author, c("Ana", "Ben")), identical(a$commits, c(2L, 1L)))
stopifnot(identical(a$last[1], log$date[3]))

# summarise_by_week
w <- summarise_by_week(log)
stopifnot(identical(w$week, c("2026-W02", "2026-W03")), identical(w$commits, c(2L, 1L)))
stopifnot(identical(w$subjects[1], "feat: one; fix: two"), w$authors[1] == 2L)
log$date[3] <- NA
stopifnot(sum(summarise_by_week(log)$commits) == 2L)
