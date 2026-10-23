lints <- data.frame(
  line = c(3L, 1L, 1L, 5L, 2L),
  column = c(1L, 3L, 9L, 2L, 4L),
  rule = c("browser_call", "assign_eq", "semicolon", "true_false", "assign_eq"),
  severity = c("error", "style", "style", "warning", "style"),
  message = c("No browser.", "Use <-.", "No ;.", "Use TRUE.", "Use <-."),
  stringsAsFactors = FALSE
)
none <- lints[0, ]

sc <- severityCounts(lints)
stopifnot(identical(sc$severity, c("error", "warning", "style", "info")))
stopifnot(identical(sc$n, c(1L, 1L, 3L, 0L)), identical(sc$lines, c(1L, 1L, 2L, 0L)))
stopifnot(identical(sc$rules, c("browser_call", "true_false", "assign_eq, semicolon", "")))
stopifnot(all(severityCounts(none)$n == 0L))
bad <- transform(lints, severity = "fatal")
stopifnot(inherits(tryCatch(severityCounts(bad), error = identity), "error"))

pl <- formatLints(lints, "a.R")
stopifnot(identical(pl[1], "a.R:3:1: E No browser. [browser_call]"))
stopifnot(identical(pl[4], "a.R:5:2: W Use TRUE. [true_false]"))
gh <- formatLints(lints, "a.R", "github")
stopifnot(identical(gh[2], "::notice file=a.R,line=1,col=3::Use <-. [assign_eq]"))
stopifnot(startsWith(gh[1], "::error "), startsWith(gh[4], "::warning "))
cs <- formatLints(lints, "a.R", "checkstyle")
stopifnot(length(cs) == 9L, identical(cs[2], "  <file name=\"a.R\">"))
stopifnot(identical(cs[3], paste0("    <error line=\"3\" column=\"1\" severity=\"error\"",
  " message=\"No browser.\" source=\"browser_call\"/>")), grepl("severity=\"info\"", cs[4]))
md <- formatLints(lints, "a.R", "markdown")
stopifnot(length(md) == 9L, identical(md[5], "| 3:1 | error | browser_call | No browser. |"))
csv <- formatLints(transform(lints, message = "say \"hi\""), "a.R", "csv")
stopifnot(identical(csv[2], "a.R,3,1,error,browser_call,\"say \"\"hi\"\"\""), length(csv) == 6L)
stopifnot(length(formatLints(none, "a.R")) == 0L)

rp <- reportBySeverity(lints)
stopifnot(identical(rp[1], "5 lints"), identical(rp[3], "Errors (1)"))
stopifnot(identical(rp[6], "Warnings (1)"), identical(rp[9], "Style (3)"))
stopifnot(identical(rp[10], "  1:3 Use <-. [assign_eq]"), length(rp) == 12L)
stopifnot(!any(grepl("^Notes", rp)), identical(reportBySeverity(none), "0 lints"))
info <- transform(lints[1, ], severity = "info")
stopifnot(identical(reportBySeverity(info)[3], "Notes (1)"))
path <- tempfile()
stopifnot(identical(reportBySeverity(lints, path), rp), identical(readLines(path), rp))
unlink(path)

cmp <- compareSeverity(lints, lints[c(2, 4, 4), ])
stopifnot(identical(cmp$table$before, c(1L, 1L, 3L, 0L)))
stopifnot(identical(cmp$table$after, c(0L, 2L, 1L, 0L)))
stopifnot(identical(cmp$table$trend, c("better", "worse", "better", "same")))
stopifnot(identical(cmp$fixed, c("browser_call", "semicolon")))
stopifnot(identical(cmp$introduced, character()))
stopifnot(identical(compareSeverity(none, lints)$introduced,
  c("assign_eq", "browser_call", "semicolon", "true_false")))

v <- lintVerdict(lints)
stopifnot(!v$pass, v$status == 1L, identical(v$reasons, "1 error lints, budget 0"))
v2 <- lintVerdict(lints, "style", budget = c(error = 1, warning = 0, style = 2, info = 0))
stopifnot(identical(v2$reasons, c("1 warning lints, budget 0", "3 style lints, budget 2")))
stopifnot(lintVerdict(lints[2, ], "warning")$pass, lintVerdict(none, "info")$status == 0L)
stopifnot(identical(lintVerdict(info, "info")$reasons, "1 info lints, budget 0"))
