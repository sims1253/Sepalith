summ <- data.frame(student = c("a", "b", "c"), sessions = c(10L, 10L, 8L),
                   attended.rate = c(0.7, 0.92, 1))
roster <- data.frame(
  student = c("a", "b", "c"), year = c("y7", "", NA), group = c("g1", "g1", NA),
  program = c(NA, "arts", "arts"), support = c(TRUE, NA, FALSE)
)

fr <- fill.roster(roster)
stopifnot(identical(fr$year, c("y7", "unknown", "unknown")), identical(fr$group[3], "unassigned"))
stopifnot(identical(fr$program, c("general", "arts", "arts")))
stopifnot(identical(fr$support, c(TRUE, FALSE, FALSE)))
stopifnot(identical(attr(fr, "filled"), c(year = 2L, group = 1L, program = 1L, support = 1L)))
defs <- list(year = "Y1", group = "g", program = "p", support = TRUE)
bare <- fill.roster(data.frame(student = "z"), defaults = defs)
stopifnot(identical(bare$year, "Y1"), identical(bare$group, "g"), identical(bare$program, "p"))
stopifnot(identical(bare$support, TRUE))

cb <- cohort.breakdown(summ, fr)
stopifnot(nrow(cb) == 8L)
stopifnot(identical(unique(cb$dimension), c("year", "group", "program", "support")))
yr <- cb[cb$dimension == "year", ]
stopifnot(identical(yr$level, c("unknown", "y7")), identical(yr$students, c(2L, 1L)))
stopifnot(identical(yr$mean.rate, c(0.96, 0.7)), identical(yr$below, c(0L, 1L)))
gr <- cb[cb$dimension == "group", ]
stopifnot(identical(gr$level, c("g1", "unassigned")), identical(gr$mean.rate, c(0.81, 1)))
pr <- cb[cb$dimension == "program", ]
stopifnot(identical(pr$level, c("arts", "general")), identical(pr$below, c(0L, 1L)))
sp <- cb[cb$dimension == "support", ]
stopifnot(identical(sp$level, c("FALSE", "TRUE")), identical(sp$students, c(2L, 1L)))
stopifnot(identical(cohort.breakdown(summ, fr, threshold = 1)$below[1:2], c(1L, 1L)))

fl <- data.frame(student = c("a", "b", "c"), n.flags = c(2, 0, 0))
rl <- report.lines(summ, fl)
stopifnot(length(rl) == 17L, identical(rl[1:3], c("## overview", "3 students, 28 sessions", "")))
stopifnot(identical(rl[5:7], c("a: 70.0%", "b: 92.0%", "c: 100.0%")))
stopifnot(identical(rl[10], "a: 2 flag(s)"))
stopifnot(identical(rl[14:17], c("## bands", "a: red", "b: amber", "c: green")))
stopifnot(identical(report.lines(summ, fl, "flags"),
                    c("## flags", "a: 2 flag(s)", "b: 0 flag(s)", "c: 0 flag(s)")))
stopifnot(identical(report.lines(summ, fl, c("bands", "overview"))[1], "## overview"))

out.dir <- file.path(tempfile("presentia-"), "out")
tabs <- list(summary = summ, breakdown = cb)
for (fmt in c("csv", "tsv", "rds")) {
  paths <- export.report(tabs, out.dir, fmt)
  stopifnot(all(file.exists(paths)), identical(names(paths), c("summary", "breakdown")))
  back <- import.report(out.dir, fmt)
  stopifnot(identical(names(back), c("breakdown", "summary")))
  stopifnot(isTRUE(all.equal(back$summary, summ, check.attributes = FALSE)))
  stopifnot(identical(back$breakdown$level, cb$level))
}
txt <- export.report(list(summary = summ), out.dir, "txt")
stopifnot(file.exists(txt), length(readLines(txt)) == 4L)
unlink(dirname(out.dir), recursive = TRUE)

# skipped: needs dplyr (weekly.cohort)
