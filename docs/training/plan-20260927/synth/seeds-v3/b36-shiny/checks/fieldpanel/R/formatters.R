stopifnot(identical(formatRate(c(0.256, NA)), c("25.6%", "-")))
stopifnot(identical(formatRate(0.0123, "permille"), "12.3 per mille"))
stopifnot(identical(formatRate(0.25, "fraction"), "0.250"))
stopifnot(identical(formatRate(c(0.25, 0.001), "ratio"), c("1 in 4", "1 in 1,000")))

p = formatProgress(80, 100)
stopifnot(identical(p, list(pct = 0.8, label = "80 of 100 (20 short)", status = "under")))
stopifnot(identical(formatProgress(120, 100)$label, "120 of 100 (20 over)"))
stopifnot(identical(formatProgress(101, 100)$status, "on"))
stopifnot(identical(formatProgress(5, 0)$label, "no target"))

cnt = formatCells(c(1500, NA), "count")
stopifnot(identical(cnt$text, c("1,500", "-")), identical(cnt$align, "right"))
stopifnot(identical(formatCells(0.5, "rate")$text, "50.0%"))
dt = formatCells(as.Date(c("2024-02-03", NA)), "date", na = "")
stopifnot(identical(dt$text, c("03 Feb 2024", "")), identical(dt$sortKey[1], 19756))
stopifnot(identical(formatCells(c(65, 600), "duration")$text, c("1:05", "10:00")))
tx = formatCells(c("Bob", "alice"), "text")
stopifnot(identical(tx$align, "left"), identical(tx$sortKey, c("bob", "alice")))

s = formatWaveSummary(list(waveId = "WV001", startDate = as.Date("2024-03-01"),
                           endDate = as.Date("2024-03-31"), targetN = 1200,
                           completes = 640, responseRate = 0.42))
stopifnot(identical(names(s), c("waveId", "startDate", "endDate", "targetN", "completes",
                                "responseRate")))
stopifnot(identical(s[["waveId"]], "Wave: WV001"), identical(s[["endDate"]], "End: 31 Mar 2024"))
stopifnot(identical(s[["targetN"]], "Target: 1,200"))
stopifnot(identical(s[["responseRate"]], "Response rate: 42.0%"))

iv = data.frame(interviewer = c("Ann", "Ben", "Cy"), completes = c(5, 12, 5),
                refusals = c(1, 0, NA), rate = c(0.5, 0.8, 0.4),
                lastContact = as.Date(c("2024-03-02", "2024-03-05", NA)),
                avgDuration = c(900, 1260, 75))
ft = formatInterviewerTable(iv)
stopifnot(identical(ft$table$interviewer, c("Ben", "Ann", "Cy")))
stopifnot(identical(ft$table$refusals, c("0", "1", "-")),
          identical(ft$table$avgDuration[3], "1:15"))
stopifnot(identical(unname(ft$align), c("left", rep("right", 5))))
stopifnot(inherits(try(formatInterviewerTable(iv[1:3]), silent = TRUE), "try-error"))

oc = formatOutcomeCounts(c("complete", "complete", "refusal", "noncontact", NA))
stopifnot(identical(oc$outcome, c("complete", "partial", "refusal", "noncontact", "ineligible")))
stopifnot(identical(oc$n, c(2L, 0L, 1L, 1L, 0L)))
stopifnot(identical(oc$share, c("50.0%", "0.0%", "25.0%", "25.0%", "0.0%")))
stopifnot(identical(formatOutcomeCounts(c("partial", "ineligible"), digits = 0)$share[c(2, 5)],
                    c("50%", "50%")))
stopifnot(all(formatOutcomeCounts(character(0))$share == "-"))
stopifnot(inherits(try(formatOutcomeCounts("hangup"), silent = TRUE), "try-error"))
