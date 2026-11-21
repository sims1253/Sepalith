# skipped: needs shiny (tabsUi, tabsServer)
q = data.frame(cell = c("m18", "f18", "m65"), target = c(100, 100, 50),
               achieved = c(80, 120, 50), weight = c(1, 1, 2))
s = summariseQuota(q)
stopifnot(identical(names(s), c("cells", "target", "achieved", "pctDone", "underCells",
                                "overCells")))
stopifnot(s$cells == 3L, s$target == 300, s$achieved == 280, all.equal(s$pctDone, 280 / 300))
stopifnot(identical(s$underCells, "m18"), identical(s$overCells, "f18"))
stopifnot(identical(summariseQuota(q, slack = 0.25)$underCells, character(0)))
stopifnot(inherits(try(summariseQuota(q[1:2]), silent = TRUE), "try-error"))

r = data.frame(
  interviewer = c("ann", "ann", "ben", "ben", "ben"),
  outcome = c("complete", "refusal", "complete", "complete", "noncontact"),
  date = as.Date(c("2024-03-04", "2024-03-05", "2024-03-06", "2024-03-12", "2024-04-02")),
  duration = c(600, 30, 900, 1100, NA),
  stringsAsFactors = FALSE
)
iv = summariseInterviewers(r)
stopifnot(identical(names(iv), c("interviewer", "completes", "refusals", "rate",
                                 "lastContact", "avgDuration")))
stopifnot(identical(iv$completes, c(1L, 2L)), identical(iv$refusals, c(1L, 0L)))
stopifnot(all.equal(iv$rate, c(0.5, 2 / 3)), identical(iv$avgDuration, c(600, 1000)))
stopifnot(identical(iv$lastContact, as.Date(c("2024-03-05", "2024-04-02"))))

d = timelineCounts(r, "day")
stopifnot(identical(d$completes, c(1L, 1L, 1L)), identical(d$period[1], as.Date("2024-03-04")))
w = timelineCounts(r, "week")
stopifnot(identical(w$period, as.Date(c("2024-03-04", "2024-03-11"))),
          identical(w$completes, c(2L, 1L)))
m = timelineCounts(r, "month")
stopifnot(identical(m$period, as.Date("2024-03-01")), identical(m$completes, 3L))
stopifnot(nrow(timelineCounts(r[r$outcome != "complete", ])) == 0L)

t0 = tabTitles(list())
stopifnot(identical(unname(t0), c("Overview", "Quotas", "Interviewers", "Timeline")))
t1 = tabTitles(list(quotas = 3, interviewers = 0, timeline = NA))
stopifnot(identical(t1[["quotas"]], "Quotas (3)"), identical(t1[["interviewers"]], "Interviewers"))
stopifnot(identical(t1[["timeline"]], "Timeline"))

stopifnot(nrow(tabFilterResponses(r)) == 5L)
stopifnot(identical(nrow(tabFilterResponses(r, interviewer = "ann")), 2L))
stopifnot(identical(tabFilterResponses(r, outcome = "complete")$duration, c(600, 900, 1100)))
stopifnot(identical(tabFilterResponses(r, from = "2024-03-06")$interviewer, c("ben", "ben",
                    "ben")))
stopifnot(identical(tabFilterResponses(r, to = "2024-03-05", interviewer = "ben"), r[0, ]))
