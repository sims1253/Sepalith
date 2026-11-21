registerS3method("format", "pb_tally", format.pb_tally)
registerS3method("format", "pb_roster", format.pb_roster)
registerS3method("format", "pb_ballot", format.pb_ballot)

tally <- structure(c(a = 3L, b = 1L), contest = "mayor", class = "pb_tally")
stopifnot(identical(format(tally), c("a: 3 (75.0%)", "b: 1 (25.0%)")))
stopifnot(identical(format(structure(c(a = 0L), contest = "", class = "pb_tally")), "a: 0 (0.0%)"))
stopifnot(identical(format(structure(integer(), contest = "", class = "pb_tally")), character()))

roster <- structure(list(voter = c("v1", "v2"), district = c("N", "S"),
  status = c("active", "pending")), class = "pb_roster")
stopifnot(identical(format(roster), c("v1 (N, active)", "v2 (S, pending)")))
empty.roster <- structure(list(voter = character(), district = character(),
  status = character()), class = "pb_roster")
stopifnot(identical(format(empty.roster), character()))

ballots <- structure(list(c("x", "y"), character(), "z"), options = c("x", "y", "z"),
  class = "pb_ballot")
stopifnot(identical(format(ballots), c("x > y", "(blank)", "z")))
stopifnot(identical(format(structure(list(), options = "x", class = "pb_ballot")), character()))

out <- capture.output(res <- print(tally))
stopifnot(identical(res, tally), identical(out, c("<pb_tally: mayor>", "  a: 3 (75.0%)",
  "  b: 1 (25.0%)")))
out <- capture.output(print(structure(c(a = 1L, b = 1L), contest = "", class = "pb_tally"), n = 1))
stopifnot(identical(out, c("<pb_tally: unnamed>", "  a: 1 (50.0%)", "  # ... with 1 more")))

out <- capture.output(res <- print(roster))
stopifnot(identical(res, roster), identical(out[1], "<pb_roster: 2 voters in 2 districts>"))
stopifnot(identical(capture.output(print(roster, n = 1))[3], "  # ... with 1 more"))
stopifnot(identical(capture.output(print(empty.roster)), "<pb_roster: 0 voters in 0 districts>"))

out <- capture.output(res <- print(ballots))
stopifnot(identical(res, ballots), identical(out[1], "<pb_ballot: 3 ballots over 3 options>"))
stopifnot(identical(out[3], "  (blank)"), length(capture.output(print(ballots, n = 2))) == 4L)
