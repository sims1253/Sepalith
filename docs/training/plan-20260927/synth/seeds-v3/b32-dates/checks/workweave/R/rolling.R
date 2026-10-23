isWeekend = function(x, weekend = c(6L, 7L)) {
  as.integer(format(as.Date(x), "%u")) %in% weekend
}
isBusinessDay = function(x, holidays = as.Date(character()), weekend = c(6L, 7L)) {
  !isWeekend(x, weekend) & !(as.Date(x) %in% as.Date(holidays))
}
nextBusinessDay = function(x, holidays = as.Date(character()), weekend = c(6L, 7L)) {
  vapply(as.Date(x), function(d) {
    d = d + 1L
    while (!isBusinessDay(d, holidays, weekend)) d = d + 1L
    as.numeric(d)
  }, numeric(1)) |> as.Date(origin = "1970-01-01")
}
previousBusinessDay = function(x, holidays = as.Date(character()), weekend = c(6L, 7L)) {
  vapply(as.Date(x), function(d) {
    d = d - 1L
    while (!isBusinessDay(d, holidays, weekend)) d = d - 1L
    as.numeric(d)
  }, numeric(1)) |> as.Date(origin = "1970-01-01")
}

sat = as.Date("2024-08-31")
hol = as.Date("2024-09-02")

stopifnot(rollFollowing(sat, hol) == as.Date("2024-09-03"))
stopifnot(rollFollowing(as.Date("2024-08-30")) == as.Date("2024-08-30"))
stopifnot(rollPreceding(sat, hol) == as.Date("2024-08-30"))
stopifnot(identical(rollPreceding(as.Date(c(NA, "2024-09-02")), hol),
  as.Date(c(NA, "2024-08-30"))))

stopifnot(adjustDate(sat) == sat)
stopifnot(adjustDate(sat, "following", hol) == as.Date("2024-09-03"))
stopifnot(adjustDate(sat, "preceding", hol) == as.Date("2024-08-30"))
stopifnot(adjustDate(sat, "modifiedFollowing", hol) == as.Date("2024-08-30"))
stopifnot(adjustDate(as.Date("2024-08-17"), "modifiedFollowing") == as.Date("2024-08-19"))
stopifnot(adjustDate(as.Date("2024-09-01"), "modifiedPreceding", hol) ==
  as.Date("2024-09-03"))
stopifnot(adjustDate(as.Date("2024-08-18"), "modifiedPreceding") == as.Date("2024-08-16"))
stopifnot(adjustDate(as.Date("2024-08-17"), "nearest") == as.Date("2024-08-16"))
stopifnot(adjustDate(as.Date("2024-08-18"), "nearest") == as.Date("2024-08-19"))
stopifnot(adjustDate(sat, "nearest", hol) == as.Date("2024-08-30"))

s = adjustSchedule(as.Date(c("2024-08-30", "2024-08-31", "2024-09-01")), "following", hol)
stopifnot(identical(names(s), c("original", "adjusted", "shift", "moved")))
stopifnot(identical(s$shift, c(0L, 3L, 2L)), identical(s$moved, c(FALSE, TRUE, TRUE)))

stopifnot(monthBusinessDay(as.Date("2024-09-15"), "first", hol) == as.Date("2024-09-03"))
stopifnot(monthBusinessDay(as.Date("2024-08-15"), "last") == as.Date("2024-08-30"))
stopifnot(monthBusinessDay(as.Date("2024-12-15"), "last") == as.Date("2024-12-31"))

cc = compareConventions(sat, holidays = hol)
stopifnot(identical(names(cc), c("date", "following", "preceding", "modifiedFollowing",
  "modifiedPreceding")))
stopifnot(cc$following == as.Date("2024-09-03"), cc$modifiedFollowing == as.Date("2024-08-30"))
stopifnot(ncol(compareConventions(sat, "unadjusted")) == 2L)
