# skipped: needs stringr, lubridate (parse.visit.date)
# skipped: needs dplyr (derive.study.day, derive.days.since.prev,
#   assign.visit.window, flag.early.visits, flag.late.visits)
# skipped: needs tibble (visit.schedule)

stopifnot(identical(.window.start(c(1L, 8L, 15L), c(3L, 3L, 3L)), c(1L, 5L, 12L)))
stopifnot(identical(.window.start(c(-7L, 2L), c(2L, 5L)), c(-9L, 1L)))
stopifnot(identical(.window.start(c(28, 56), c(7L, 7L)), c(21L, 49L)))
