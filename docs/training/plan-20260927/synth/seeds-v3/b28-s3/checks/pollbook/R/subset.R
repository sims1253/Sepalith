pb.new.tally <- function(counts = integer(), contest = "") {
  structure(counts, contest = contest, class = "pb_tally")
}
pb.new.roster <- function(voter = character(), district = character(), status = character()) {
  structure(list(voter = voter, district = district, status = status), class = "pb_roster")
}

roster <- pb.new.roster(c("v1", "v2", "v3"), c("N", "S", "S"), c("active", "active", "pending"))
ballots <- structure(list(c("x", "y"), character(), "z"), options = c("x", "y", "z"),
  class = "pb_ballot")
tally <- pb.new.tally(c(a = 3L, b = 1L, c = 0L), "mayor")

stopifnot(length(roster) == 3L, length(pb.new.roster()) == 0L)
stopifnot(length(ballots) == 3L, length(structure(list(), class = "pb_ballot")) == 0L)

stopifnot(identical(tally[c("c", "a")], pb.new.tally(c(c = 0L, a = 3L), "mayor")))
stopifnot(identical(tally[2], pb.new.tally(c(b = 1L), "mayor")), identical(tally[], tally))
stopifnot(grepl("Unknown option 'z'", try(tally["z"], silent = TRUE)))

stopifnot(identical(roster[c("v3", "v1")], pb.new.roster(c("v3", "v1"), c("S", "N"),
  c("pending", "active"))))
stopifnot(identical(unclass(roster[c(TRUE, FALSE, TRUE)])$voter, c("v1", "v3")))
stopifnot(identical(roster[], roster), inherits(try(roster["v9"], silent = TRUE), "try-error"))

stopifnot(identical(unclass(ballots[c(1, 3)])[[2]], "z"), identical(attr(ballots[1], "options"),
  c("x", "y", "z")))
stopifnot(identical(ballots[], ballots), inherits(try(ballots["a"], silent = TRUE), "try-error"))
stopifnot(inherits(try(ballots[NA_integer_], silent = TRUE), "try-error"))

other <- pb.new.tally(c(b = 2L, d = 5L), "ward")
s <- tally + other
stopifnot(identical(c(unclass(s)), c(a = 3L, b = 3L, c = 0L, d = 5L)))
stopifnot(identical(attr(s, "contest"), "mayor"))
d <- s - other
stopifnot(identical(c(unclass(d)), c(a = 3L, b = 1L, c = 0L, d = 0L)))
stopifnot(grepl("than 'b' has", try(tally - other, silent = TRUE)))
stopifnot(identical(tally == pb.new.tally(c(a = 3L, b = 2L)), c(a = TRUE, b = FALSE, c = TRUE)))
stopifnot(identical(tally != tally, c(a = FALSE, b = FALSE, c = FALSE)))
stopifnot(inherits(try(tally + 1L, silent = TRUE), "try-error"))
stopifnot(inherits(try(-tally, silent = TRUE), "try-error"))
stopifnot(inherits(try(tally * tally, silent = TRUE), "try-error"))

df <- as.data.frame(roster)
stopifnot(identical(names(df), c("voter", "district", "status")))
stopifnot(identical(df$district, c("N", "S", "S")))
df <- as.data.frame(roster, row.names = c("a", "b", "c"))
stopifnot(identical(rownames(df), c("a", "b", "c")))

db <- as.data.frame(ballots)
stopifnot(identical(names(db), c("ballot", "rank", "option")), identical(db$ballot, c(1L, 1L, 3L)))
stopifnot(identical(db$rank, c(1L, 2L, 1L)), identical(db$option, c("x", "y", "z")))
stopifnot(nrow(as.data.frame(structure(list(), options = "x", class = "pb_ballot"))) == 0L)
