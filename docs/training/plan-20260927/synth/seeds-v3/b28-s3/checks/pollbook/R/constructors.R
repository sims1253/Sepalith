t0 <- pb.new.tally(c(a = 2L, b = 1L), "mayor")
stopifnot(inherits(t0, "pb_tally"), identical(attr(t0, "contest"), "mayor"))
stopifnot(inherits(try(pb.new.tally(c(a = 1)), silent = TRUE), "try-error"))

stopifnot(identical(pb.validate.tally(t0), t0))
stopifnot(identical(pb.validate.tally(pb.new.tally()), pb.new.tally()))
stopifnot(inherits(try(pb.validate.tally(pb.new.tally(c(a = 1L), c("x", "y"))), silent = TRUE),
  "try-error"))
stopifnot(inherits(try(pb.validate.tally(pb.new.tally(c(a = 1L), NA_character_)),
  silent = TRUE), "try-error"))
stopifnot(inherits(try(pb.validate.tally(pb.new.tally(1:2)), silent = TRUE), "try-error"))
stopifnot(inherits(try(pb.validate.tally(pb.new.tally(c(a = 1L, 2L))), silent = TRUE),
  "try-error"))
err <- try(pb.validate.tally(pb.new.tally(c(a = 1L, a = 2L))), silent = TRUE)
stopifnot(grepl("Option 'a' appears", err))
stopifnot(inherits(try(pb.validate.tally(pb.new.tally(c(a = -1L))), silent = TRUE), "try-error"))
stopifnot(inherits(try(pb.validate.tally(pb.new.tally(c(a = NA_integer_))), silent = TRUE),
  "try-error"))

t1 <- pb.tally(c("x", "y", NA, "x"), contest = "council")
stopifnot(identical(c(unclass(t1)), c(x = 2L, y = 1L)), identical(attr(t1, "contest"), "council"))
t2 <- pb.tally(c("b", "b"), options = c("a", "b", "c"))
stopifnot(identical(as.integer(t2), c(0L, 2L, 0L)))
stopifnot(identical(names(unclass(t2)), c("a", "b", "c")))
stopifnot(grepl("unknown option 'z'", try(pb.tally("z", options = "a"), silent = TRUE)))
stopifnot(length(unclass(pb.tally(character()))) == 0L)

r0 <- pb.new.roster(c("v1", "v2"), c("N", "S"), c("active", "pending"))
stopifnot(inherits(r0, "pb_roster"), identical(unclass(r0)$district, c("N", "S")))
stopifnot(inherits(try(pb.new.roster(1, "N", "active"), silent = TRUE), "try-error"))

stopifnot(identical(pb.validate.roster(r0), r0))
stopifnot(inherits(try(pb.validate.roster(pb.new.roster(NA_character_, "N", "active")),
  silent = TRUE), "try-error"))
stopifnot(grepl("Voter 'v1' is listed twice", try(pb.validate.roster(
  pb.new.roster(c("v1", "v1"), c("N", "N"), c("active", "active"))), silent = TRUE)))
stopifnot(inherits(try(pb.validate.roster(pb.new.roster("v1", NA_character_, "active")),
  silent = TRUE), "try-error"))
stopifnot(inherits(try(pb.validate.roster(pb.new.roster("v1", c("N", "S"), "active")),
  silent = TRUE), "try-error"))
stopifnot(inherits(try(pb.validate.roster(pb.new.roster("v1", "N", NA_character_)),
  silent = TRUE), "try-error"))
stopifnot(inherits(try(pb.validate.roster(pb.new.roster("v1", "N", character())),
  silent = TRUE), "try-error"))
stopifnot(grepl("Unknown status 'moved'",
  try(pb.validate.roster(pb.new.roster("v1", "N", "moved")), silent = TRUE)))
stopifnot(identical(pb.validate.roster(pb.new.roster("v1", "N", "moved"), statuses = "moved"),
  pb.new.roster("v1", "N", "moved")))

r1 <- pb.roster(c("a", "b", "c"), "East")
stopifnot(identical(unclass(r1)$district, rep("East", 3)))
stopifnot(identical(unclass(r1)$status, rep("active", 3)))
r2 <- pb.roster(1:2, c("N", "S"), c("inactive", "active"))
stopifnot(identical(unclass(r2)$voter, c("1", "2")), identical(unclass(r2)$status[1], "inactive"))
stopifnot(inherits(try(pb.roster("a", "N", "gone"), silent = TRUE), "try-error"))

b <- pb.ballot(list(c("x", "y"), character(), "z"), options = c("x", "y", "z"))
stopifnot(inherits(b, "pb_ballot"), identical(attr(b, "options"), c("x", "y", "z")))
stopifnot(identical(unclass(b)[[2]], character()))
stopifnot(inherits(try(pb.ballot(list("x"), c("x", "x")), silent = TRUE), "try-error"))
stopifnot(grepl("Ballot 2 ranks unknown option 'q'",
  try(pb.ballot(list("x", "q"), "x"), silent = TRUE)))
stopifnot(grepl("Ballot 1 ranks 'x' twice", try(pb.ballot(list(c("x", "x")), "x"), silent = TRUE)))
stopifnot(identical(pb.validate.ballot(b), b))
stopifnot(inherits(try(pb.validate.ballot(structure(list(), options = character(),
  class = "pb_ballot")), silent = TRUE), "try-error"))
stopifnot(grepl("Ballot 1 must be a character vector", try(pb.validate.ballot(
  structure(list(1), options = "x", class = "pb_ballot")), silent = TRUE)))
stopifnot(inherits(try(pb.ballot("x", "x"), silent = TRUE), "try-error"))
stopifnot(is.null(names(unclass(pb.ballot(list(a = "x"), "x")))))
r3 <- pb.roster(factor(c("b", "a")), factor("N"), factor(c("active", "pending")))
stopifnot(identical(unclass(r3)$voter, c("b", "a")), identical(unclass(r3)$district, c("N", "N")))
stopifnot(identical(unclass(r3)$status, c("active", "pending")))
