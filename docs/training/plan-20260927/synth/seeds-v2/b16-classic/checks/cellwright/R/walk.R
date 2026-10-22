# check.count() and check.probability() live in R/dice.R; copies are needed
# to run this file alone
check.count = function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("'%s' must be a single number", name), call. = FALSE)
  }
  if (x < 1 || x != round(x)) {
    stop(sprintf("'%s' must be a positive whole number", name), call. = FALSE)
  }
  invisible(x)
}
check.probability = function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("'%s' must be a single number", name), call. = FALSE)
  }
  if (x < 0 || x > 1) {
    stop(sprintf("'%s' must lie between 0 and 1", name), call. = FALSE)
  }
  invisible(x)
}
set.seed(7)

# walk.1d
w = walk.1d(50, start = 3)
stopifnot(length(w) == 51, w[1] == 3L, all(abs(diff(w)) == 1L))
stopifnot(identical(walk.1d(5, p.up = 1), 0:5))

# walk.2d
w2 = walk.2d(20, start = c(2, -1))
stopifnot(identical(dim(w2), c(21L, 2L)), identical(colnames(w2), c("x", "y")))
stopifnot(all(rowSums(abs(diff(w2))) == 1L), identical(unname(w2[1, ]), c(2L, -1L)))

# walk.first.passage
stopifnot(walk.first.passage(4, p.up = 1) == 4L)
stopifnot(is.na(walk.first.passage(3, p.up = 0, max.steps = 10)))
stopifnot(inherits(try(walk.first.passage(0), silent = TRUE), "try-error"))

# walk.max.excursion / walk.return.count
p = c(0L, 1L, 0L, -1L, -2L, -1L, 0L)
stopifnot(walk.max.excursion(p) == 2L, walk.return.count(p) == 2L)

# gambler.ruin
stopifnot(identical(gambler.ruin(5, 8, p.win = 1), list(result = "target", rounds = 3L)))
stopifnot(gambler.ruin(2, 10, p.win = 0)$result == "ruin")
stopifnot(inherits(try(gambler.ruin(5, 7.5), silent = TRUE), "try-error"))

# walk.summary.table
st = walk.summary.table(n.reps = 10, n.steps = 20)
stopifnot(nrow(st) == 10, all(st$excursion >= abs(st$final)), all(st$final %% 2 == 0))
