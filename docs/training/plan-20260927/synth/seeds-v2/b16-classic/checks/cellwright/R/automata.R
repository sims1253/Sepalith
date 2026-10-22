# check.count() lives in R/dice.R; a copy is needed to run this file alone
check.count = function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("'%s' must be a single number", name), call. = FALSE)
  }
  if (x < 1 || x != round(x)) {
    stop(sprintf("'%s' must be a positive whole number", name), call. = FALSE)
  }
  invisible(x)
}

# ca.rule.table
r30 = ca.rule.table(30)
stopifnot(identical(unname(r30), c(0L, 0L, 0L, 1L, 1L, 1L, 1L, 0L)), names(r30)[8] == "000")
stopifnot(inherits(try(ca.rule.table(256), silent = TRUE), "try-error"))

# ca.step
stopifnot(identical(ca.step(c(0L, 0L, 1L, 0L, 0L), r30), c(0L, 1L, 1L, 1L, 0L)))
stopifnot(identical(ca.step(c(1L, 0L, 0L), ca.rule.table(2), wrap = FALSE), c(0L, 0L, 0L)))

# ca.run
cr = ca.run(c(0, 0, 0, 1, 0, 0, 0), rule = 30, generations = 3)
stopifnot(identical(dim(cr), c(4L, 7L)), identical(cr[2, ], c(0L, 0L, 1L, 1L, 1L, 0L, 0L)))

# life.neighbours, including a single-row grid
blk = matrix(0, 4, 4)
blk[2:3, 2:3] = 1
nb = life.neighbours(blk)
stopifnot(identical(dim(nb), c(4L, 4L)), nb[2, 2] == 3, nb[1, 1] == 1)
row1 = life.neighbours(matrix(c(1, 1, 0, 0, 0), nrow = 1))
stopifnot(identical(dim(row1), c(1L, 5L)), row1[1, 3] == 3, row1[1, 4] == 0)

# life.step / life.run
stopifnot(identical(life.step(blk), blk == 1))
stopifnot(life.run(blk)$reason == "still", life.run(blk)$generations == 1L)
single = matrix(0, 3, 3)
single[2, 2] = 1
stopifnot(life.run(single)$reason == "extinct")
blinker = matrix(0, 5, 5)
blinker[3, 2:4] = 1
stopifnot(life.run(blinker, max.gen = 6)$reason == "limit")

# ca.render
out = capture.output(rows <- ca.render(matrix(c(1, 0, 0, 1), 2)))
stopifnot(identical(rows, c("#.", ".#")), identical(out, rows))
