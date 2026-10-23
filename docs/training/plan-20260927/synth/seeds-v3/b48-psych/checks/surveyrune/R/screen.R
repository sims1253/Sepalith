d <- data.frame(a = c(1, 3, 3, NA), b = c(9, 3, 2, NA), c = c(2, 3, 4, 1))
sr <- screen.responses(d, c("a", "b", "c"), max.longstring = 2, max.missing = 0.5,
  seconds = c(100, 30, 200, NA))
stopifnot(identical(names(sr), c("out.of.range", "longstring", "missing", "speeder", "n.flags")))
stopifnot(identical(sr$out.of.range, c(TRUE, FALSE, FALSE, FALSE)))
stopifnot(identical(sr$longstring, c(FALSE, TRUE, FALSE, FALSE)))
stopifnot(identical(sr$missing, c(FALSE, FALSE, FALSE, TRUE)))
stopifnot(identical(sr$speeder, c(FALSE, TRUE, FALSE, FALSE)))
stopifnot(identical(sr$n.flags, c(1, 2, 0, 1)))
stopifnot(!any(screen.responses(d, c("a", "c"))$speeder))
stopifnot(inherits(try(screen.responses(d, "a", seconds = 1), silent = TRUE), "try-error"))

stopifnot(all(careless.index(d, c("a", "b", "c")) == c(1, 3, 1, 1)))
irv <- careless.index(d, c("a", "b", "c"), "irv")
stopifnot(irv[2] == 0, attr(irv, "index") == "irv", is.na(irv[4]))
set.seed(2)
m <- data.frame(matrix(rnorm(60), 20))
m[1, ] <- 10
mh <- careless.index(m, names(m), "mahalanobis")
stopifnot(which.max(mh) == 1, length(mh) == 20)
eo <- data.frame(x1 = c(1, 1), x2 = c(1, 5), y1 = c(3, 3), y2 = c(3, 1), z1 = c(5, 5),
  z2 = c(5, 3))
sc <- list(x = c("x1", "x2"), y = c("y1", "y2"), z = c("z1", "z2"))
ev <- careless.index(eo, names(eo), "even.odd", scales = sc)
stopifnot(all.equal(ev[1], 1), ev[2] < 0)
stopifnot(inherits(try(careless.index(eo, names(eo), "even.odd"), silent = TRUE), "try-error"))
stopifnot(identical(c(careless.index(d, c("a", "b", "c"), "span")), c(8, 0, 2, 0)))

at <- data.frame(i1 = c(3, 2, 3), f1 = c(1, 1, NA), b1 = c(1, 5, 1), b2 = c(1, 1, 5))
ac <- attention.check(at, instructed = c(i1 = 3), infrequency = c(f1 = 1),
  bogus = c(b1 = 1, b2 = 1))
stopifnot(identical(ac$instructed, c(0L, 1L, 0L)), identical(ac$infrequency, c(0L, 0L, 1L)))
stopifnot(identical(ac$bogus, c(0L, 1L, 1L)), identical(ac$n.failed, c(0L, 2L, 2L)))
stopifnot(all(attention.check(at)$n.failed == 0))
stopifnot(inherits(try(attention.check(at, instructed = c(zz = 1)), silent = TRUE), "try-error"))
stopifnot(inherits(try(attention.check(at, infrequency = c(zz = 1)), silent = TRUE), "try-error"))
stopifnot(inherits(try(attention.check(at, bogus = c(zz = 1)), silent = TRUE), "try-error"))

secs <- c(30, 100, 120, 140, NA, 400)
stopifnot(identical(speed.flags(secs, 50), c(TRUE, FALSE, FALSE, FALSE, FALSE, FALSE)))
stopifnot(identical(speed.flags(secs, 50, "relative"), c(TRUE, FALSE, FALSE, FALSE, FALSE,
  FALSE)))
stopifnot(identical(speed.flags(secs, 50, "per.item", limit = 2.5), c(TRUE, TRUE, TRUE, FALSE,
  FALSE, FALSE)))
stopifnot(identical(speed.flags(secs, 50, "mad"), c(TRUE, FALSE, FALSE, FALSE, FALSE, FALSE)))

ss <- screen.summary(sr)
stopifnot(identical(names(ss), c("n", "per.check", "share.per.check", "n.any", "share.any",
  "keep")))
stopifnot(ss$n == 4, ss$per.check[["longstring"]] == 1, ss$n.any == 3, ss$share.any == 0.75)
stopifnot(identical(ss$keep, 3L), ss$share.per.check[["missing"]] == 0.25)
stopifnot(identical(screen.summary(sr, "speeder")$keep, c(1L, 3L, 4L)))
stopifnot(inherits(try(screen.summary(sr, "zz"), silent = TRUE), "try-error"))
