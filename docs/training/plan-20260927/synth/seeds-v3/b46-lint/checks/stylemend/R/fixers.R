# sys.source() turns parse data off while it evaluates this file
options(keep.parse.data = TRUE)

src <- c("x = T; y <- F ", "", "  if (x) { a <- 1;b = 2; }", "s <- 'it\\'s \"q\"'")

f1 <- fix.rule(src, "assign_eq")
stopifnot(identical(f1$rule, "assign_eq"), identical(f1$before, src))
stopifnot(identical(f1$changed, c(1L, 3L)))
stopifnot(identical(f1$after[c(1, 3)], c("x <- T; y <- F ", "  if (x) { a <- 1;b <- 2; }")))
f2 <- fix.rule(src, "true_false")
stopifnot(identical(f2$after[1], "x = TRUE; y <- FALSE "), identical(f2$changed, 1L))
f3 <- fix.rule(src, "semicolon")
stopifnot(identical(f3$after, c("x = T", "y <- F ", "", "  if (x) { a <- 1", "  b = 2", "  }",
  "s <- 'it\\'s \"q\"'")))
stopifnot(identical(fix.rule("x <- 1;", "semicolon")$after, "x <- 1"))
f4 <- fix.rule(src, "trailing_space")
stopifnot(identical(f4$after[1], "x = T; y <- F"), identical(f4$changed, 1L))
f5 <- fix.rule(src, "single_quotes")
stopifnot(identical(f5$after[4], "s <- \"it's \\\"q\\\"\""), identical(f5$changed, 4L))
stopifnot(identical(eval(str2lang(sub("s <- ", "", f5$after[4]))), "it's \"q\""))
stopifnot(identical(fix.rule("y <- 1", "assign_eq")$changed, integer()))

fa <- fix.all(src)
stopifnot(identical(fa$after[1:2], c("x <- TRUE", "y <- FALSE")), length(fa$after) == 7L)
stopifnot(identical(fa$log$lines_after, c(4L, 4L, 7L, 7L, 7L)))
stopifnot(identical(fa$log$edited, c(2L, 1L, 2L, 1L, 1L)), identical(fa$before, src))
fb <- fix.all(src, rules = c("trailing_space", "true_false"))
stopifnot(identical(fb$log$rule, c("trailing_space", "true_false")))
stopifnot(identical(fb$after[1], "x = TRUE; y <- FALSE"))

gaps <- c("", " ", "a", "", "", "", "b", "", "")
bl <- fix.blank.lines(gaps)
stopifnot(identical(bl$after, c("a", "", "b")), identical(bl$removed, c(start = 2L, middle = 2L,
  end = 2L)))
stopifnot(identical(fix.blank.lines(gaps, at = "start")$after, c("a", "", "b", "")))
stopifnot(identical(fix.blank.lines(gaps, at = "end", max.blank = 2L)$after,
  c("", " ", "a", "", "", "b")))
stopifnot(identical(fix.blank.lines(c("", ""))$after, character()))

q1 <- fix.quotes(src)
stopifnot(identical(q1$after[4], "s <- \"it's \\\"q\\\"\""), identical(q1$changed, 4L))
q2 <- fix.quotes(q1$after, "single")
stopifnot(identical(q2$after, src), identical(q2$before, q1$after))
q3 <- fix.quotes("f(\"a\", 'b', \"c\")", "single")
stopifnot(identical(q3$after, "f('a', 'b', 'c')"), identical(q3$changed, 1L))

lam <- c("f <- function(x) lapply(x, function(v) v + 1)",
  "g = function() Map(function(a, b) a, 1, 2)")
fl <- fix.lambda(lam)
stopifnot(identical(fl$after, c("f <- function(x) lapply(x, \\(v) v + 1)",
  "g = function() Map(\\(a, b) a, 1, 2)")), identical(fl$changed, c(1L, 2L)))
fl2 <- fix.lambda(fl$after, "function")
stopifnot(identical(fl2$after, lam), identical(fl2$before, fl$after))
stopifnot(identical(fix.lambda("h <- function() 1")$changed, integer()))
