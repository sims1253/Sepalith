src <- c(
  "f <- function(x, y = 1) {",
  "  if (x && y) z <- m[1, ] else return(2)",
  "  for (i in 1:3) { w = i }",
  "  x",
  "}",
  "g = function(...) h(...)",
  "k <- 3",
  "`long one` <- function(a, b, c) {",
  "  while (a) a <- a - 1",
  "  repeat break",
  "  if (b || c) 1",
  "}"
)

fm <- functionMetrics(src)
stopifnot(identical(fm$name, c("f", "g", "long one")), identical(fm$line, c(1L, 6L, 8L)))
stopifnot(identical(fm$args, c(2L, 1L, 3L)), identical(fm$lines, c(5L, 1L, 5L)))
stopifnot(identical(fm$statements, c(3L, 1L, 3L)), identical(fm$complexity, c(4L, 1L, 5L)))
stopifnot(identical(fm$assignments, c(2L, 0L, 1L)), identical(fm$returns, c(1L, 0L, 0L)))
stopifnot(nrow(functionMetrics("x <- 1")) == 0L)

rk <- rankFunctions(fm, "complexity")
stopifnot(identical(rk$name, c("long one", "f", "g")), identical(rk$rank, 1:3))
stopifnot(identical(names(rk), c("rank", "name", "line", "complexity")))
stopifnot(identical(rankFunctions(fm, "lines", n = 2L)$name, c("f", "long one")))
stopifnot(identical(rankFunctions(fm, "args")$name, c("long one", "f", "g")))
stopifnot(identical(rankFunctions(fm, "statements")$name, c("f", "long one", "g")))
stopifnot(identical(rankFunctions(fm, "returns", n = 1L)$returns, 1L))

fl <- flagFunctions(fm, c(lines = 2, args = 1, complexity = 4, statements = 2))
stopifnot(identical(fl$name, c("f", "long one")), identical(fl$line, c(1L, 8L)))
stopifnot(identical(fl$flags[1], "lines 5 > 2; args 2 > 1; statements 3 > 2"))
stopifnot(identical(fl$flags[2], "lines 5 > 2; args 3 > 1; complexity 5 > 4; statements 3 > 2"))
stopifnot(nrow(flagFunctions(fm)) == 0L)

ap <- argumentProfile(function(a, b = TRUE, c = NULL, d = list(), ..., e = NA) NULL)
stopifnot(identical(ap$required, "a"), identical(ap$optional, c("b", "c", "d", "e")))
stopifnot(identical(ap$dots, "..."), identical(ap$flags, "b"))
stopifnot(identical(ap$nullable, "c"), identical(ap$computed, "d"))
stopifnot(identical(argumentProfile(function() 1)$required, character()))
stopifnot(inherits(tryCatch(argumentProfile("f"), error = identity), "error"))

h <- function(a, b = TRUE, ...) NULL
stopifnot(identical(functionSignature(h, "h"), "h(a, b = TRUE, ...)"))
stopifnot(identical(functionSignature(h, "h", "call"), "h(a = a, b = b, ...)"))
stopifnot(identical(functionSignature(h, "h", "roxygen"),
  c("#' @param a", "#' @param b Defaults to `TRUE`.", "#' @param ...")))
md <- functionSignature(h, "h", "markdown")
stopifnot(length(md) == 5L, identical(md[4], "| `b` | `TRUE` |"), identical(md[3], "| `a` |  |"))
stopifnot(identical(functionSignature(function() NULL, "z"), "z()"))
stopifnot(identical(functionSignature(function() NULL, "z", "roxygen"), character()))
