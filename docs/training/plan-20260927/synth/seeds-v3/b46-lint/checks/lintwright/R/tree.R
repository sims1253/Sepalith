# sys.source() turns parse data off while it evaluates this file
options(keep.parse.data = TRUE)

src <- c(
  "f <- function(x, y = 1) {",
  "  if (x) { if (y) g(x) else return(1) }",
  "  for (i in 1:3) while (TRUE) break",
  "}",
  "g <- function(z) f(z, m[1, ])",
  "library(stats)",
  "h = function() { repeat { h() } }",
  "x"
)

cp <- construct_profile(src)
stopifnot(identical(cp$construct, c("if", "for", "while", "repeat", "function")))
stopifnot(identical(cp$n, c(2L, 1L, 1L, 1L, 3L)), identical(cp$max_depth, c(2L, 1L, 1L, 1L, 1L)))
stopifnot(identical(cp$lines, c(2L, 1L, 1L, 1L, 6L)))
cp2 <- construct_profile(c("for (i in 1) {", "  for (j in 1) {", "    while (FALSE) NULL",
  "  }", "}"))
stopifnot(identical(cp2$max_depth, c(0L, 2L, 1L, 0L, 0L)), identical(cp2$lines[2], 8L))

ek <- expr_kinds(src)
stopifnot(identical(names(ek), c("call", "symbol", "character", "numeric", "logical", "null",
  "missing")))
stopifnot(ek[["numeric"]] == 5L, ek[["logical"]] == 1L, ek[["missing"]] == 3L)
ek2 <- expr_kinds("paste('a', NULL, 1L, 2i, x)")
stopifnot(identical(unname(ek2), c(1L, 2L, 1L, 2L, 0L, 1L, 0L)))

ft <- function_table(src)
stopifnot(identical(ft$name, c("f", "g", "h")), identical(ft$line, c(1L, 5L, 7L)))
stopifnot(identical(ft$formals, c(2L, 1L, 0L)), identical(ft$lines, c(4L, 1L, 1L)))
stopifnot(identical(ft$calls, c(2L, 1L, 1L)), identical(ft$returns, c(1L, 0L, 0L)))
stopifnot(nrow(function_table("x <- 1")) == 0L)

cg <- call_graph(src)
stopifnot(identical(cg$nodes, c("f", "g", "h")), identical(cg$edges$to, c("g", "f", "h")))
stopifnot(identical(cg$roots, character()), identical(cg$leaves, "h"))
stopifnot(identical(cg$recursive, "h"))
cg2 <- call_graph(c("a <- function() b()", "b <- function() 1"))
stopifnot(identical(cg2$roots, "a"), identical(cg2$leaves, "b"), length(cg2$recursive) == 0L)

tk <- top_level_kinds(c(src, "{1; 2}", "\"a\"", "base::sum(1)", "y <- 2", "for (i in 1) i"))
stopifnot(identical(tk$kind, c("function", "function", "attach", "function", "value", "block",
  "value", "call", "assignment", "control")))
stopifnot(identical(tk$name[c(3, 5, 6, 7, 8)], c("stats", "x", "2", "character", "base::sum")))
stopifnot(identical(tk$line, c(1L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L, 13L)), tk$name[10] == "for")
