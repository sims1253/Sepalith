new_span <- function(start = double(), end = double(), label = character()) {
  structure(list(start = start, end = end, label = label), class = "spanwise_span")
}
validate_span <- function(x) {
  if (any(x$end < x$start)) stop("Span ends before it starts.")
  invisible(x)
}
new_gap <- function(at = double(), width = double()) {
  structure(list(at = at, width = width), class = "spanwise_gap")
}
validate_gap <- function(x) {
  if (any(x$width < 0)) stop("Gap has a negative width.")
  invisible(x)
}

a <- new_span(c(0, 10), c(2, 12), c("a", "b"))
b <- new_span(c(3, 1), c(5, 11), c("c", "d"))

stopifnot(identical(a == a, c(TRUE, TRUE)), identical(a != b, c(TRUE, TRUE)))
stopifnot(identical(a < b, c(TRUE, FALSE)), identical(a <= new_span(2, 4, "e"), c(TRUE, FALSE)))
stopifnot(identical(b > a, c(TRUE, FALSE)), identical(a >= new_span(2, 10, "e"), c(FALSE, TRUE)))
stopifnot(inherits(try(a == 1, silent = TRUE), "try-error"))
stopifnot(inherits(try(-a, silent = TRUE), "try-error"))

stopifnot(identical(unclass(a + 1)$start, c(1, 11)), identical(unclass(1 + a)$end, c(3, 13)))
stopifnot(identical(unclass(a - 1L)$start, c(-1, 9)), identical(unclass(a * 2)$end, c(4, 24)))
stopifnot(identical(unclass(a / 2)$start, c(0, 5)), identical(unclass(a + 1)$label, c("a", "b")))
stopifnot(inherits(try(1 - a, silent = TRUE), "try-error"))
stopifnot(inherits(try(a + b, silent = TRUE), "try-error"))
stopifnot(inherits(try(a + "x", silent = TRUE), "try-error"))
stopifnot(inherits(try(a * -1, silent = TRUE), "try-error"))
stopifnot(inherits(try(a^2, silent = TRUE), "try-error"))

g <- new_gap(c(1, 5), c(2, 1))
h <- new_gap(c(0, 0), c(2, 3))
stopifnot(identical(g == h, c(TRUE, FALSE)), identical(g < h, c(FALSE, TRUE)))
stopifnot(identical(g >= h, c(TRUE, FALSE)), identical(g != h, c(FALSE, TRUE)))
stopifnot(inherits(try(g > 1, silent = TRUE), "try-error"))
stopifnot(inherits(try(!g, silent = TRUE), "try-error"))
stopifnot(identical(unclass(g + 1)$at, c(2, 6)), identical(unclass(2 + g)$at, c(3, 7)))
stopifnot(identical(unclass(g - 1)$at, c(0, 4)), identical(unclass(g * 3)$width, c(6, 3)))
stopifnot(identical(unclass(g / 2)$width, c(1, 0.5)), identical(unclass(g / 2)$at, c(1, 5)))
stopifnot(inherits(try(2 / g, silent = TRUE), "try-error"))
stopifnot(inherits(try(g + h, silent = TRUE), "try-error"))
stopifnot(inherits(try(g * -1, silent = TRUE), "try-error"))
stopifnot(inherits(try(g %% 2, silent = TRUE), "try-error"))

r <- new_span(c(0.4, 1.6), c(2.2, 3.7), c("p", "q"))
stopifnot(identical(unclass(floor(r))$start, c(0, 1)), identical(unclass(ceiling(r))$end, c(3, 4)))
stopifnot(identical(unclass(round(r, 0))$end, c(2, 4)))
stopifnot(identical(unclass(trunc(r))$start, c(0, 1)))
stopifnot(inherits(try(sqrt(r), silent = TRUE), "try-error"))
q <- new_gap(c(0.4, 1.6), c(2.2, 3.7))
stopifnot(identical(unclass(floor(q))$at, c(0, 1)), identical(unclass(ceiling(q))$width, c(3, 4)))
stopifnot(identical(unclass(signif(q, 1))$width, c(2, 4)))
stopifnot(inherits(try(exp(q), silent = TRUE), "try-error"))

x <- new_span(c(0, 4, 9), c(3, 6, 12), c("a", "b", "c"))
cl <- span_clip(x, lower = 2, upper = 10)
stopifnot(identical(unclass(cl)$start, c(2, 4, 9)), identical(unclass(cl)$end, c(3, 6, 10)))
cl <- span_clip(x, lower = 5)
stopifnot(identical(unclass(cl)$label, c("b", "c")), identical(unclass(cl)$start, c(5, 9)))
cl <- span_clip(x, upper = 5, drop = FALSE)
stopifnot(identical(unclass(cl)$start, c(0, 4, 5)), identical(unclass(cl)$end, c(3, 5, 5)))
stopifnot(identical(span_clip(x), x))
stopifnot(inherits(try(span_clip(x, lower = 3, upper = 1), silent = TRUE), "try-error"))

y <- new_gap(c(0, 4, 9), c(3, 2, 3))
cg <- span_clip(y, lower = 2, upper = 10)
stopifnot(identical(unclass(cg)$at, c(2, 4, 9)), identical(unclass(cg)$width, c(1, 2, 1)))
cg <- span_clip(y, upper = 3)
stopifnot(identical(unclass(cg)$at, 0), identical(unclass(cg)$width, 3))
cg <- span_clip(y, lower = 7, drop = FALSE)
stopifnot(identical(unclass(cg)$at, c(7, 7, 9)), identical(unclass(cg)$width, c(0, 0, 3)))
stopifnot(inherits(try(span_clip(1:3, lower = 1), silent = TRUE), "try-error"))
