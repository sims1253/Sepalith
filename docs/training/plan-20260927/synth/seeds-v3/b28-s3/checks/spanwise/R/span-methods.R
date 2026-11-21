new_span <- function(start = double(), end = double(), label = character()) {
  structure(list(start = start, end = end, label = label), class = "spanwise_span")
}
new_gap <- function(at = double(), width = double()) {
  structure(list(at = at, width = width), class = "spanwise_gap")
}
registerS3method("format", "spanwise_span", format.spanwise_span)
registerS3method("format", "spanwise_gap", format.spanwise_gap)

s <- new_span(c(1, 2.5, 4), c(3, 6, 4.25), c("a", "b", "c"))
g <- new_gap(c(10, 20), c(1.5, 0))

stopifnot(identical(format(s), c("a [1.0, 3.00]", "b [2.5, 6.00]", "c [4.0, 4.25]")))
stopifnot(identical(format(new_span()), character()))
stopifnot(identical(format(g), c("10 (+1.5)", "20 (+0.0)")))
stopifnot(identical(format(new_gap()), character()))

out <- capture.output(res <- print(s))
stopifnot(identical(res, s), identical(out[1], "<spanwise_span[3]>"), length(out) == 4L)
out <- capture.output(print(s, max = 2L))
stopifnot(identical(out[4], "  ... and 1 more"))
stopifnot(identical(capture.output(print(new_span())), "<spanwise_span[0]>"))

out <- capture.output(res <- print(g))
stopifnot(identical(res, g), identical(out, c("<spanwise_gap[2]>", "  10 (+1.5)", "  20 (+0.0)")))
stopifnot(identical(capture.output(print(g, max = 1L))[3], "  ... and 1 more"))
stopifnot(identical(capture.output(print(new_gap())), "<spanwise_gap[0]>"))

stopifnot(length(s) == 3L, length(new_span()) == 0L)
stopifnot(length(g) == 2L, length(new_gap()) == 0L)

stopifnot(identical(unclass(s[2:3])$label, c("b", "c")))
stopifnot(identical(unclass(s["c"])$end, 4.25), identical(s[], s))
stopifnot(inherits(try(s["z"], silent = TRUE), "try-error"))

stopifnot(identical(unclass(g[c(FALSE, TRUE)])$at, 20), identical(g[], g))
stopifnot(inherits(try(g["a"], silent = TRUE), "try-error"))
stopifnot(inherits(try(g[NA_integer_], silent = TRUE), "try-error"))
