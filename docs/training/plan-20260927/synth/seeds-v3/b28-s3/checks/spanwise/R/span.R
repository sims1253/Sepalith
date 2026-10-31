stopifnot(identical(span_fields("span"), c("start", "end", "label")))
stopifnot(identical(span_fields("gap"), c("at", "width")))
stopifnot(inherits(try(span_fields("box"), silent = TRUE), "try-error"))

s0 <- new_span(c(1, 2), c(3, 4), c("a", "b"))
stopifnot(inherits(s0, "spanwise_span"), identical(unclass(s0)$label, c("a", "b")))
stopifnot(inherits(try(new_span(1L, 2, "a"), silent = TRUE), "try-error"))

stopifnot(identical(validate_span(s0), s0))
stopifnot(inherits(try(validate_span(new_span(c(1, NA), c(2, 3), c("a", "b"))),
  silent = TRUE), "try-error"))
stopifnot(inherits(try(validate_span(new_span(1, Inf, "a")), silent = TRUE), "try-error"))
stopifnot(inherits(try(validate_span(new_span(1, c(2, 3), "a")), silent = TRUE), "try-error"))
stopifnot(inherits(try(validate_span(new_span(1, 2, NA_character_)), silent = TRUE),
  "try-error"))
stopifnot(inherits(try(validate_span(new_span(1, 2, c("a", "b"))), silent = TRUE),
  "try-error"))
err <- try(validate_span(new_span(c(1, 5), c(2, 4), c("a", "b"))), silent = TRUE)
stopifnot(grepl("Span 2 ends before", err))
stopifnot(inherits(try(validate_span(structure(list(start = 1), class = "spanwise_span")),
  silent = TRUE), "try-error"))

s1 <- span(0, c(2L, 5L))
stopifnot(identical(unclass(s1)$start, c(0, 0)), identical(unclass(s1)$label, c("span1", "span2")))
s2 <- span(c(1, 2, 3), 10, label = "x")
stopifnot(identical(unclass(s2)$label, rep("x", 3)), identical(unclass(s2)$end, rep(10, 3)))
stopifnot(inherits(try(span(5, 1), silent = TRUE), "try-error"))

g0 <- new_gap(c(1, 2), c(0.5, 0))
stopifnot(inherits(g0, "spanwise_gap"), identical(validate_gap(g0), g0))
stopifnot(inherits(try(new_gap(1L, 2), silent = TRUE), "try-error"))
stopifnot(inherits(try(validate_gap(new_gap(NaN, 1)), silent = TRUE), "try-error"))
stopifnot(inherits(try(validate_gap(new_gap(1, NA_real_)), silent = TRUE), "try-error"))
stopifnot(inherits(try(validate_gap(new_gap(1, c(1, 2))), silent = TRUE), "try-error"))
stopifnot(grepl("Gap 1 has a negative", try(validate_gap(new_gap(1, -1)), silent = TRUE)))
stopifnot(inherits(try(validate_gap(structure(list(at = 1), class = "spanwise_gap")),
  silent = TRUE), "try-error"))

g1 <- gap(c(1, 4, 9), 2)
stopifnot(identical(unclass(g1)$width, c(2, 2, 2)), identical(unclass(g1)$at, c(1, 4, 9)))
g2 <- gap(3, c(1, 2))
stopifnot(identical(unclass(g2)$at, c(3, 3)))
stopifnot(inherits(try(gap(1, -2), silent = TRUE), "try-error"))
stopifnot(grepl("`start` must be numeric, not <character>", try(span("a", 1), silent = TRUE)))
stopifnot(grepl("`end` must not be empty", try(span(1, numeric()), silent = TRUE)))
stopifnot(inherits(try(span(numeric(), 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(span(1, TRUE), silent = TRUE), "try-error"))
stopifnot(grepl("`at` must be numeric", try(gap("1", 1), silent = TRUE)))
stopifnot(grepl("`width` must not be empty", try(gap(1, double()), silent = TRUE)))
stopifnot(inherits(try(gap(integer(), 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(gap(1, list(1)), silent = TRUE), "try-error"))
