# skipped: needs ggplot2/rlang/dplyr (facet_by, facet_wrap_balanced, add_facet_counts,
#   label_wrapped, strip_theme_top, strip_theme_left)
stopifnot(identical(choose_facet_layout(1), c(nrow = 1L, ncol = 1L)))
stopifnot(identical(choose_facet_layout(5), c(nrow = 2L, ncol = 3L)))
stopifnot(identical(choose_facet_layout(30, max_cols = 4L), c(nrow = 8L, ncol = 4L)))
stopifnot(inherits(try(choose_facet_layout(0), silent = TRUE), "try-error"))
