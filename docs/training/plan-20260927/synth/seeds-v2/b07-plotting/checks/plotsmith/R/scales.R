# skipped: needs ggplot2 (scale_x_percent, scale_y_percent, scale_x_compact,
#   scale_y_compact, scale_colour_plotsmith, scale_fill_plotsmith)
stopifnot(identical(label_compact(c(950, 1500, 2.5e6, 3e9)), c("950", "1.5K", "2.5M", "3B")))
stopifnot(identical(label_compact(c(-12000, NA)), c("-12K", NA)))
stopifnot(identical(label_percent_simple(c(0.25, 1)), c("25%", "100%")))
stopifnot(identical(label_percent_simple(0.1234, accuracy = 1), "12.3%"))
stopifnot(identical(label_percent_simple(c(0.5, NA)), c("50%", NA)))
stopifnot(length(plotsmith_colours()) == 7L)
stopifnot(all(grepl("^#[0-9a-f]{6}$", plotsmith_colours())))
