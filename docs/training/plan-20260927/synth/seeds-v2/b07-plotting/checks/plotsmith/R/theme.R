# skipped: needs ggplot2 (theme_plotsmith, theme_plotsmith_dark, theme_no_x_axis,
#   theme_no_y_axis, theme_legend_position, use_plotsmith_theme)
stopifnot(identical(check_base_size(11), 11))
stopifnot(inherits(try(check_base_size(4), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_base_size(c(10, 12)), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_base_size(NA_real_), silent = TRUE), "try-error"))
