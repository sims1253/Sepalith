# skipped: needs ggplot2 (ps_scatter, ps_line, ps_bar, add_trend_line,
#   add_reference_line, add_mean_points, build_layered_plot)
df <- data.frame(a = 1:3, b = letters[1:3])
stopifnot(identical(check_plot_columns(df, list("a", NULL, "b")), df))
err <- try(check_plot_columns(df, list("a", "z")), silent = TRUE)
stopifnot(inherits(err, "try-error"), grepl("`z`", err))
stopifnot(inherits(try(check_plot_columns(list(a = 1), "a"), silent = TRUE), "try-error"))
