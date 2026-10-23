msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
pdf(NULL)

b <- c(bottom = 4, left = 4, top = 4, right = 2)
stopifnot(identical(margin_preset("default", b), c(4, 4, 4, 2)))
stopifnot(identical(margin_preset("compact", b), c(2, 2, 1, 1)))
stopifnot(identical(margin_preset("wide", b), c(4, 8, 4, 4)))
stopifnot(identical(margin_preset("title", b), c(2, 2, 4, 1)))
stopifnot(identical(margin_preset("none", b), c(0, 0, 0, 0)))
stopifnot(identical(msg(margin_preset(base = c(1, 2, 3, 4))),
                    "`base` must be a numeric vector named bottom, left, top, right."))

stopifnot(identical(panel_grid(5), c(2L, 3L)))
stopifnot(identical(panel_grid(5, ncol = 2), c(3L, 2L)))
stopifnot(identical(panel_grid(5, nrow = 1), c(1L, 5L)))
stopifnot(identical(panel_grid(4, nrow = 2, ncol = 2), c(2L, 2L)))
stopifnot(identical(msg(panel_grid(5, 2, 2)), "A 2 x 2 grid cannot hold 5 panels."))
stopifnot(identical(msg(panel_grid(5, nrow = 0)), "`nrow` must be at least 1."))
stopifnot(identical(msg(panel_grid(5, ncol = 0)), "`ncol` must be at least 1."))
stopifnot(identical(msg(panel_grid(0)), "`n` must be a single positive number."))

l <- layout_with_legend(3, ncol = 2L)
stopifnot(identical(l$mat, matrix(c(1L, 3L, 2L, 0L), 2)), identical(l$widths, c(1, 1)))
l <- layout_with_legend(3, ncol = 2L, legend = "right")
stopifnot(identical(dim(l$mat), c(2L, 3L)), all(l$mat[, 3] == 4L))
stopifnot(identical(l$widths, c(1, 1, 0.3)))
l <- layout_with_legend(3, ncol = 2L, legend = "left")
stopifnot(all(l$mat[, 1] == 4L), identical(l$widths, c(0.3, 1, 1)))
l <- layout_with_legend(3, ncol = 2L, legend = "bottom", size = 0.5)
stopifnot(all(l$mat[3, ] == 4L), identical(l$heights, c(1, 1, 0.5)))
l <- layout_with_legend(3, ncol = 2L, legend = "top")
stopifnot(all(l$mat[1, ] == 4L), identical(l$heights, c(0.3, 1, 1)))
stopifnot(layout(l$mat, widths = l$widths, heights = l$heights) == 4L)

stopifnot(identical(check_par_settings(list(mar = c(1, 1, 1, 1), las = 1)),
                    list(mar = c(1, 1, 1, 1), las = 1)))
stopifnot(identical(msg(check_par_settings(list(pch = 1))), "Unsupported par setting(s): pch."))
stopifnot(identical(msg(check_par_settings(list(mar = 1))),
                    "`mar` must be 4 non-negative numbers."))
stopifnot(identical(msg(check_par_settings(list(oma = c(-1, 0, 0, 0)))),
                    "`oma` must be 4 non-negative numbers."))
stopifnot(identical(msg(check_par_settings(list(mgp = 1:4))),
                    "`mgp` must be 3 non-negative numbers."))
stopifnot(identical(msg(check_par_settings(list(mfrow = c(0, 1)))),
                    "`mfrow` must be 2 positive numbers."))
stopifnot(identical(msg(check_par_settings(list(cex = 0))), "`cex` must be 1 positive number."))
stopifnot(identical(msg(check_par_settings(list(las = 5))), "`las` must be 1 of 0, 1, 2, 3."))

before <- par("cex")
inside <- with_par(list(cex = 0.5), par("cex"))
stopifnot(identical(inside, 0.5), identical(par("cex"), before))

invisible(dev.off())
