msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
pdf(NULL)

stopifnot(identical(recycle_legend_arg("red", 3L, "col"), rep("red", 3)))
stopifnot(identical(recycle_legend_arg(1:2, 2L, "pch"), 1:2))
stopifnot(identical(msg(recycle_legend_arg(1:2, 3L, "pch")),
                    "`pch` must have length 1 or 3, not 2."))

p <- legend_points(c("a", "b"), col = c("red", "blue"), title = "grp")
stopifnot(inherits(p, "plotloom_legend"), identical(p$pch, c(19L, 19L)), identical(p$title, "grp"))
stopifnot(identical(msg(legend_points(character())), "`labels` must contain at least one label."))
stopifnot(identical(msg(legend_points(c("a", "a"))), "`labels` must be unique."))
stopifnot(identical(msg(legend_points("a", pch = 30L)),
                    "`pch` must only contain codes from 0 to 25."))
stopifnot(identical(msg(legend_points("a", cex = 0)), "`cex` must be positive."))

l <- legend_lines(1:3, lty = 1:3)
stopifnot(identical(l$labels, c("1", "2", "3")), identical(l$lwd, c(1, 1, 1)), is.null(l$title))
stopifnot(identical(msg(legend_lines("a", lwd = 1:2)), "`lwd` must have length 1 or 1, not 2."))
stopifnot(identical(msg(legend_lines(c("a", "a"))), "`labels` must be unique."))
stopifnot(identical(msg(legend_lines("a", lty = 9L)),
                    "`lty` must only contain codes from 0 to 6."))
stopifnot(identical(msg(legend_lines("a", lwd = -1)), "`lwd` must be positive."))

f <- legend_fill(c("x", "y"), density = c(NA, 20))
stopifnot(identical(f$type, "fill"), identical(f$border, c("black", "black")))
stopifnot(identical(msg(legend_fill(c("x", "y"), border = 1:3)),
                    "`border` must have length 1 or 2, not 3."))
stopifnot(identical(msg(legend_fill("x", col = NA)), "`col` must not be missing."))
stopifnot(identical(msg(legend_fill("x", density = -2)), "`density` must be positive."))

stopifnot(identical(legend_position("topleft")$x, "topleft"))
stopifnot(identical(legend_position("bottomright", inset = 0.1)$inset, c(0.1, 0.1)))
stopifnot(identical(legend_position("bottomleft")$xpd, FALSE))
stopifnot(identical(legend_position("top")$inset, c(0, 0.02)))
stopifnot(identical(legend_position("bottom")$x, "bottom"))
stopifnot(identical(legend_position("outside")$xpd, TRUE))
stopifnot(identical(legend_position()$x, "topright"))

plot.new()
plot.window(0:1, 0:1)
stopifnot(is.list(draw_legend(p)), is.list(draw_legend(l, "bottomleft", title = "t")))
stopifnot(is.list(draw_legend(f, "outside")), is.list(draw_legend(f, "top")))
stopifnot(identical(msg(draw_legend(list())), "`spec` must be a plotloom_legend."))
bad <- p
bad$type <- "ribbon"
stopifnot(identical(msg(draw_legend(bad)), "Unknown legend type `ribbon`."))

cmb <- legend_combine(legend_points("a", col = "red", title = "first"),
                      legend_points(c("b", "c"), pch = 1L, title = "second"))
stopifnot(identical(cmb$labels, c("a", "b", "c")), identical(cmb$pch, c(19L, 1L, 1L)))
stopifnot(identical(cmb$title, "first"), identical(names(cmb)[6], "title"))
stopifnot(is.null(legend_combine(l)$title), "title" %in% names(legend_combine(l)))
stopifnot(identical(msg(legend_combine(p, l)), "Cannot combine legend types: points, lines."))
stopifnot(identical(msg(legend_combine()), "Supply at least one legend spec."))
stopifnot(identical(msg(legend_combine(p, p)), "`labels` must be unique."))

invisible(dev.off())
