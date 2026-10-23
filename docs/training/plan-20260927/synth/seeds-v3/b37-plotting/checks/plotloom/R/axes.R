msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
pdf(NULL)

l <- axis_limits(c(1, 3), c(10, 20), pad = 0.5)
stopifnot(identical(l$xlim, c(0, 4)), identical(l$ylim, c(5, 25)))
l <- axis_limits(c(2, 4), c(5, 5), pad = 0.5, include_zero = "x")
stopifnot(identical(l$xlim, c(-2, 6)), identical(l$ylim, c(2.5, 7.5)))
l <- axis_limits(c(2, 4), c(1, 3), pad = 0, include_zero = "both")
stopifnot(identical(l$xlim, c(0, 4)), identical(l$ylim, c(0, 3)))
stopifnot(identical(msg(axis_limits(1, 1, pad = -1)),
                    "`pad` must be a single non-negative number."))

stopifnot(identical(axis_label_format(c(0.5, 1)), c("0.5", "1.0")))
stopifnot(identical(axis_label_format(c(1000, 25000), "comma"), c("1,000", "25,000")))
stopifnot(identical(axis_label_format(c(0.1, 0.25), "percent"), c("10%", "25%")))
stopifnot(identical(axis_label_format(12345, "scientific"), "1.23e+04"))
stopifnot(identical(axis_label_format(c(500, 2000, 3e6, 4e9), "si"),
                    c("500", "2k", "3M", "4G")))

plot.new()
plot.window(xlim = c(0, 10), ylim = c(0, 1))
ax <- draw_axis_x(c(0, 10), title = "x")
stopifnot(identical(ax$breaks, c(0, 2, 4, 6, 8, 10)), identical(ax$minor, c(1, 3, 5, 7, 9)))
ay <- draw_axis_y(c(0, 1), n = 2L, minor = FALSE)
stopifnot(identical(ay$breaks, c(0, 0.5, 1)), identical(ay$minor, numeric()))
stopifnot(identical(ay$labels, c("0.0", "0.5", "1.0")))
stopifnot(identical(msg(draw_axis_x(c(0, 1), side = 2L)), "`side` must be 1 or 3 for the x axis."))
stopifnot(identical(msg(draw_axis_y(c(0, 1), side = 1L)), "`side` must be 2 or 4 for the y axis."))
stopifnot(identical(msg(draw_axis_x(1)), "`lim` must be two non-missing numbers."))
stopifnot(identical(msg(draw_axis_y(c(0, NA))), "`lim` must be two non-missing numbers."))

s <- setup_axes(c(0, 10), c(0, 100), flip = "y")
stopifnot(identical(names(s), c("xlim", "ylim", "xbreaks", "ybreaks")))
stopifnot(s$ylim[1] > s$ylim[2], s$xlim[1] < s$xlim[2])
s <- setup_axes(c(0, 10), c(0, 100), flip = "x")
stopifnot(s$xlim[1] > s$xlim[2])
s <- setup_axes(c(1, 1000), c(10, 100), log = "both")
stopifnot(all(c(0, 3) %in% s$xbreaks), s$ylim[2] < 2.1)
stopifnot(identical(msg(setup_axes(c(0, 1), 1:2, log = "x")),
                    "`x` must be positive for a log axis."))
stopifnot(identical(msg(setup_axes(1:2, c(-1, 1), log = "y")),
                    "`y` must be positive for a log axis."))

g <- grid_lines(c(1, 2, NA), c(10, 20, 30))
stopifnot(identical(g, list(n_vertical = 2L, n_horizontal = 3L)))
stopifnot(identical(grid_lines(ybreaks = 1)$n_vertical, 0L))
g <- grid_lines(c(3, 1, 2), c(10, 20), minor = TRUE)
stopifnot(identical(g, list(n_vertical = 5L, n_horizontal = 3L)))

plot.new()
plot.window(xlim = c(0, 10), ylim = c(0, 1))
v <- draw_secondary_axis("x", trans = function(u) u * 2, inverse = function(u) u / 2, title = "t")
stopifnot(min(v) >= 0, max(v) <= 20 * 1.04 + 1)
v <- draw_secondary_axis("y", trans = function(u) u * 100, inverse = function(u) u / 100,
                         title = "pct")
stopifnot(all(v %% 20 == 0))
stopifnot(identical(msg(draw_secondary_axis("x", trans = 2)),
                    "`trans` and `inverse` must be functions."))

invisible(dev.off())
