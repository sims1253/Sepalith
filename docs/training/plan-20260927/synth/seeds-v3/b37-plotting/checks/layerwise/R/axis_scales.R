msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
# skipped: needs ggplot2 (lw_axes; success paths of lw_scale_x, lw_scale_y)

stopifnot(identical(axis_expand(c(0, 10)), c(-0.5, 10.5)))
stopifnot(identical(axis_expand(c(10, 0), mult = 0, add = 1, side = "lower"), c(-1, 10)))
stopifnot(identical(axis_expand(c(0, 10), mult = 0.1, side = "upper"), c(0, 11)))
stopifnot(identical(msg(axis_expand(c(1, NA))), "`limits` must be two non-missing numbers."))
stopifnot(identical(msg(axis_expand(c(0, 1), mult = -1)),
                    "`mult` must be a single non-negative number."))
stopifnot(identical(msg(axis_expand(c(0, 1), add = NA_real_)),
                    "`add` must be a single non-negative number."))

stopifnot(identical(axis_labeller("number")(c(1, 2.25)), c("1.0", "2.2")))
stopifnot(identical(axis_labeller("percent", digits = 0L)(0.5), "50%"))
stopifnot(identical(axis_labeller("comma")(1234567), "1,234,567"))
stopifnot(identical(axis_labeller("unit", unit = "kg")(2), "2.0 kg"))
stopifnot(identical(axis_labeller("log10")(c(10, 1000)), c("10^1", "10^3")))
stopifnot(identical(axis_labeller("comma")(c(NA, 1000)), c("", "1,000")))

s <- axis_pair_spec(c(0, 10), c(0, 1), ystyle = "percent")
stopifnot(identical(names(s), c("x", "y")), identical(names(s$x), c("limits", "breaks", "labels")))
stopifnot(identical(s$x$limits, c(-0.5, 10.5)), identical(s$x$breaks, c(0, 2, 4, 6, 8, 10)))
stopifnot(identical(s$y$labels[1], "0.0%"))
e <- axis_pair_spec(c(0, 10), c(0, 2), equal = TRUE)
stopifnot(all.equal(diff(e$x$limits), diff(e$y$limits)), all.equal(mean(e$y$limits), 1))

df <- data.frame(a = c(1, 5, NA), b = c(-2, 3, 4), g = c("u", "v", "w"))
l <- data_limits(df, "a", "b")
stopifnot(identical(l, list(xlim = c(1, 5), ylim = c(-2, 4))))
stopifnot(identical(data_limits(df, "a", "b", include = list(x = 0, y = 10))$ylim, c(-2, 10)))
stopifnot(identical(msg(data_limits(df, "zz", "b")), "Column `zz` not found in `data`."))
stopifnot(identical(msg(data_limits(df, "a", "zz")), "Column `zz` not found in `data`."))
stopifnot(identical(msg(data_limits(df, "g", "b")), "Column `g` must be numeric."))
stopifnot(identical(msg(data_limits(df, "a", "g")), "Column `g` must be numeric."))

sp <- list(limits = c(0, 10), breaks = c(0, 5, 10), labels = c("0", "5", "10"))
stopifnot(identical(msg(lw_scale_x(list(limits = 1))),
                    "`spec` must have elements limits, breaks, labels."))
stopifnot(identical(msg(lw_scale_y(list())), "`spec` must have elements limits, breaks, labels."))
stopifnot(identical(msg(lw_scale_x(sp, trans = "log10")), "A log10 x axis needs positive limits."))
stopifnot(identical(msg(lw_scale_y(sp, trans = "sqrt")), "A sqrt y axis needs positive limits."))
