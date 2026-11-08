msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
# skipped: needs ggplot2 (success paths of lw_points, lw_lines, lw_bars, lw_boxes, lw_smooth)

stopifnot(identical(layer_defaults("point"), list(size = 2, alpha = 0.8, shape = 19L)))
stopifnot(identical(layer_defaults("line", size = 4)$linewidth, 1))
stopifnot(identical(layer_defaults("bar")$colour, NA))
stopifnot(identical(layer_defaults("bar", width = 1)$width, 1))
stopifnot(identical(layer_defaults("boxplot", width = 1, size = 4)[c("width", "outlier.size")],
                    list(width = 0.75, outlier.size = 2)))
stopifnot(identical(layer_defaults("smooth", alpha = 0.6)$alpha,
                    0.3), isTRUE(layer_defaults("smooth")$se))
stopifnot(identical(layer_defaults("text")$size, 3), identical(layer_defaults("text")$vjust, -0.5))

m <- merge_params(list(size = 2, alpha = 0.8), list(alpha = 1, colour = "red"))
stopifnot(identical(m, list(size = 2, alpha = 1, colour = "red")))
stopifnot(identical(merge_params(list(size = 2, alpha = 1), list(size = NULL)), list(alpha = 1)))
stopifnot(identical(merge_params(list(a = 1), list()), list(a = 1)))
stopifnot(identical(msg(merge_params(list(), list(1))), "All layer parameters must be named."))

stopifnot(identical(build_layers(list()), list()))
stopifnot(identical(msg(build_layers(list(list(geom = "violin")))),
                    "No layer builder for geom `violin`."))

# Input checks run before any ggplot2 call
stopifnot(identical(msg(lw_points(mapping = list(x = 1))),
                    "`mapping` must be created with `aes()`."))
stopifnot(identical(msg(lw_lines(data = 1:3)), "`data` must be a data frame or NULL."))
stopifnot(identical(msg(lw_bars(mapping = "x")), "`mapping` must be created with `aes()`."))
stopifnot(grepl("should be one of", msg(lw_bars(position = "fill")), fixed = TRUE))
stopifnot(identical(msg(lw_boxes(data = list(a = 1))), "`data` must be a data frame or NULL."))
stopifnot(identical(msg(lw_smooth(mapping = 1)), "`mapping` must be created with `aes()`."))
