msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
pdf(NULL)
par(xaxs = "i", yaxs = "i")
plot.new()
plot.window(xlim = c(0, 10), ylim = c(0, 10))

stopifnot(identical(need_fields(list(a = 1), "a", "t"), list(a = 1)))
stopifnot(identical(msg(need_fields(list(), c("x", "y"), "text")),
                    "A `text` annotation needs: x, y."))
stopifnot(identical(msg(need_fields(list(x = "1"), "x", "vline")),
                    "Field `x` of a `vline` annotation must be numeric."))

anns <- list(
  annotation("hline", y = 5),
  annotation("vline", x = 2),
  annotation("band_x", from = 3, to = 1),
  annotation("band_y", from = 4, to = 6),
  annotation("text", x = 5, y = 5, label = 1),
  annotation("arrow", x0 = 1, y0 = 1, x1 = 3, y1 = 3)
)
stopifnot(all(vapply(anns, inherits, logical(1), "plotloom_annotation")))
stopifnot(identical(anns[[3]]$from, 1), identical(anns[[3]]$to, 3), identical(anns[[4]]$to, 6))
stopifnot(identical(anns[[5]]$label, "1"))
stopifnot(identical(msg(annotation("hline")), "A `hline` annotation needs: y."))
stopifnot(identical(msg(annotation("vline")), "A `vline` annotation needs: x."))
stopifnot(identical(msg(annotation("band_x", from = 1)), "A `band_x` annotation needs: to."))
stopifnot(identical(msg(annotation("band_y", to = 1)), "A `band_y` annotation needs: from."))
stopifnot(identical(msg(annotation("text", x = 1, y = 2)), "A `text` annotation needs: label."))
stopifnot(identical(msg(annotation("arrow", x0 = 1)), "A `arrow` annotation needs: y0, x1, y1."))

stopifnot(identical(draw_annotation(anns[[1]]), "hline"))
stopifnot(identical(msg(draw_annotation(list(type = "hline"))),
                    "`ann` must be a plotloom_annotation."))
stopifnot(identical(draw_annotations(anns),
                    c("hline", "vline", "band_x", "band_y", "text", "arrow")))
stopifnot(identical(draw_annotations(list()), character()))

stopifnot(identical(annotate_centre(c(1, 3), c(2, 6, NA)), c(x = 2, y = 4)))
stopifnot(identical(annotate_centre(c(1, 3), 1, which = "x"), c(x = 2)))
stopifnot(identical(annotate_centre(1, c(1, 2, 9), which = "y", stat = "median"), c(y = 2)))

e <- annotate_extremes(1:5, c(3, 1, 5, 4, 2), n = 2L)
stopifnot(identical(e$min, c(2L, 5L)), identical(e$max, c(3L, 4L)))
e <- annotate_extremes(1:3, c(2, NA, 1), n = 5L)
stopifnot(identical(e$min, c(3L, 1L)), identical(e$max, c(1L, 3L)))
stopifnot(identical(annotate_extremes(1:3, 3:1, which = "min"), list(min = 3L, max = integer())))
stopifnot(identical(annotate_extremes(1:3, 3:1, which = "max"), list(min = integer(), max = 1L)))
stopifnot(identical(msg(annotate_extremes(1:2, 1:3)),
                    "`x`, `y` and `labels` must have the same length."))

stopifnot(identical(shade_interval(1, 2), c(1, 2)))
stopifnot(identical(shade_interval(3, Inf, axis = "y", label = "ok"), c(3, 10)))
stopifnot(identical(shade_interval(-Inf, 4, label = "ok"), c(0, 4)))
stopifnot(identical(msg(shade_interval(2, 1)), "`lower` must not be greater than `upper`."))
stopifnot(identical(msg(shade_interval(NA, 1)), "`lower` must be a single number."))
stopifnot(identical(msg(shade_interval(1, "a")), "`upper` must be a single number."))

invisible(dev.off())
