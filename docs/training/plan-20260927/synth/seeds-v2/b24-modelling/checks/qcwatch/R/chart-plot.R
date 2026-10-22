grDevices::pdf(NULL)
stopifnot(isTRUE(all.equal(chart.ylim(c(1, 2), 0, 10), c(-0.5, 10.5))))
stopifnot(isTRUE(all.equal(chart.ylim(c(5, 5), 5, 5, pad = 0.1), c(4.5, 5.5))))
stopifnot(isTRUE(all.equal(chart.ylim(c(NA, 1, Inf), 0, 2), c(-0.1, 2.1))))

shew <- structure(list(stat = c(1, 5, 2), center = 2, lcl = rep(0, 3), ucl = rep(4, 3),
  type = "xbar", signals = 2L), class = "qc_shewhart")
stopifnot(identical(plot(shew), shew))
cus <- structure(list(upper = c(0, 2, 6), lower = c(1, 0, 0), h = 5,
  signals.upper = 3L, signals.lower = integer(0)), class = "qc_cusum")
stopifnot(identical(plot(cus, main = "c"), cus))
ew <- structure(list(x = c(1, 3, 2), stat = c(1, 2, 2), lcl = c(0, -1, -2), ucl = c(2, 3, 4),
  target = 1, lambda = 0.2, signals = integer(0)), class = "qc_ewma")
stopifnot(identical(plot(ew), ew))
stopifnot(is.null(mark.signals(1:3, 1:3, integer(0))), is.null(draw.limits(1:3, 0, -1, 1)))
grDevices::dev.off()
