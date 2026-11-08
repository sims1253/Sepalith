msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
wmsg <- function(expr) conditionMessage(tryCatch(expr, warning = function(w) w))
tmp <- tempfile("devices")
dir.create(tmp)
draw <- function() plot(1:3)
boom <- function() stop("no data")

stopifnot(identical(device_size(7, 5), list(width = 7, height = 5)))
stopifnot(identical(device_size(25.4, 50.8, "mm"), list(width = 1, height = 2)))
stopifnot(identical(device_size(5.08, 2.54, "cm")$width, 2))
stopifnot(identical(device_size(300, 150, "px", res = 150), list(width = 2, height = 1)))
stopifnot(identical(msg(device_size(0, 1)), "`width` must be a single positive number."))
stopifnot(identical(msg(device_size(1, -1)), "`height` must be a single positive number."))

stopifnot(identical(format_from_path("a/b.PDF"), "pdf"))
stopifnot(identical(format_from_path("x.jpg"), "jpeg"))
stopifnot(identical(msg(format_from_path("dir/plot")), "Cannot guess a format from `plot`."))

n_dev <- length(dev.list())
p <- save_pdf(draw, file.path(tmp, "a.pdf"), width = 3, height = 3)
stopifnot(file.exists(p), identical(attr(p, "format"), "pdf"), isTRUE(attr(p, "ok")))
p <- save_svg(draw, file.path(tmp, "a.svg"), width = 3, height = 3)
stopifnot(file.exists(p), identical(attr(p, "format"), "svg"))
p <- save_png(draw, file.path(tmp, "new", "a.png"), width = 2, height = 2, res = 50)
stopifnot(file.exists(p), identical(attr(p, "format"), "png"))
p <- save_jpeg(draw, file.path(tmp, "a.jpg"), width = 2, height = 2, res = 50)
stopifnot(file.exists(p), identical(attr(p, "format"), "jpeg"))
stopifnot(identical(msg(save_pdf(1, "x.pdf")), "`plot_fn` must be a function."))
stopifnot(identical(msg(save_png("f", "x.png")), "`plot_fn` must be a function."))
stopifnot(identical(msg(save_pdf(draw, "x.png")), "`path` must end in .pdf."))
stopifnot(identical(msg(save_svg(draw, "x.pdf")), "`path` must end in .svg."))
stopifnot(identical(msg(save_png(draw, "x.svg")), "`path` must end in .png."))
stopifnot(identical(msg(save_jpeg(draw, "x.png")), "`path` must end in .jpeg."))
stopifnot(identical(wmsg(save_pdf(boom, file.path(tmp, "b.pdf"))),
                    "Plot for `b.pdf` failed: no data"))
p <- suppressWarnings(save_svg(boom, file.path(tmp, "b.svg")))
stopifnot(identical(attr(p, "ok"), FALSE))
stopifnot(identical(length(dev.list()), n_dev))

p <- save_plot(draw, file.path(tmp, "sub", "b.png"), width = 100, height = 100, units = "px",
               res = 50)
stopifnot(file.exists(p), identical(attr(p, "format"), "png"))
p <- save_plot(draw, file.path(tmp, "c.svg"), width = 3, height = 3)
stopifnot(identical(attr(p, "format"), "svg"))
stopifnot(identical(msg(save_plot(draw, file.path(tmp, "b.gif"))), "Unsupported format `gif`."))
stopifnot(identical(msg(save_plot(draw, "w.png", width = 100, res = 200)),
                    "`width` is too large: 20000 pixels at 200 dpi."))
stopifnot(identical(msg(save_plot(draw, "h.jpg", height = 80, res = 150)),
                    "`height` is too large: 12000 pixels at 150 dpi."))

plots <- list(one = draw, two = draw)
s <- save_plot_set(plots, file.path(tmp, "set"), format = "jpeg", width = 3, height = 3, res = 30)
stopifnot(identical(s$name, c("one", "two")), identical(basename(s$path), c("one.jpg", "two.jpg")))
stopifnot(all(s$bytes > 0), identical(s$status, c("written", "written")))
plots$three <- boom
s <- suppressWarnings(save_plot_set(plots, file.path(tmp, "set"), format = "jpeg",
                                    width = 3, height = 3, res = 30))
stopifnot(identical(s$status, c("skipped", "skipped", "failed")))
s <- save_plot_set(plots[1], file.path(tmp, "set"), format = "jpeg", overwrite = TRUE,
                   width = 3, height = 3, res = 30)
stopifnot(identical(s$status, "written"))
stopifnot(identical(msg(save_plot_set(list(draw), tmp)),
                    "`plots` must be a list with unique, non-empty names."))

unlink(tmp, recursive = TRUE)
