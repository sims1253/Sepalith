msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
# skipped: needs ggplot2 and scales (success paths of the scale_lw_*() family)
is_hex <- function(x) all(grepl("^#[0-9A-F]{6}$", x))

stopifnot(is.null(check_palette_args(3, 100, -1)))
stopifnot(identical(msg(check_palette_args(0, 1, 1)),
                    "`n` must be a single number of at least 1."))
stopifnot(identical(msg(check_palette_args(2, 400, 1)),
                    "`hue` must be a single number between 0 and 360."))
stopifnot(identical(msg(check_palette_args(2, 1, 2)), "`direction` must be 1 or -1."))

s <- lw_palette("sequential", 5)
stopifnot(length(s) == 5L, is_hex(s), identical(s[1], grDevices::hcl(220, 20, 90)))
stopifnot(identical(lw_palette("sequential", 5, direction = -1), rev(s)))
d <- lw_palette("diverging", 5, hue = 10)
stopifnot(identical(d[3], grDevices::hcl(190, 20, 90)))
stopifnot(identical(d[1], grDevices::hcl(10, 70, 30)))
q <- lw_palette("qualitative", 4, hue = 0)
stopifnot(identical(q, grDevices::hcl(c(0, 90, 180, 270), 70, 60)))
cy <- lw_palette("cyclic", 3, hue = 0)
stopifnot(identical(cy[1], cy[3]), identical(cy[2], grDevices::hcl(180, 70, 60)))

pdf(NULL)
stopifnot(identical(palette_preview("diverging", 5, hue = 10), d))
stopifnot(length(palette_preview(n = 2, labels = FALSE)) == 2L)
invisible(dev.off())

stopifnot(identical(msg(scale_lw_sequential(na.value = NA)),
                    "`na.value` must be a single colour."))
stopifnot(identical(msg(scale_lw_diverging(na.value = 1)), "`na.value` must be a single colour."))
stopifnot(identical(msg(scale_lw_diverging(midpoint = Inf)),
                    "`midpoint` must be a single finite number."))
stopifnot(identical(msg(scale_lw_binned(n_bins = 1)),
                    "`n_bins` must be a single number of at least 2."))
stopifnot(identical(msg(scale_lw_binned(na.value = c("a", "b"))),
                    "`na.value` must be a single colour."))
stopifnot(identical(msg(scale_lw_qualitative(hue = -5)),
                    "`hue` must be a single number between 0 and 360."))
stopifnot(grepl("should be one of", msg(scale_lw_qualitative(aesthetics = "size")), fixed = TRUE))
stopifnot(identical(msg(scale_lw_sequential(limits = 1)), "`limits` must be NULL or two numbers."))
stopifnot(identical(msg(scale_lw_diverging(limits = "a")),
                    "`limits` must be NULL or two numbers."))
stopifnot(identical(msg(scale_lw_binned(limits = 1:3)), "`limits` must be NULL or two numbers."))
stopifnot(identical(msg(scale_lw_qualitative(drop = NA)), "`drop` must be TRUE or FALSE."))
