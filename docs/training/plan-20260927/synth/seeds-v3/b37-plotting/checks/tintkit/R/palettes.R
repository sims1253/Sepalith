msg = function(expr) conditionMessage(tryCatch(expr, error = function(e) e))

stopifnot(isTRUE(checkPaletteArgs(3, 10, -1, 0.5)))
stopifnot(identical(msg(checkPaletteArgs(1.5, 0, 1, 1)),
                    "`n` must be a single positive whole number."))
stopifnot(identical(msg(checkPaletteArgs(2, 361, 1, 1)),
                    "`hue` must be a single number between 0 and 360."))
stopifnot(identical(msg(checkPaletteArgs(2, 0, 0, 1)), "`direction` must be 1 or -1."))
stopifnot(identical(msg(checkPaletteArgs(2, 0, 1, 2)),
                    "`alpha` must be a single number between 0 and 1."))

f = finishPalette(c("#000000", "#FFFFFF"), -1, 1, "x")
stopifnot(identical(unclass(f), structure(c("#FFFFFF", "#000000"), type = "x")))
stopifnot(identical(unclass(finishPalette("#000000", 1, 0.5, "x"))[1], "#00000080"))

s = tintSequential(3, hue = 0)
stopifnot(inherits(s, "tint_palette"), identical(attr(s, "type"), "sequential"))
stopifnot(identical(s[[1]], grDevices::hcl(0, 15, 92)))
stopifnot(identical(s[[3]], grDevices::hcl(0, 60, 25)))
stopifnot(identical(unclass(tintSequential(3, hue = 0, direction = -1))[3], s[[1]]))
stopifnot(identical(tintSequential(1, hue = 0, lum = c(80, 40))[[1]], grDevices::hcl(0, 37.5, 60)))
stopifnot(identical(msg(tintSequential(2, lum = c(10, 120))),
                    "`lum` must be two numbers between 0 and 100."))
stopifnot(identical(msg(tintSequential(2, chroma = -1)),
                    "`chroma` must be two non-negative numbers."))

d = tintDiverging(3, hue = 0)
stopifnot(identical(d[[2]], grDevices::hcl(180, 15, 92)), identical(attr(d, "type"), "diverging"))
stopifnot(identical(d[[1]], grDevices::hcl(0, 60, 25)))
stopifnot(identical(d[[3]], grDevices::hcl(180, 60, 25)))
stopifnot(identical(tintDiverging(1, hue = 0)[[1]], grDevices::hcl(180, 15, 92)))
stopifnot(identical(msg(tintDiverging(2, lum = 50)),
                    "`lum` must be two numbers between 0 and 100."))
stopifnot(identical(msg(tintDiverging(2, chroma = "a")),
                    "`chroma` must be two non-negative numbers."))

q = tintQualitative(2, hue = 10, alpha = 0.5)
stopifnot(identical(nchar(unclass(q)), c(9L, 9L)), identical(attr(q, "type"), "qualitative"))
stopifnot(identical(unclass(tintQualitative(4, hue = 0))[1:4],
                    grDevices::hcl(c(0, 90, 180, 270), 60, 60)))
stopifnot(identical(msg(tintQualitative(2, lum = c(1, 2))),
                    "`lum` must be a single number between 0 and 100."))
stopifnot(identical(msg(tintQualitative(2, chroma = -5)),
                    "`chroma` must be a single non-negative number."))
stopifnot(identical(msg(tintQualitative(0)), "`n` must be a single positive whole number."))

stopifnot(identical(tintPalette("sequential", 3, hue = 0), s))
stopifnot(identical(tintPalette("diverging", 3, hue = 0), d))
stopifnot(identical(tintPalette("qualitative", 2, hue = 10, alpha = 0.5), q))

tab = paletteTable(c("sequential", "qualitative"), n = 2L)
stopifnot(identical(names(tab), c("type", "index", "colour")), nrow(tab) == 4L)
stopifnot(identical(tab$type, c("sequential", "sequential", "qualitative", "qualitative")))
stopifnot(identical(tab$colour[1:2], as.vector(unclass(tintSequential(2)))))
