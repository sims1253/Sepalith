msg = function(expr) conditionMessage(tryCatch(expr, error = function(e) e))

m = hexToRgb(c("#FF8000", "#00ff0080", "#F80"))
stopifnot(identical(colnames(m), c("red", "green", "blue", "alpha")))
stopifnot(identical(m[1, ], c(red = 255L, green = 128L, blue = 0L, alpha = 255L)))
stopifnot(identical(unname(m[2, c("green", "alpha")]), c(255L, 128L)))
stopifnot(identical(unname(m[3, ]), c(255L, 136L, 0L, 255L)))
stopifnot(identical(msg(hexToRgb(c("#FFFF", "red"))), "Invalid hex colour(s): #FFFF, red."))

stopifnot(identical(rgbToHex(m), c("#FF8000", "#00FF0080", "#FF8800")))
stopifnot(identical(rgbToHex(cbind(red = 300, green = -5, blue = 15.6)), "#FF0010"))
stopifnot(identical(msg(rgbToHex(1:3)), "`rgb` must be a matrix with columns red, green, blue."))

stopifnot(identical(mixColours("#000000", "#FFFFFF"), "#808080"))
stopifnot(identical(mixColours(c("#000000", "#FF0000"), "#FFFFFF", weight = 0),
                    c("#000000", "#FF0000")))
stopifnot(identical(msg(mixColours("#000000", "#FFFFFF", 2)), "`weight` must be between 0 and 1."))

stopifnot(identical(lighten("#000000", 1), "#FFFFFF"))
stopifnot(identical(lighten("#000000", 0.5), "#808080"))
stopifnot(identical(darken("#FFFFFF", 1), "#000000"), identical(darken("#FF0000", 0), "#FF0000"))
stopifnot(identical(msg(lighten("#000000", -1)),
                    "`amount` must be a single number between 0 and 1."))
stopifnot(identical(msg(darken("#000000", 2)),
                    "`amount` must be a single number between 0 and 1."))

stopifnot(identical(adjustChannels("#808080", red = 2), "#FF8080"))
stopifnot(identical(adjustChannels("#808080", green = 0.5, blue = 0), "#804000"))
stopifnot(identical(msg(adjustChannels("#808080", red = -1)),
                    "`red` must be a single non-negative number."))
stopifnot(identical(msg(adjustChannels("#808080", green = "a")),
                    "`green` must be a single non-negative number."))
stopifnot(identical(msg(adjustChannels("#808080", blue = 1:2)),
                    "`blue` must be a single non-negative number."))

stopifnot(identical(relativeLuminance(c("#000000", "#FFFFFF")), c(0, 1)))
stopifnot(abs(relativeLuminance("#FF0000") - 0.2126) < 1e-12)

stopifnot(identical(contrastReport("#FFFFFF")$ratio, 1))
stopifnot(identical(contrastReport("#FFFFFF", "#000000")$ratio, 21))
stopifnot(identical(contrastReport("#767676")$ratio, 4.54))

r = contrastReport(c("#000000", "#767676", "#AAAAAA"))
stopifnot(identical(names(r), c("foreground", "background", "ratio", "AA_large", "AA", "AAA")))
stopifnot(identical(r$ratio[1], 21), identical(r$AA, c(TRUE, TRUE, FALSE)))
stopifnot(identical(r$AAA, c(TRUE, FALSE, FALSE)), identical(r$AA_large, c(TRUE, TRUE, FALSE)))
stopifnot(identical(contrastReport("#FFFFFF", c("#000000", "#FFFFFF"))$AA, c(TRUE, FALSE)))
stopifnot(identical(msg(contrastReport("#000000", levels = c(AA = 4.5))),
                    "`levels` must be a numeric vector named AA_large, AA, AAA."))
