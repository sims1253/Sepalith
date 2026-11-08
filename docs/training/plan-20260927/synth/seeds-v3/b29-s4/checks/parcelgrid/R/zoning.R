r <- zoning_rules()
stopifnot(nrow(r) == 5L, identical(r$prefix[5], "MX"), r$min_lot_m2[r$prefix == "I"] == 2000)

z <- zoning_code(c("r-2", "C 1", "mx", "A"))
stopifnot(is(z, "ZoningCode"), identical(z@code, c("R2", "C1", "MX", "A")))
stopifnot(inherits(try(zoning_code("R22"), silent = TRUE), "try-error"))
stopifnot(identical(zoning_code(c("res2", "COM-1", "ind", "ag", "mu 3"))@code,
  c("R2", "C1", "I", "A", "MX3")))
stopifnot(inherits(try(zoning_code(2), silent = TRUE), "try-error"))

stopifnot(isTRUE(valid_zoning_code(z)))
bad <- z
bad@code <- c("R2", "r3", "123")
stopifnot(identical(valid_zoning_code(bad), "Malformed zoning code(s): r3, 123."))
bad@code <- c("R2", "Q1", "ZZ")
stopifnot(identical(valid_zoning_code(bad), "Unknown zoning prefix(es): Q, ZZ."))

info <- zoning_lookup(z)
stopifnot(identical(info$density, c(2L, 1L, 1L, 1L)),
  identical(info$prefix, c("R", "C", "MX", "A")))
stopifnot(identical(info$category[3], "mixed"), info$far[2] == 1.5, info$min_lot_m2[4] == 10000)

lim <- zoning_limits(z, c(400, 400, 400, 400))
stopifnot(identical(names(lim), c("code", "max_floor_m2", "max_height_m", "max_units", "parking",
  "buildable")))
stopifnot(identical(lim$max_floor_m2, c(400, 600, 800, 20)))
stopifnot(identical(lim$max_height_m, c(20, 25, 30, 12)))
stopifnot(identical(lim$max_units, c(2, 0, 2, 0)), identical(lim$parking, c(4, 0, 12, 0)))
stopifnot(identical(zoning_limits(zoning_code("I1"), 1000)$parking, 0),
  identical(zoning_limits(z, rep(600, 4))$parking[2], 23))
stopifnot(inherits(try(zoning_limits(z, rep("1", 4)), silent = TRUE), "try-error"))
stopifnot(identical(lim$buildable, c(TRUE, FALSE, TRUE, FALSE)))
stopifnot(inherits(try(zoning_limits(z, 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(zoning_limits(z, c(1, 1, 0, 1)), silent = TRUE), "try-error"))

stopifnot(identical(convert_zoning(z), z@code), identical(as(z, "character"), z@code))
f <- as(z, "factor")
stopifnot(is.factor(f), nlevels(f) == 5L, identical(as.character(f), c("residential", "commercial",
  "mixed", "agricultural")))
stopifnot(identical(as(z, "integer"), c(R2 = 2L, C1 = 1L, MX = 1L, A = 1L)))
l <- as(z, "list")
stopifnot(identical(names(l), c("residential", "commercial", "agricultural", "mixed")))
stopifnot(identical(l$mixed, "MX"), identical(as("i3", "ZoningCode")@code, "I3"))
zdf <- as(z, "data.frame")
stopifnot(is.data.frame(zdf), nrow(zdf) == 4L, identical(zdf, convert_zoning(z, "data.frame")))
stopifnot(zdf$units_per_ha[1] == 25)

out <- capture.output(res <- show_zoning(zoning_code(c("R1", "R2", "I1"))))
stopifnot(length(out) == 3L, grepl("3 codes", out[1]), grepl("residential +2", out[2]))
stopifnot(grepl("industrial +1", out[3]), length(capture.output(show(z))) == 5L)
