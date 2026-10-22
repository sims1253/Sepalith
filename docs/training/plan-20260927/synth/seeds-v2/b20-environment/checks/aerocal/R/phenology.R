# skipped: needs dplyr (seasonMetrics, stageOnsets)

# stageFromDd
st <- stageFromDd(c(0, 100, 250, NA), c(budbreak = 100, bloom = 200))
stopifnot(identical(as.character(st), c("dormant", "budbreak", "bloom", NA)), is.ordered(st))
stopifnot(inherits(try(stageFromDd(1, c(a = 2, b = 1)), silent = TRUE), "try-error"))

# onsetDate interpolates and skips missing days
dts <- as.Date("2025-04-01") + 0:4
stopifnot(onsetDate(dts, c(0, 10, 20, 30, 40), 17) == as.Date("2025-04-03"))
stopifnot(onsetDate(dts, c(0, NA, NA, 30, 40), 15) == as.Date("2025-04-03"))
stopifnot(is.na(onsetDate(dts, 1:5, 100)), onsetDate(dts, 1:5, 0) == dts[1])

# smoothVi
sv <- smoothVi(c(1, 1, 1, 0, 1, 1, 1))
stopifnot(length(sv) == 7, all(sv == 1))
stopifnot(inherits(try(smoothVi(1:5, 4), silent = TRUE), "try-error"))

# seasonStart / seasonEnd
doy <- seq(10, 100, by = 10)
vi <- c(0, 0, 2, 4, 6, 8, 6, 4, 2, 0)
stopifnot(seasonStart(doy, vi) == 40, seasonEnd(doy, vi) == 80)
stopifnot(is.na(seasonStart(doy, rep(NA_real_, 10))), is.na(seasonEnd(doy, rep(NA_real_, 10))))

# growingSeason
dd <- seq(as.Date("2025-01-01"), as.Date("2025-12-31"), by = "day")
doyn <- as.integer(format(dd, "%j"))
tm <- ifelse(doyn >= 100 & doyn < 280, 12, 0)
gs <- growingSeason(dd, tm)
stopifnot(gs$start == as.Date("2025-04-10"), gs$end == as.Date("2025-10-06"), gs$length == 180)
