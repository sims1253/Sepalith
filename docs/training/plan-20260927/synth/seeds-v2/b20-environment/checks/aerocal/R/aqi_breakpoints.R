# aqiLevels
lv <- aqiLevels()
stopifnot(nrow(lv) == 6, lv$indexHi[6] == 500, lv$category[1] == "Good")

# newBreakpoints
nb <- newBreakpoints("x", "ppb", c(0, 11), c(10, 20), digits = 0)
stopifnot(inherits(nb, "aq_breakpoints"), identical(nb$indexHi, c(50, 100)))
stopifnot(attr(nb, "digits") == 0)
stopifnot(inherits(try(newBreakpoints("x", "u", 1:7, 2:8, 0), silent = TRUE), "try-error"))

# per-pollutant tables and getBreakpoints
stopifnot(nrow(breakpointsPm25()) == 6, attr(breakpointsPm25(), "digits") == 1)
stopifnot(nrow(breakpointsO3()) == 5, breakpointsNo2()$concHi[1] == 53)
stopifnot(identical(getBreakpoints("pm10"), breakpointsPm10()))
stopifnot(inherits(try(getBreakpoints("so2"), silent = TRUE), "try-error"))

# validateBreakpoints
for (p in c("pm25", "pm10", "o3", "no2")) {
  stopifnot(isTRUE(validateBreakpoints(getBreakpoints(p))))
}
gappy <- breakpointsPm10()
gappy$concLo[3] <- 160
stopifnot(inherits(try(validateBreakpoints(gappy), silent = TRUE), "try-error"))

# reviseBreakpoints keeps neighbours contiguous
rv <- reviseBreakpoints(breakpointsPm25(), "Moderate", 12.1, 35.4)
stopifnot(abs(rv$concHi[1] - 12.0) < 1e-9, rv$concLo[2] == 12.1)
stopifnot(inherits(try(reviseBreakpoints(rv, "Nope", 1, 2), silent = TRUE), "try-error"))
