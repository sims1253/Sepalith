# skipped: needs data.table (runElo, eloTable, eloForecast)

# expectedScore
stopifnot(all.equal(expectedScore(1500, 1500, homeAdvantage = 0), 0.5))
stopifnot(all.equal(expectedScore(1900, 1500, homeAdvantage = 0), 10 / 11))
stopifnot(expectedScore(1500, 1500) > 0.5)

# matchScore
stopifnot(identical(matchScore(c(2, 1, 0), c(1, 1, 3)), c(1, 0.5, 0)))
stopifnot(is.na(matchScore(NA, 1)))

# marginMultiplier
stopifnot(all.equal(marginMultiplier(3, 0), log(4) + 1))
stopifnot(identical(marginMultiplier(c(1, 2, 5), c(0, 0, 0), "linear"), c(1, 1.5, 2)))
stopifnot(identical(marginMultiplier(c(4, 0), c(0, 0), "none"), c(1, 1)))

# eloDelta
d <- eloDelta(1500, 1500, 1, 0, k = 20, homeAdvantage = 0, method = "none")
stopifnot(all.equal(d, 10))
stopifnot(eloDelta(1500, 1500, 0, 1, homeAdvantage = 0) < 0)
stopifnot(all.equal(eloDelta(1500, 1500, 1, 1, homeAdvantage = 0), 0))
