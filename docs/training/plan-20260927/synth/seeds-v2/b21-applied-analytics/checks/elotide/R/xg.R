# skipped: needs data.table (xgByMatch, xgForByTeam, xgAgainstByTeam)

# shotDistance
stopifnot(all.equal(shotDistance(105, 34), 0), all.equal(shotDistance(94, 34), 11))
stopifnot(all.equal(shotDistance(102, 30), 5))

# shotAngle
stopifnot(all.equal(shotAngle(94, 34), 2 * atan(3.66 / 11)))
stopifnot(shotAngle(94, 34) > shotAngle(94, 10), shotAngle(80, 34) < shotAngle(94, 34))

# shotXg
shots <- data.frame(
  x = c(94, 94, 70, 99),
  y = c(34, 34, 20, 34),
  bodyPart = c("foot", "foot", "head", NA),
  situation = c("open", "penalty", "open", NA)
)
xg <- shotXg(shots)
stopifnot(length(xg) == 4, xg[2] == 0.76, all(!is.na(xg)), xg[3] < xg[1])

# simulateFromXg
p <- simulateFromXg(c(0.5, 0.5), numeric(0), nSims = 2000L, seed = 1)
stopifnot(all.equal(sum(p[c("home", "draw", "away")]), 1), p[["away"]] == 0)
stopifnot(all.equal(p[["xPtsHome"]], 3 * p[["home"]] + p[["draw"]]))
sure <- simulateFromXg(1, 0, nSims = 10L)
stopifnot(sure[["home"]] == 1)
