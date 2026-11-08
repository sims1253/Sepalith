r = matrix(c("E", "E", "N",
             "E", "L", "O",
             "L", "O", "O",
             "O", "N", "E"), nrow = 4, byrow = TRUE,
           dimnames = list(c("ann", "bob", "cy", "dee"), c("d1", "d2", "d3")))
cv = coverageByShift(r)
stopifnot(nrow(cv) == 9, identical(cv$shift[1:3], c("E", "L", "N")))
stopifnot(identical(cv$staffed[cv$day == "d1"], c(2, 1, 0)))
stopifnot(identical(cv$gap[cv$day == "d2"], c(-1, -1, 0)), identical(cv$required[3], 1))
unnamed = r
dimnames(unnamed) = NULL
stopifnot(identical(unique(coverageByShift(unnamed, 1, 1, 1)$day), c("d1", "d2", "d3")))

g = coverageGaps(cv, "shortfall")
stopifnot(all(g$gap < 0), all(g$status == "short"), nrow(g) == 6)
stopifnot(nrow(coverageGaps(coverageByShift(r, 1, 0, 0), "surplus")) == 5)
stopifnot(nrow(coverageGaps(cv, "both")) == 6)
cr = coverageGaps(cv, "critical")
stopifnot(identical(cr$day, c("d1", "d3")), identical(cr$shift, c("N", "L")))
stopifnot(nrow(coverageGaps(cv, "critical", critical = 1)) == 6)

big = cbind(rbind(r, eve = c("O", "O", "O"), fay = c("O", "O", "O")))
f = fillGaps(big)
cf = coverageByShift(f)
stopifnot(all(cf$gap >= 0), attr(f, "added") == sum(-pmin(coverageByShift(big)$gap, 0)))
stopifnot(!any(f[5:6, 1:2] == "N" & f[5:6, 2:3] == "E"))
f2 = fillGaps(big, maxShifts = 2)
stopifnot(all(rowSums(f2[c("eve", "fay"), ] != "O") <= 2), attr(f2, "added") < attr(f, "added"))

cs = coverageSummary(cv)
stopifnot(identical(cs$shift, c("E", "L", "N")), identical(cs$daysShort, c(2L, 3L, 1L)))
stopifnot(identical(cs$headsShort, c(2, 4, 1)), cs$worstDay[2] == "d3")
stopifnot(isTRUE(all.equal(cs$fillRate, c(4 / 6, 2 / 6, 2 / 3))))
cs0 = coverageSummary(coverageByShift(r, 1, 1, 0))
stopifnot(cs0$fillRate[3] == 1, cs0$headsOver[3] == 2, cs0$daysOver[1] == 1)

sk = data.frame(staff = c("ann", "bob", "cy"), forklift = c(TRUE, FALSE, TRUE),
  firstAid = c(FALSE, TRUE, FALSE), supervisor = c(FALSE, FALSE, TRUE),
  qa = c(TRUE, TRUE, FALSE))
s1 = skillCoverage(c("ann", "bob"), sk)
stopifnot(identical(s1$present, c(1L, 1L, 0L, 2L)), identical(s1$met, c(TRUE, TRUE, FALSE, TRUE)))
stopifnot(s1$holders[4] == "ann, bob", identical(s1$needed, c(1, 1, 1, 0)))
s2 = suppressWarnings(skillCoverage(c("cy", "zed"), sk, needs = c(forklift = 2, firstAid = 0,
  supervisor = 1, qa = 1)))
stopifnot(identical(s2$met, c(FALSE, TRUE, TRUE, FALSE)))
stopifnot(identical(tryCatch(skillCoverage("zed", sk), warning = function(w) "w"), "w"))
