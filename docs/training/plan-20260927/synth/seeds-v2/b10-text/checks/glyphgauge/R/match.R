# skipped: needs stringi: matchKey
# skipped: isEmail, isTicketId and isIsoDate need extractPatterns() from
# R/extract.R; jaccardSim (the default simFun) lives in R/set-similarity.R

same = function(a, b) as.numeric(tolower(a) == tolower(b))

# simMatrix
m = simMatrix(c("a", "B"), c("b", "c"), simFun = same)
stopifnot(identical(dim(m), c(2L, 2L)), m["B", "b"] == 1, m["a", "c"] == 0)

# closestMatch
cm = closestMatch(c("A", "z"), c("x", "a"), simFun = same)
stopifnot(identical(cm$match, c("a", NA)), identical(cm$similarity, c(1, 0)))
stopifnot(all(is.na(closestMatch("a", character(), simFun = same)$match)))

# groupNear
stopifnot(identical(groupNear(c("a", "A", "b", "B"), simFun = same), c(1L, 1L, 2L, 2L)))
stopifnot(identical(groupNear(c("a", NA, "a"), simFun = same), c(1L, 2L, 1L)))
