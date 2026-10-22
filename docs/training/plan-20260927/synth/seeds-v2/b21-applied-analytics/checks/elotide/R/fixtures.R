# skipped: needs data.table (readResultsCsv, readResultsScores, validateResults,
# roundRobin, splitSeason)

# cleanTeamNames
stopifnot(identical(cleanTeamNames("  Man   Utd "), "Man Utd"))
al <- c("Man Utd" = "Manchester United")
stopifnot(identical(cleanTeamNames(c("Man  Utd", "Leeds"), al), c("Manchester United", "Leeds")))
stopifnot(identical(cleanTeamNames(character()), character()))
