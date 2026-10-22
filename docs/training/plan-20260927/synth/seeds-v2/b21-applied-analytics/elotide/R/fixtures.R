#' Standardise team names
#'
#' Trims whitespace, collapses repeated spaces and applies an alias table so
#' that "Man Utd" and "Manchester United" end up as the same team.
#'
#' @param x Character vector of raw team names.
#' @param aliases Named character vector mapping aliases to canonical names.
#' @return A character vector of canonical team names.
#' @export
cleanTeamNames <- function(x, aliases = character()) {
  x <- trimws(gsub("\\s+", " ", x))
  hit <- x %in% names(aliases)
  x[hit] <- unname(aliases[x[hit]])
  x
}

#' Read results from a semicolon-separated export
#'
#' Expects columns `Date`, `HomeTeam`, `AwayTeam`, `FTHG` and `FTAG` with day
#' first dates, as in typical betting-data exports.
#'
#' @param path Path to the file.
#' @param aliases Named character vector mapping aliases to canonical names.
#' @return A data.table of results with columns `date`, `home`, `away`,
#'   `goalsHome` and `goalsAway`.
#' @export
readResultsCsv <- function(path, aliases = character()) {
  raw <- data.table::fread(path, sep = ";", encoding = "UTF-8")
  needed <- c("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
  if (!all(needed %in% names(raw))) {
    stop("File lacks columns: ", paste(setdiff(needed, names(raw)), collapse = ", "))
  }

  out <- raw[, .(
    date = as.Date(Date, format = "%d/%m/%Y"),
    home = cleanTeamNames(HomeTeam, aliases),
    away = cleanTeamNames(AwayTeam, aliases),
    goalsHome = as.integer(FTHG),
    goalsAway = as.integer(FTAG)
  )]
  validateResults(out)
}

#' Read results from a score-string export
#'
#' Expects columns `kickoff`, `fixture` (as `"Home v Away"`) and `score` (as
#' `"2-1"`); postponed matches have an empty score.
#'
#' @param path Path to the file.
#' @param aliases Named character vector mapping aliases to canonical names.
#' @return A data.table of results with columns `date`, `home`, `away`,
#'   `goalsHome` and `goalsAway`.
#' @export
readResultsScores <- function(path, aliases = character()) {
  raw <- data.table::fread(path, sep = ",", encoding = "UTF-8")
  needed <- c("kickoff", "fixture", "score")
  if (!all(needed %in% names(raw))) {
    stop("File lacks columns: ", paste(setdiff(needed, names(raw)), collapse = ", "))
  }

  teams <- data.table::tstrsplit(raw$fixture, " v ", fixed = TRUE)
  # Postponed matches have an empty score; parse only well-formed "h-a"
  # strings so a file with no played match at all still reads cleanly.
  score <- gsub("\\s", "", as.character(raw$score))
  played <- grepl("^[0-9]+-[0-9]+$", score)
  goalsHome <- ifelse(played, sub("-.*$", "", score), NA_character_)
  goalsAway <- ifelse(played, sub("^.*-", "", score), NA_character_)

  out <- data.table::data.table(
    date = as.Date(substr(raw$kickoff, 1, 10)),
    home = cleanTeamNames(teams[[1]], aliases),
    away = cleanTeamNames(teams[[2]], aliases),
    goalsHome = as.integer(goalsHome),
    goalsAway = as.integer(goalsAway)
  )
  validateResults(out)
}

#' Check a results table for common data problems
#'
#' Flags self-matches, duplicated fixtures and negative scores; drops rows
#' without a date.
#'
#' @param results A data.table of results with columns `date`, `home`, `away`,
#'   `goalsHome` and `goalsAway`.
#' @return `results` with undated rows removed.
#' @noRd
validateResults <- function(results) {
  undated <- is.na(results$date)
  if (any(undated)) {
    warning(sum(undated), " row(s) without a valid date were dropped.")
    results <- results[!undated]
  }

  if (any(results$home == results$away, na.rm = TRUE)) {
    stop("Results contain a team playing itself.")
  }
  dupes <- duplicated(results, by = c("date", "home", "away"))
  if (any(dupes)) {
    warning(sum(dupes), " duplicated fixture(s) found; keeping the first.")
    results <- results[!dupes]
  }

  negative <- results$goalsHome < 0 | results$goalsAway < 0
  if (any(negative, na.rm = TRUE)) {
    stop("Results contain negative scores.")
  }
  results
}

#' Generate a double round-robin schedule
#'
#' Uses the circle method: one team stays fixed while the others rotate. The
#' second half of the season mirrors the first with venues swapped.
#'
#' @param teams Character vector of team names.
#' @param startDate Date of the first matchday.
#' @param daysBetween Days between consecutive matchdays.
#' @return A data.table with columns `round`, `date`, `home` and `away`.
#' @export
roundRobin <- function(teams, startDate = Sys.Date(), daysBetween = 7L) {
  # With an odd number of teams, an NA placeholder gives one team a bye
  # each round; its pairings are dropped below.
  if (length(teams) %% 2 == 1) {
    teams <- c(teams, NA_character_)
  }

  n <- length(teams)
  nRounds <- n - 1
  rotating <- teams[-1]
  rounds <- vector("list", 2 * nRounds)

  for (r in seq_len(nRounds)) {
    lineup <- c(teams[1], rotating)
    first <- lineup[seq_len(n / 2)]
    second <- rev(lineup)[seq_len(n / 2)]

    # Alternate venues so the fixed team does not play at home every week.
    if (r %% 2 == 0) {
      pairs <- data.table::data.table(home = second, away = first)
    } else {
      pairs <- data.table::data.table(home = first, away = second)
    }
    pairs <- pairs[!is.na(home) & !is.na(away)]
    rounds[[r]] <- pairs[, round := r]
    rounds[[r + nRounds]] <- pairs[, .(home = away, away = home, round = r + nRounds)]

    rotating <- c(rotating[length(rotating)], rotating[-length(rotating)])
  }

  out <- data.table::rbindlist(rounds, use.names = TRUE)
  out[, date := as.Date(startDate) + (round - 1L) * daysBetween]
  data.table::setcolorder(out, c("round", "date", "home", "away"))
  out[]
}

#' Split a season into played and remaining fixtures
#'
#' Results with both scores present count as played; everything else is
#' returned as a fixture list ready for [eloForecast()].
#'
#' @param results A data.table of results with columns `date`, `home`, `away`,
#'   `goalsHome` and `goalsAway`.
#' @param asOf Date at which to split; defaults to today.
#' @return A list with data.tables `played` and `remaining`.
#' @export
splitSeason <- function(results, asOf = Sys.Date()) {
  dt <- data.table::as.data.table(results)
  isPlayed <- !is.na(dt$goalsHome) & !is.na(dt$goalsAway) & dt$date <= asOf
  played <- dt[isPlayed]
  remaining <- dt[!isPlayed, .(date, home, away)]
  list(played = played, remaining = remaining)
}
