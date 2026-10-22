#' Stack results into one row per team per match
#'
#' Each match contributes a home row and an away row, so that per-team
#' aggregates become simple grouped sums.
#'
#' @param matches A data.table of results with columns `date`, `home`, `away`,
#'   `goalsHome` and `goalsAway`.
#' @return A data.table with columns `date`, `team`, `opponent`, `venue`,
#'   `goalsFor`, `goalsAgainst` and `points`.
#' @keywords internal
resultsLong = function(matches) {
  dt = data.table::as.data.table(matches)
  dt = dt[!is.na(goalsHome) & !is.na(goalsAway)]
  homeRows = dt[, .(date, team = home, opponent = away, venue = "home",
    goalsFor = goalsHome, goalsAgainst = goalsAway)]
  awayRows = dt[, .(date, team = away, opponent = home, venue = "away",
    goalsFor = goalsAway, goalsAgainst = goalsHome)]
  long = data.table::rbindlist(list(homeRows, awayRows))
  long[, points := data.table::fifelse(goalsFor > goalsAgainst, 3L,
    data.table::fifelse(goalsFor == goalsAgainst, 1L, 0L))]
  data.table::setorder(long, team, date)
  long[]
}

#' Compute a league table
#'
#' Aggregates played, won, drawn, lost, goals and points per team, then orders
#' the table using the chosen tiebreak rule.
#'
#' @param matches A data.table of results with columns `date`, `home`, `away`,
#'   `goalsHome` and `goalsAway`.
#' @param tiebreak Rule used to separate teams level on points.
#' @param pointsWin,pointsDraw Points awarded for a win and a draw.
#' @return A data.table with one row per team and a `position` column.
#' @export
leagueTable = function(matches, tiebreak = c("goalDifference", "headToHead", "goalsFor"),
                       pointsWin = 3L, pointsDraw = 1L) {
  tiebreak = match.arg(tiebreak)
  long = resultsLong(matches)
  long[, points := data.table::fifelse(goalsFor > goalsAgainst, pointsWin,
    data.table::fifelse(goalsFor == goalsAgainst, pointsDraw, 0L))]
  tab = long[, .(
    played = .N,
    won = sum(goalsFor > goalsAgainst),
    drawn = sum(goalsFor == goalsAgainst),
    lost = sum(goalsFor < goalsAgainst),
    goalsFor = sum(goalsFor),
    goalsAgainst = sum(goalsAgainst),
    points = sum(points)
  ), by = team]
  tab[, goalDifference := goalsFor - goalsAgainst]
  tab = applyTiebreak(tab, long, tiebreak)
  tab[, position := seq_len(.N)]
  data.table::setcolorder(tab, c("position", "team"))
  tab[]
}

#' Order a league table using a tiebreak rule
#'
#' Sorts by points first and then by the requested secondary criterion.
#'
#' @param tab A data.table of team aggregates as built by [leagueTable()].
#' @param long A data.table of per-team match rows from `resultsLong()`.
#' @param tiebreak Rule used to separate teams level on points.
#' @return `tab`, reordered.
#' @keywords internal
applyTiebreak = function(tab, long, tiebreak) {
  switch(tiebreak,
    goalDifference = data.table::setorder(tab, -points, -goalDifference, -goalsFor, team),
    goalsFor = data.table::setorder(tab, -points, -goalsFor, -goalDifference, team),
    headToHead = {
      tab[, h2h := 0]
      for (pts in unique(tab$points)) {
        level = tab[points == pts, team]
        if (length(level) < 2) {
          next
        }
        mini = headToHead(long, level)
        tab[mini, h2h := i.points, on = "team"]
      }
      data.table::setorder(tab, -points, -h2h, -goalDifference, team)
      tab[, h2h := NULL]
    }
  )
  tab
}

#' Head-to-head mini table among a set of teams
#'
#' Restricts the results to matches played between the given teams and sums
#' points and goal difference.
#'
#' @param long A data.table of per-team match rows from `resultsLong()`.
#' @param teams Character vector of team names.
#' @return A data.table with columns `team`, `points` and `goalDifference`.
#' @export
headToHead = function(long, teams) {
  sub = long[team %in% teams & opponent %in% teams]
  if (nrow(sub) == 0) {
    return(data.table::data.table(team = teams, points = 0L, goalDifference = 0L))
  }
  out = sub[, .(points = sum(points), goalDifference = sum(goalsFor - goalsAgainst)),
    by = team]
  missingTeams = setdiff(teams, out$team)
  if (length(missingTeams) > 0) {
    out = data.table::rbindlist(list(
      out,
      data.table::data.table(team = missingTeams, points = 0L, goalDifference = 0L)
    ))
  }
  out[]
}

#' Recent form string per team
#'
#' Builds a string such as `"WWDLW"` from each team's last `n` results, most
#' recent last.
#'
#' @param matches A data.table of results with columns `date`, `home`, `away`,
#'   `goalsHome` and `goalsAway`.
#' @param n Number of recent matches to include.
#' @return A data.table with columns `team`, `form` and `formPoints`.
#' @export
formGuide = function(matches, n = 5L) {
  long = resultsLong(matches)
  long[, letter := data.table::fcase(
    goalsFor > goalsAgainst, "W",
    goalsFor == goalsAgainst, "D",
    default = "L"
  )]
  recent = long[, utils::tail(.SD, n), by = team, .SDcols = c("letter", "points")]
  recent[, .(form = paste(letter, collapse = ""), formPoints = sum(points)), by = team][]
}

#' League position after every matchday
#'
#' Recomputes the table after each distinct match date so that the trajectory
#' of each team can be plotted.
#'
#' @param matches A data.table of results with columns `date`, `home`, `away`,
#'   `goalsHome` and `goalsAway`.
#' @param tiebreak Rule used to separate teams level on points.
#' @return A data.table with columns `date`, `team`, `position` and `points`.
#' @export
positionHistory = function(matches, tiebreak = "goalDifference") {
  dt = data.table::as.data.table(matches)
  dates = sort(unique(dt$date))
  snapshots = vector("list", length(dates))
  for (i in seq_along(dates)) {
    played = dt[date <= dates[i]]
    tab = leagueTable(played, tiebreak = tiebreak)
    snapshots[[i]] = tab[, .(date = dates[i], team, position, points)]
  }
  data.table::rbindlist(snapshots)
}
