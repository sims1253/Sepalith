#' Validate a data frame of match results
#'
#' Checks the columns `home`, `away`, `home_goals` and `away_goals` in turn.
#'
#' @param results A data frame with columns `home`, `away`, `home_goals` and
#'   `away_goals`, one row per match.
#' @return `results`, invisibly, when it is valid.
#' @keywords internal
check_results <- function(results) {
  if (!is.data.frame(results)) {
    stop("`results` must be a data frame.", call. = FALSE)
  }
  required <- c("home", "away", "home_goals", "away_goals")
  missing <- setdiff(required, names(results))
  if (length(missing) > 0L) {
    stop(sprintf("`results` is missing column(s): %s.", paste(missing, collapse = ", ")),
      call. = FALSE)
  }

  # Check `home`
  x <- results$home
  if (!is.character(x) || anyNA(x)) {
    stop(sprintf("Column `%s` must be a character vector without NA.", "home"), call. = FALSE)
  }
  # Check `away`
  x <- results$away
  if (!is.character(x) || anyNA(x)) {
    stop(sprintf("Column `%s` must be a character vector without NA.", "away"), call. = FALSE)
  }
  # Check `home_goals`
  x <- results$home_goals
  if (!is.numeric(x) || anyNA(x) || any(x < 0 | x %% 1 != 0)) {
    stop(sprintf("Column `%s` must hold non-negative whole numbers.", "home_goals"),
      call. = FALSE)
  }
  # Check `away_goals`
  x <- results$away_goals
  if (!is.numeric(x) || anyNA(x) || any(x < 0 | x %% 1 != 0)) {
    stop(sprintf("Column `%s` must hold non-negative whole numbers.", "away_goals"),
      call. = FALSE)
  }

  if (any(results$home == results$away)) {
    stop("A team cannot play itself.", call. = FALSE)
  }
  invisible(results)
}

#' Build a league table from match results
#'
#' Counts one block per statistic column, then adds goal difference and points.
#'
#' @inheritParams check_results
#' @param points Named points awarded for a `win`, a `draw` and a `loss`.
#' @return A data frame with columns `team`, `played`, `won`, `drawn`, `lost`,
#'   `goals_for`, `goals_against`, `goal_diff` and `points`, best team first.
#' @export
standings_table <- function(results, points = c(win = 3, draw = 1, loss = 0)) {
  check_results(results)
  teams <- sort(unique(c(results$home, results$away)))
  home_f <- factor(results$home, levels = teams)
  away_f <- factor(results$away, levels = teams)
  margin <- results$home_goals - results$away_goals
  ones <- rep(1L, nrow(results))

  # Column `played`
  home_part <- tapply(ones, home_f, sum, default = 0L)
  away_part <- tapply(ones, away_f, sum, default = 0L)
  played <- as.integer(home_part + away_part)
  # Column `won`
  home_part <- tapply(margin > 0, home_f, sum, default = 0L)
  away_part <- tapply(margin < 0, away_f, sum, default = 0L)
  won <- as.integer(home_part + away_part)
  # Column `drawn`
  home_part <- tapply(margin == 0, home_f, sum, default = 0L)
  away_part <- tapply(margin == 0, away_f, sum, default = 0L)
  drawn <- as.integer(home_part + away_part)
  # Column `lost`
  home_part <- tapply(margin < 0, home_f, sum, default = 0L)
  away_part <- tapply(margin > 0, away_f, sum, default = 0L)
  lost <- as.integer(home_part + away_part)
  # Column `goals_for`
  home_part <- tapply(results$home_goals, home_f, sum, default = 0L)
  away_part <- tapply(results$away_goals, away_f, sum, default = 0L)
  goals_for <- as.integer(home_part + away_part)
  # Column `goals_against`
  home_part <- tapply(results$away_goals, home_f, sum, default = 0L)
  away_part <- tapply(results$home_goals, away_f, sum, default = 0L)
  goals_against <- as.integer(home_part + away_part)

  goal_diff <- goals_for - goals_against
  total <- won * points[["win"]] + drawn * points[["draw"]] + lost * points[["loss"]]
  out <- data.frame(
    team = teams, played = played, won = won, drawn = drawn, lost = lost,
    goals_for = goals_for, goals_against = goals_against, goal_diff = goal_diff,
    points = total
  )
  out <- out[order(-out$points, -out$goal_diff, -out$goals_for, out$team), ]
  rownames(out) <- NULL
  out
}

#' Rates per game for each statistic column
#'
#' Divides `won`, `drawn`, `lost`, `goals_for`, `goals_against` and `points`
#' by games played; teams without a game get `NA`.
#'
#' @param table A league table as returned by [standings_table()].
#' @param digits Number of decimal places to keep.
#' @return A data frame with columns `team`, `played`, `won_pg`, `drawn_pg`,
#'   `lost_pg`, `goals_for_pg`, `goals_against_pg` and `points_pg`.
#' @export
standings_per_game <- function(table, digits = 2) {
  played <- pmax(table$played, 1L)
  idle <- table$played == 0L
  out <- data.frame(team = table$team, played = table$played)

  # Per game: `won`
  rate <- table$won / played
  rate[idle] <- NA_real_
  out$won_pg <- round(rate, digits)
  # Per game: `drawn`
  rate <- table$drawn / played
  rate[idle] <- NA_real_
  out$drawn_pg <- round(rate, digits)
  # Per game: `lost`
  rate <- table$lost / played
  rate[idle] <- NA_real_
  out$lost_pg <- round(rate, digits)
  # Per game: `goals_for`
  rate <- table$goals_for / played
  rate[idle] <- NA_real_
  out$goals_for_pg <- round(rate, digits)
  # Per game: `goals_against`
  rate <- table$goals_against / played
  rate[idle] <- NA_real_
  out$goals_against_pg <- round(rate, digits)
  # Per game: `points`
  rate <- table$points / played
  rate[idle] <- NA_real_
  out$points_pg <- round(rate, digits)

  out
}

#' Leaders for each statistic column
#'
#' Most `points`, `won` and `goals_for`; fewest `lost` and `goals_against`.
#' Ties are joined with a slash.
#'
#' @inheritParams standings_per_game
#' @return A data frame with columns `stat`, `team` and `value`, one row per
#'   statistic.
#' @export
column_leaders <- function(table) {
  stats <- c("points", "won", "goals_for", "lost", "goals_against")
  team <- character(length(stats))
  value <- numeric(length(stats))
  names(team) <- names(value) <- stats

  # Most `points`
  best <- max(table$points)
  team[["points"]] <- paste(table$team[table$points == best], collapse = "/")
  value[["points"]] <- best
  # Most `won`
  best <- max(table$won)
  team[["won"]] <- paste(table$team[table$won == best], collapse = "/")
  value[["won"]] <- best
  # Most `goals_for`
  best <- max(table$goals_for)
  team[["goals_for"]] <- paste(table$team[table$goals_for == best], collapse = "/")
  value[["goals_for"]] <- best
  # Fewest `lost`
  best <- min(table$lost)
  team[["lost"]] <- paste(table$team[table$lost == best], collapse = "/")
  value[["lost"]] <- best
  # Fewest `goals_against`
  best <- min(table$goals_against)
  team[["goals_against"]] <- paste(table$team[table$goals_against == best], collapse = "/")
  value[["goals_against"]] <- best

  data.frame(stat = stats, team = unname(team), value = unname(value))
}

#' Compare two league tables
#'
#' Reports, per team of `after`, the earlier value and the change of
#' `position`, `points`, `goal_diff`, `won` and `lost`.
#'
#' @param before,after League tables as returned by [standings_table()].
#' @return A data frame with column `team` and a `_before` and `_change`
#'   column for each compared statistic.
#' @export
compare_standings <- function(before, after) {
  idx <- match(after$team, before$team)
  if (anyNA(idx)) {
    stop(sprintf("Team(s) not in `before`: %s.", paste(after$team[is.na(idx)], collapse = ", ")),
      call. = FALSE)
  }
  out <- data.frame(team = after$team)

  # Compare `position`
  old <- idx
  new <- seq_along(after$team)
  out$position_before <- old
  out$position_change <- old - new
  # Compare `points`
  old <- before$points[idx]
  new <- after$points
  out$points_before <- old
  out$points_change <- new - old
  # Compare `goal_diff`
  old <- before$goal_diff[idx]
  new <- after$goal_diff
  out$goal_diff_before <- old
  out$goal_diff_change <- new - old
  # Compare `won`
  old <- before$won[idx]
  new <- after$won
  out$won_before <- old
  out$won_change <- new - old
  # Compare `lost`
  old <- before$lost[idx]
  new <- after$lost
  out$lost_before <- old
  out$lost_change <- new - old

  out
}

#' Format a league table as text lines
#'
#' Shows `played`, `won`, `drawn`, `lost`, `goal_diff` and `points` per team.
#'
#' @inheritParams standings_per_game
#' @param style One of `"plain"`, `"compact"`, `"markdown"` or `"csv"`.
#' @return A character vector with a header line followed by one line per team.
#' @export
format_standings <- function(table, style = c("plain", "compact", "markdown", "csv")) {
  style <- match.arg(style)
  cols <- c("played", "won", "drawn", "lost", "goal_diff", "points")
  abbrev <- c(played = "P", won = "W", drawn = "D", lost = "L", goal_diff = "GD", points = "Pts")
  values <- as.matrix(table[cols])
  pos <- seq_len(nrow(table))
  switch(style,
    plain = {
      cells <- apply(values, 1, function(v) paste(sprintf("%5d", v), collapse = ""))
      heads <- paste(sprintf("%5s", abbrev), collapse = "")
      header <- sprintf("%3s  %-16s%s", "Pos", "Team", heads)
      rows <- sprintf("%3d  %-16s%s", pos, table$team, cells)
      c(header, rows)
    },
    compact = {
      cells <- apply(values, 1, function(v) paste(v, collapse = " "))
      heads <- paste(abbrev, collapse = " ")
      header <- sprintf("%s %s %s", "Pos", "Team", heads)
      rows <- sprintf("%d %s %s", pos, table$team, cells)
      c(header, rows)
    },
    markdown = {
      cells <- apply(values, 1, function(v) paste(v, collapse = " | "))
      heads <- paste(abbrev, collapse = " | ")
      header <- sprintf("| %s | %s | %s |", "Pos", "Team", heads)
      rule <- paste0("|", strrep("---|", length(cols) + 2L))
      rows <- sprintf("| %d | %s | %s |", pos, table$team, cells)
      c(header, rule, rows)
    },
    csv = {
      cells <- apply(values, 1, function(v) paste(v, collapse = ","))
      heads <- paste(cols, collapse = ",")
      header <- sprintf("%s,%s,%s", "pos", "team", heads)
      rows <- sprintf("%d,%s,%s", pos, table$team, cells)
      c(header, rows)
    }
  )
}
