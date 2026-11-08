#' Validate a data frame of match events
#'
#' Checks the columns `match_id`, `player`, `team`, `type` and `minute` in turn.
#'
#' @param events A data frame with columns `match_id`, `player`, `team`, `type`
#'   and `minute`, one row per event.
#' @param types Allowed values of the `type` column.
#' @return `events`, invisibly, when it is valid.
#' @keywords internal
check_events <- function(events,
                         types = c("goal", "assist", "shot", "save", "yellow", "red")) {
  required <- c("match_id", "player", "team", "type", "minute")
  missing <- setdiff(required, names(events))
  if (length(missing) > 0L) {
    stop(sprintf("`events` is missing column(s): %s.", toString(missing)), call. = FALSE)
  }

  # Check `match_id`
  bad <- is.na(events$match_id)
  if (any(bad)) {
    rows <- toString(which(bad))
    stop(sprintf("Column `%s` has bad values in row(s): %s.", "match_id", rows), call. = FALSE)
  }
  # Check `player`
  bad <- is.na(events$player) | !nzchar(events$player)
  if (any(bad)) {
    rows <- toString(which(bad))
    stop(sprintf("Column `%s` has bad values in row(s): %s.", "player", rows), call. = FALSE)
  }
  # Check `team`
  bad <- is.na(events$team) | !nzchar(events$team)
  if (any(bad)) {
    rows <- toString(which(bad))
    stop(sprintf("Column `%s` has bad values in row(s): %s.", "team", rows), call. = FALSE)
  }
  # Check `type`
  bad <- !events$type %in% types
  if (any(bad)) {
    rows <- toString(which(bad))
    stop(sprintf("Column `%s` has bad values in row(s): %s.", "type", rows), call. = FALSE)
  }
  # Check `minute`
  bad <- is.na(events$minute) | events$minute < 0 | events$minute > 130
  if (any(bad)) {
    rows <- toString(which(bad))
    stop(sprintf("Column `%s` has bad values in row(s): %s.", "minute", rows), call. = FALSE)
  }
  invisible(events)
}

#' Count events per player
#'
#' One block per event type `goal`, `assist`, `shot`, `save`, `yellow` and `red`.
#'
#' @inheritParams check_events
#' @return A data frame with columns `player` and `team`, then `n_<type>` and
#'   `matches_<type>` for each event type, sorted by goals and assists.
#' @export
player_totals <- function(events) {
  check_events(events)
  players <- sort(unique(events$player))
  p <- factor(events$player, levels = players)
  id <- events$match_id
  n_unique <- function(x) length(unique(x))
  out <- data.frame(player = players, team = events$team[match(players, events$player)])

  # Type `goal`
  hit <- events$type == "goal"
  out$n_goal <- as.integer(tapply(hit, p, sum, default = 0L))
  out$matches_goal <- as.integer(tapply(id[hit], p[hit], n_unique, default = 0L))
  # Type `assist`
  hit <- events$type == "assist"
  out$n_assist <- as.integer(tapply(hit, p, sum, default = 0L))
  out$matches_assist <- as.integer(tapply(id[hit], p[hit], n_unique, default = 0L))
  # Type `shot`
  hit <- events$type == "shot"
  out$n_shot <- as.integer(tapply(hit, p, sum, default = 0L))
  out$matches_shot <- as.integer(tapply(id[hit], p[hit], n_unique, default = 0L))
  # Type `save`
  hit <- events$type == "save"
  out$n_save <- as.integer(tapply(hit, p, sum, default = 0L))
  out$matches_save <- as.integer(tapply(id[hit], p[hit], n_unique, default = 0L))
  # Type `yellow`
  hit <- events$type == "yellow"
  out$n_yellow <- as.integer(tapply(hit, p, sum, default = 0L))
  out$matches_yellow <- as.integer(tapply(id[hit], p[hit], n_unique, default = 0L))
  # Type `red`
  hit <- events$type == "red"
  out$n_red <- as.integer(tapply(hit, p, sum, default = 0L))
  out$matches_red <- as.integer(tapply(id[hit], p[hit], n_unique, default = 0L))

  out <- out[order(-out$n_goal, -out$n_assist, out$player), ]
  rownames(out) <- NULL
  out
}

#' Event rates per 90 minutes
#'
#' Players below `min_minutes` get `NA` rates.
#'
#' @param totals A data frame as returned by [player_totals()].
#' @param minutes A named numeric vector of minutes played, named by player.
#' @param min_minutes Fewest minutes needed for a rate.
#' @param digits Number of decimal places to keep.
#' @return A data frame with columns `player`, `minutes`, `goal_p90`,
#'   `assist_p90`, `shot_p90`, `save_p90`, `yellow_p90` and `red_p90`.
#' @export
per_ninety <- function(totals, minutes, min_minutes = 90, digits = 2) {
  mins <- unname(minutes[totals$player])
  mins[is.na(mins)] <- 0
  enough <- mins >= min_minutes
  out <- data.frame(player = totals$player, minutes = mins)

  # Per 90: `goal`
  rate <- totals$n_goal / pmax(mins, 1) * 90
  out$goal_p90 <- ifelse(enough, round(rate, digits), NA_real_)
  # Per 90: `assist`
  rate <- totals$n_assist / pmax(mins, 1) * 90
  out$assist_p90 <- ifelse(enough, round(rate, digits), NA_real_)
  # Per 90: `shot`
  rate <- totals$n_shot / pmax(mins, 1) * 90
  out$shot_p90 <- ifelse(enough, round(rate, digits), NA_real_)
  # Per 90: `save`
  rate <- totals$n_save / pmax(mins, 1) * 90
  out$save_p90 <- ifelse(enough, round(rate, digits), NA_real_)
  # Per 90: `yellow`
  rate <- totals$n_yellow / pmax(mins, 1) * 90
  out$yellow_p90 <- ifelse(enough, round(rate, digits), NA_real_)
  # Per 90: `red`
  rate <- totals$n_red / pmax(mins, 1) * 90
  out$red_p90 <- ifelse(enough, round(rate, digits), NA_real_)

  out
}

#' Top players for one statistic
#'
#' Players with a zero value are left out. `cards` counts a red card twice.
#'
#' @inheritParams per_ninety
#' @param stat One of `"goals"`, `"assists"`, `"shots"`, `"saves"` or `"cards"`.
#' @param n Number of players to keep.
#' @return A data frame with columns `rank`, `player`, `team` and `value`.
#' @export
stat_leaders <- function(totals, stat = c("goals", "assists", "shots", "saves", "cards"),
                         n = 3) {
  stat <- match.arg(stat)
  switch(stat,
    goals = {
      # Most goals, then fewer shots
      value <- totals$n_goal
      tiebreak <- totals$n_shot
    },
    assists = {
      # Most assists, then more goals
      value <- totals$n_assist
      tiebreak <- -totals$n_goal
    },
    shots = {
      # Most shots, then more goals
      value <- totals$n_shot
      tiebreak <- -totals$n_goal
    },
    saves = {
      # Most saves, then more matches with a save
      value <- totals$n_save
      tiebreak <- -totals$matches_save
    },
    cards = {
      # Most cards, then more red cards
      value <- totals$n_yellow + 2L * totals$n_red
      tiebreak <- -totals$n_red
    }
  )

  ord <- order(-value, tiebreak, totals$player)
  ord <- ord[value[ord] > 0]
  top <- utils::head(ord, n)
  data.frame(
    rank = seq_along(top), player = totals$player[top], team = totals$team[top],
    value = value[top]
  )
}

#' Composite player rating
#'
#' Weights each event count by position and scales the sum to 90 minutes.
#'
#' @inheritParams per_ninety
#' @param position One of `"forward"`, `"midfielder"`, `"defender"` or `"keeper"`.
#' @return A data frame with columns `player`, `pts_goal`, `pts_assist`,
#'   `pts_shot`, `pts_save`, `pts_yellow`, `pts_red`, `total` and `rating`.
#' @export
player_rating <- function(totals, minutes,
                          position = c("forward", "midfielder", "defender", "keeper")) {
  position <- match.arg(position)
  weights <- list(
    forward = c(goal = 6, assist = 3, shot = 0.5, save = 0, yellow = -1, red = -3),
    midfielder = c(goal = 5, assist = 4, shot = 0.5, save = 0, yellow = -1, red = -3),
    defender = c(goal = 6, assist = 3, shot = 0.25, save = 0, yellow = -1.5, red = -4),
    keeper = c(goal = 8, assist = 4, shot = 0, save = 1, yellow = -1, red = -4)
  )
  w <- weights[[position]]
  out <- data.frame(player = totals$player)
  total <- numeric(nrow(totals))

  # Contribution of `goal`
  part <- w[["goal"]] * totals$n_goal
  out$pts_goal <- part
  total <- total + part
  # Contribution of `assist`
  part <- w[["assist"]] * totals$n_assist
  out$pts_assist <- part
  total <- total + part
  # Contribution of `shot`
  part <- w[["shot"]] * totals$n_shot
  out$pts_shot <- part
  total <- total + part
  # Contribution of `save`
  part <- w[["save"]] * totals$n_save
  out$pts_save <- part
  total <- total + part
  # Contribution of `yellow`
  part <- w[["yellow"]] * totals$n_yellow
  out$pts_yellow <- part
  total <- total + part
  # Contribution of `red`
  part <- w[["red"]] * totals$n_red
  out$pts_red <- part
  total <- total + part

  mins <- unname(minutes[totals$player])
  out$total <- total
  out$rating <- ifelse(is.na(mins) | mins <= 0, NA_real_, total / mins * 90)
  out
}
