#' Validate rating settings
#'
#' Checks `init`, `home_adv`, `scale` and `k_base` in turn.
#'
#' @param init Starting rating for every team.
#' @param home_adv Rating points added to the home side.
#' @param scale Rating difference that multiplies the odds by ten.
#' @param k_base Named K factors, one per competition type.
#' @return `TRUE`, invisibly, when all settings are valid.
#' @keywords internal
check_rating_args <- function(init, home_adv, scale, k_base) {
  # Check `init`
  if (!is.numeric(init) || length(init) != 1L || is.na(init)) {
    stop(sprintf("`%s` must be a single number.", "init"), call. = FALSE)
  }
  # Check `home_adv`
  if (!is.numeric(home_adv) || length(home_adv) != 1L || is.na(home_adv)) {
    stop(sprintf("`%s` must be a single number.", "home_adv"), call. = FALSE)
  }
  if (home_adv < 0) {
    stop(sprintf("`%s` must not be negative.", "home_adv"), call. = FALSE)
  }
  # Check `scale`
  if (!is.numeric(scale) || length(scale) != 1L || is.na(scale)) {
    stop(sprintf("`%s` must be a single number.", "scale"), call. = FALSE)
  }
  if (scale <= 0) {
    stop(sprintf("`%s` must be positive.", "scale"), call. = FALSE)
  }
  # Check `k_base`
  if (!is.numeric(k_base) || is.null(names(k_base)) || anyNA(k_base)) {
    stop(sprintf("`%s` must be a named numeric vector.", "k_base"), call. = FALSE)
  }
  if (any(k_base <= 0)) {
    stop(sprintf("`%s` must be positive.", "k_base"), call. = FALSE)
  }
  invisible(TRUE)
}

#' Update two ratings after one match
#'
#' The competition type sets the K factor, the home advantage and whether a
#' winning margin of two or more goals scales the update.
#'
#' @param rating_home,rating_away Ratings before the match.
#' @param goals_home,goals_away Goals scored by each side.
#' @param competition One of `"league"`, `"cup"`, `"friendly"`, `"playoff"`
#'   or `"tournament"`.
#' @inheritParams check_rating_args
#' @return A list with elements `home`, `away`, `delta` and `expected`.
#' @export
elo_update <- function(rating_home, rating_away, goals_home, goals_away,
                       competition = c("league", "cup", "friendly", "playoff", "tournament"),
                       k_base = c(league = 20, cup = 30, friendly = 10, playoff = 40,
                                  tournament = 25),
                       home_adv = 60, scale = 400) {
  competition <- match.arg(competition)
  margin <- abs(goals_home - goals_away)
  bonus <- ifelse(margin > 1, log2(margin) + 1, 1)
  switch(competition,
    league = {
      # Full home advantage, no margin bonus
      k <- k_base[["league"]]
      adv <- home_adv
      mult <- 1
    },
    cup = {
      # Full home advantage, margin bonus
      k <- k_base[["cup"]]
      adv <- home_adv
      mult <- bonus
    },
    friendly = {
      # Half home advantage, no margin bonus
      k <- k_base[["friendly"]]
      adv <- home_adv / 2
      mult <- 1
    },
    playoff = {
      # Full home advantage, margin bonus
      k <- k_base[["playoff"]]
      adv <- home_adv
      mult <- bonus
    },
    tournament = {
      # Neutral venue, margin bonus
      k <- k_base[["tournament"]]
      adv <- 0
      mult <- bonus
    }
  )

  expected <- 1 / (1 + 10^((rating_away - rating_home - adv) / scale))
  actual <- ifelse(goals_home > goals_away, 1, ifelse(goals_home == goals_away, 0.5, 0))
  delta <- k * mult * (actual - expected)
  list(home = rating_home + delta, away = rating_away - delta, delta = delta, expected = expected)
}

#' Expected score from a rating difference
#'
#' @param diff Rating of the first side minus rating of the second side.
#' @param model One of `"logistic"`, `"normal"`, `"linear"` or `"bradley_terry"`.
#' @inheritParams check_rating_args
#' @return A numeric vector of expected scores between 0 and 1.
#' @export
expected_score <- function(diff, model = c("logistic", "normal", "linear", "bradley_terry"),
                           scale = 400) {
  model <- match.arg(model)
  if (!is.numeric(diff)) {
    stop(sprintf("`%s` must be numeric.", "diff"), call. = FALSE)
  }
  if (!is.numeric(scale) || length(scale) != 1L || scale <= 0) {
    stop(sprintf("`%s` must be a single positive number.", "scale"), call. = FALSE)
  }
  spread <- c(logistic = 1, normal = 0.5, linear = 2, bradley_terry = 1)
  z <- diff / scale

  switch(model,
    logistic = {
      p <- 1 / (1 + 10^(-z / spread[["logistic"]]))
    },
    normal = {
      p <- stats::pnorm(z / spread[["normal"]])
    },
    linear = {
      p <- 0.5 + z / spread[["linear"]]
      p <- pmin(pmax(p, 0), 1)
    },
    bradley_terry = {
      strength <- 10^(z / spread[["bradley_terry"]])
      p <- strength / (strength + 1)
    }
  )
  p
}

#' Run ratings through a season
#'
#' Applies [elo_update()] to every match in order. The `competition` column is
#' optional and defaults to `"league"`.
#'
#' @param results A data frame with columns `home`, `away`, `home_goals`,
#'   `away_goals` and optionally `competition`, in match order.
#' @inheritParams check_rating_args
#' @return A list with elements `ratings` (named, best first) and `history`
#'   (one row per match).
#' @export
run_ratings <- function(results, init = 1500, home_adv = 60, scale = 400) {
  check_rating_args(init, home_adv, scale, k_base = c(league = 20))
  teams <- sort(unique(c(results$home, results$away)))
  ratings <- stats::setNames(rep(init, length(teams)), teams)
  competition <- results$competition
  if (is.null(competition)) {
    competition <- rep("league", nrow(results))
  }
  n <- nrow(results)
  home_after <- numeric(n)
  away_after <- numeric(n)
  delta <- numeric(n)

  for (i in seq_len(n)) {
    h <- results$home[i]
    a <- results$away[i]
    upd <- elo_update(ratings[[h]], ratings[[a]], results$home_goals[i], results$away_goals[i],
      competition = competition[i], home_adv = home_adv, scale = scale)
    ratings[[h]] <- upd$home
    ratings[[a]] <- upd$away
    home_after[i] <- upd$home
    away_after[i] <- upd$away
    delta[i] <- upd$delta
  }

  history <- data.frame(
    home = results$home, away = results$away, competition = competition,
    home_after = home_after, away_after = away_after, delta = delta
  )
  list(ratings = sort(ratings, decreasing = TRUE), history = history)
}

#' Pull idle ratings back towards the mean
#'
#' @param ratings A named numeric vector of ratings.
#' @param idle_days Days since each team last played, recycled from length one.
#' @param method One of `"none"`, `"linear"`, `"exponential"`, `"step"` or `"floor"`.
#' @param mean_rating The rating that idle teams drift towards.
#' @param rate Share of the gap to `mean_rating` removed per idle day.
#' @param step_days Length of one idle block for `"step"` and `"floor"`.
#' @return A named numeric vector of decayed ratings.
#' @export
rating_decay <- function(ratings, idle_days,
                         method = c("none", "linear", "exponential", "step", "floor"),
                         mean_rating = 1500, rate = 0.01, step_days = 30) {
  method <- match.arg(method)
  if (length(idle_days) == 1L) {
    idle_days <- rep(idle_days, length(ratings))
  }
  if (length(idle_days) != length(ratings)) {
    stop("`idle_days` must have length 1 or the length of `ratings`.", call. = FALSE)
  }
  gap <- ratings - mean_rating
  switch(method,
    none = {
      # Keep ratings as they are
      shrink <- rep(0, length(ratings))
    },
    linear = {
      # Shrink by `rate` per idle day
      shrink <- rate * idle_days
    },
    exponential = {
      # Shrink by `rate` per idle day, compounded
      shrink <- 1 - (1 - rate)^idle_days
    },
    step = {
      # Shrink by `rate` per day, counted in full blocks of `step_days`
      shrink <- rate * step_days * (idle_days %/% step_days)
    },
    floor = {
      # Shrink like `step`, but only above `mean_rating`
      shrink <- rate * step_days * (idle_days %/% step_days)
      shrink[gap < 0] <- 0
    }
  )
  shrink <- pmin(pmax(shrink, 0), 1)
  out <- ratings - shrink * gap
  names(out) <- names(ratings)
  out
}

#' Peak, trough and latest rating per team
#'
#' @param history The `history` element returned by [run_ratings()].
#' @return A data frame with columns `team`, `peak`, `peak_game`, `trough`,
#'   `trough_game`, `latest` and `latest_game`.
#' @export
rating_extremes <- function(history) {
  n <- nrow(history)
  team <- c(history$home, history$away)
  rating <- c(history$home_after, history$away_after)
  game <- c(seq_len(n), seq_len(n))
  teams <- sort(unique(team))
  out <- data.frame(team = teams)

  # Peak rating
  best <- c(tapply(rating, team, max)[teams])
  at <- vapply(teams, function(t) min(game[team == t & rating == best[[t]]]), numeric(1))
  out$peak <- unname(best)
  out$peak_game <- unname(at)
  # Trough rating
  best <- c(tapply(rating, team, min)[teams])
  at <- vapply(teams, function(t) min(game[team == t & rating == best[[t]]]), numeric(1))
  out$trough <- unname(best)
  out$trough_game <- unname(at)
  # Latest rating
  at <- c(tapply(game, team, max)[teams])
  best <- vapply(teams, function(t) rating[team == t & game == at[[t]]], numeric(1))
  out$latest <- unname(best)
  out$latest_game <- unname(at)

  out
}
