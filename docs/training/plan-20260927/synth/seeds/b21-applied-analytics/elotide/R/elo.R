#' Expected score under the Elo model
#'
#' Probability-like expected score of the home side given both ratings and a
#' home advantage expressed in rating points.
#'
#' @param ratingHome,ratingAway Numeric ratings of the two sides.
#' @param homeAdvantage Rating points added to the home side.
#' @param scale Rating difference corresponding to a tenfold change in odds.
#' @return A numeric vector of expected home scores in `[0, 1]`.
#' @export
expectedScore <- function(ratingHome, ratingAway, homeAdvantage = 65, scale = 400) {
  diff <- (ratingHome + homeAdvantage) - ratingAway
  1 / (1 + 10^(-diff / scale))
}

#' Convert goals to an Elo match score
#'
#' Maps a result to 1 (home win), 0.5 (draw) or 0 (away win).
#'
#' @param goalsHome,goalsAway Integer goals scored by each side.
#' @return A numeric vector of match scores from the home perspective.
#' @export
matchScore <- function(goalsHome, goalsAway) {
  ifelse(goalsHome > goalsAway, 1, ifelse(goalsHome == goalsAway, 0.5, 0))
}

#' Margin-of-victory multiplier for the K factor
#'
#' Scales the rating change by the goal difference so that heavy wins move
#' ratings more than narrow ones.
#'
#' @param goalsHome,goalsAway Integer goals scored by each side.
#' @param method How to map the goal difference to a multiplier.
#' @return A numeric vector of multipliers, at least one.
#' @export
marginMultiplier <- function(goalsHome, goalsAway, method = c("log", "linear", "none")) {
  method <- match.arg(method)
  margin <- abs(goalsHome - goalsAway)
  switch(method,
    log = log(margin + 1) + 1,
    linear = ifelse(margin <= 1, 1, ifelse(margin == 2, 1.5, (11 + margin) / 8)),
    none = rep(1, length(margin))
  )
}

#' Rating change for a single match
#'
#' Computes the points transferred from the away side to the home side.
#'
#' @param ratingHome,ratingAway Numeric ratings of the two sides.
#' @param goalsHome,goalsAway Integer goals scored by each side.
#' @param k Base K factor.
#' @param homeAdvantage Rating points added to the home side.
#' @param method How to map the goal difference to a multiplier.
#' @return A numeric vector of rating changes for the home side; the away
#'   side changes by the negative amount.
#' @export
eloDelta <- function(ratingHome, ratingAway, goalsHome, goalsAway, k = 20,
                     homeAdvantage = 65, method = "log") {
  expected <- expectedScore(ratingHome, ratingAway, homeAdvantage)
  actual <- matchScore(goalsHome, goalsAway)
  k * marginMultiplier(goalsHome, goalsAway, method) * (actual - expected)
}

#' Run Elo ratings over a season of results
#'
#' Processes matches in date order, updating both teams after each match, and
#' returns the match table with pre- and post-match ratings attached.
#'
#' @param matches A data.table of results with columns `date`, `home`, `away`,
#'   `goalsHome` and `goalsAway`.
#' @param initial Starting rating for teams seen for the first time, or a named
#'   numeric vector of starting ratings.
#' @param k Base K factor.
#' @param homeAdvantage Rating points added to the home side.
#' @param method How to map the goal difference to a multiplier.
#' @return A data.table like `matches` with added columns `eloHomePre`,
#'   `eloAwayPre`, `pHome` and `delta`, with attribute `ratings`.
#' @export
#' @import data.table
runElo <- function(matches, initial = 1500, k = 20, homeAdvantage = 65, method = "log") {
  required <- c("date", "home", "away", "goalsHome", "goalsAway")
  missingCols <- setdiff(required, names(matches))
  if (length(missingCols) > 0) {
    stop("`matches` is missing columns: ", paste(missingCols, collapse = ", "))
  }
  dt <- data.table::as.data.table(matches)
  data.table::setorder(dt, date)
  teams <- unique(c(dt$home, dt$away))
  ratings <- setNames(rep(1500, length(teams)), teams)
  if (length(initial) == 1 && is.null(names(initial))) {
    ratings[] <- initial
  } else {
    known <- intersect(names(initial), teams)
    ratings[known] <- initial[known]
  }
  n <- nrow(dt)
  homePre <- numeric(n)
  awayPre <- numeric(n)
  pHome <- numeric(n)
  delta <- numeric(n)
  for (i in seq_len(n)) {
    h <- dt$home[i]
    a <- dt$away[i]
    homePre[i] <- ratings[[h]]
    awayPre[i] <- ratings[[a]]
    pHome[i] <- expectedScore(homePre[i], awayPre[i], homeAdvantage)
    if (is.na(dt$goalsHome[i]) || is.na(dt$goalsAway[i])) {
      next
    }
    delta[i] <- eloDelta(homePre[i], awayPre[i], dt$goalsHome[i], dt$goalsAway[i],
      k = k, homeAdvantage = homeAdvantage, method = method
    )
    ratings[[h]] <- homePre[i] + delta[i]
    ratings[[a]] <- awayPre[i] - delta[i]
  }
  dt[, `:=`(eloHomePre = homePre, eloAwayPre = awayPre, pHome = pHome, delta = delta)]
  data.table::setattr(dt, "ratings", ratings)
  dt[]
}

#' Current Elo ratings as a ranked table
#'
#' Extracts the final ratings from [runElo()] output and ranks the teams.
#'
#' @param eloMatches A data.table returned by [runElo()].
#' @return A data.table with columns `rank`, `team` and `rating`.
#' @export
eloTable <- function(eloMatches) {
  ratings <- attr(eloMatches, "ratings")
  if (is.null(ratings)) {
    stop("`eloMatches` has no `ratings` attribute; was it made by runElo()?")
  }
  out <- data.table::data.table(team = names(ratings), rating = unname(ratings))
  data.table::setorder(out, -rating)
  out[, rank := seq_len(.N)]
  data.table::setcolorder(out, c("rank", "team", "rating"))
  out[]
}

#' Forecast upcoming fixtures from current ratings
#'
#' Attaches home-win, draw and away-win probabilities to a fixture list using
#' the Elo expected score and a fixed draw rate that shrinks with the rating
#' gap.
#'
#' @param fixtures A data.table with columns `home` and `away`.
#' @param ratings A named numeric vector of team ratings.
#' @param homeAdvantage Rating points added to the home side.
#' @param drawRate Draw probability between two evenly matched sides.
#' @return `fixtures` with added columns `pHome`, `pDraw` and `pAway`.
#' @export
eloForecast <- function(fixtures, ratings, homeAdvantage = 65, drawRate = 0.27) {
  dt <- data.table::copy(data.table::as.data.table(fixtures))
  unknown <- setdiff(unique(c(dt$home, dt$away)), names(ratings))
  if (length(unknown) > 0) {
    warning("No rating for: ", paste(unknown, collapse = ", "), "; using the mean.")
    ratings[unknown] <- mean(ratings)
  }
  expected <- expectedScore(ratings[dt$home], ratings[dt$away], homeAdvantage)
  draw <- drawRate * (1 - abs(2 * expected - 1))
  dt[, `:=`(
    pHome = unname(expected - draw / 2),
    pDraw = unname(draw),
    pAway = unname(1 - expected - draw / 2)
  )]
  dt[]
}
