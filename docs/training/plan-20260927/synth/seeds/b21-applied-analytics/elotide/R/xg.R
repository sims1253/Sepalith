#' Distance from a shot location to the goal centre
#'
#' Pitch coordinates are in metres with the attacking goal centred at
#' `(pitchLength, pitchWidth / 2)`.
#'
#' @param x,y Numeric shot coordinates.
#' @param pitchLength,pitchWidth Pitch dimensions in metres.
#' @return A numeric vector of distances in metres.
#' @family shot geometry
#' @export
shotDistance <- function(x, y, pitchLength = 105, pitchWidth = 68) {
  dx <- pitchLength - x
  dy <- pitchWidth / 2 - y
  sqrt(dx^2 + dy^2)
}

#' Angle subtended by the goal mouth from a shot location
#'
#' The wider the visible goal, the larger the angle and the better the chance.
#'
#' @param x,y Numeric shot coordinates.
#' @param pitchLength,pitchWidth Pitch dimensions in metres.
#' @param goalWidth Width of the goal mouth in metres.
#' @return A numeric vector of angles in radians.
#' @family shot geometry
#' @export
shotAngle <- function(x, y, pitchLength = 105, pitchWidth = 68, goalWidth = 7.32) {
  dx <- pitchLength - x
  dy <- pitchWidth / 2 - y
  half <- goalWidth / 2
  angle <- atan2(half - dy, dx) - atan2(-half - dy, dx)
  abs(angle)
}

#' Simple logistic expected-goals model
#'
#' Scores each shot from its distance, angle and body part using fixed
#' coefficients. Penalties receive a flat value.
#'
#' @param shots A data.table of shots with columns `x`, `y`, `bodyPart` and
#'   `situation`.
#' @param coefs Named numeric vector with `intercept`, `distance`, `angle` and
#'   `header` coefficients.
#' @param penaltyXg Value assigned to penalty kicks.
#' @return A numeric vector of shot xG values.
#' @export
shotXg <- function(shots, coefs = c(intercept = -1.1, distance = -0.11, angle = 1.6,
                                    header = -0.9), penaltyXg = 0.76) {
  dist <- shotDistance(shots$x, shots$y)
  ang <- shotAngle(shots$x, shots$y)
  header <- as.numeric(shots$bodyPart == "head")
  eta <- coefs[["intercept"]] + coefs[["distance"]] * dist +
    coefs[["angle"]] * ang + coefs[["header"]] * header
  xg <- stats::plogis(eta)
  xg[shots$situation == "penalty"] <- penaltyXg
  xg
}

#' Expected goals per team per match
#'
#' Sums shot xG by match and team and attaches actual goals for comparison.
#'
#' @param shots A data.table of shots with columns `matchId`, `team`, `xg` and
#'   `isGoal`.
#' @return A data.table with columns `matchId`, `team`, `shots`, `xg` and
#'   `goals`.
#' @export
xgByMatch <- function(shots) {
  dt <- data.table::as.data.table(shots)
  if (!"xg" %in% names(dt)) {
    dt[, xg := shotXg(dt)]
  }
  dt[, .(shots = .N, xg = sum(xg), goals = sum(isGoal)), by = .(matchId, team)][]
}

#' Season expected goals for each team
#'
#' Aggregates match-level xG for the team in possession of the shot.
#'
#' @param matchXg A data.table returned by [xgByMatch()].
#' @return A data.table with columns `team`, `matches`, `xgFor`, `goalsFor`
#'   and `xgForPerMatch`.
#' @family season xG
#' @export
xgForByTeam <- function(matchXg) {
  out <- matchXg[, .(
    matches = data.table::uniqueN(matchId),
    xgFor = sum(xg),
    goalsFor = sum(goals)
  ), by = team]
  out[, xgForPerMatch := xgFor / matches]
  data.table::setorder(out, -xgFor)
  out[]
}

#' Season expected goals conceded for each team
#'
#' Aggregates match-level xG of the opponent in each match.
#'
#' @param matchXg A data.table returned by [xgByMatch()].
#' @return A data.table with columns `team`, `matches`, `xgAgainst`,
#'   `goalsAgainst` and `xgAgainstPerMatch`.
#' @family season xG
#' @export
xgAgainstByTeam <- function(matchXg) {
  opp <- matchXg[matchXg, on = "matchId", allow.cartesian = TRUE][team != i.team]
  out <- opp[, .(
    matches = data.table::uniqueN(matchId),
    xgAgainst = sum(i.xg),
    goalsAgainst = sum(i.goals)
  ), by = team]
  out[, xgAgainstPerMatch := xgAgainst / matches]
  data.table::setorder(out, xgAgainst)
  out[]
}

#' Simulate match outcomes from shot xG
#'
#' Treats every shot as an independent Bernoulli trial and simulates the
#' scoreline many times to obtain win, draw and loss probabilities.
#'
#' @param xgHome,xgAway Numeric vectors of shot xG for each side.
#' @param nSims Number of simulated replays.
#' @param seed Optional random seed for reproducibility.
#' @return A named numeric vector with `home`, `draw` and `away`
#'   probabilities and the expected points of each side.
#' @export
simulateFromXg <- function(xgHome, xgAway, nSims = 10000L, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  homeWins <- 0L
  draws <- 0L
  awayWins <- 0L
  for (s in seq_len(nSims)) {
    gh <- sum(stats::runif(length(xgHome)) < xgHome)
    ga <- sum(stats::runif(length(xgAway)) < xgAway)
    if (gh > ga) {
      homeWins <- homeWins + 1L
    } else if (gh == ga) {
      draws <- draws + 1L
    } else {
      awayWins <- awayWins + 1L
    }
  }
  pHome <- homeWins / nSims
  pDraw <- draws / nSims
  pAway <- awayWins / nSims
  c(
    home = pHome,
    draw = pDraw,
    away = pAway,
    xPtsHome = 3 * pHome + pDraw,
    xPtsAway = 3 * pAway + pDraw
  )
}
