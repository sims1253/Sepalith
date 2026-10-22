#' Simple random walk on the integers
#'
#' Each step moves up with probability `p.up` and down otherwise.
#'
#' @param n.steps Number of steps to take.
#' @param p.up Probability of a step up.
#' @param start Starting position.
#' @return An integer vector of length `n.steps + 1` holding the path,
#'   starting at `start`.
#' @export
walk.1d = function(n.steps, p.up = 0.5, start = 0L) {
  check.count(n.steps, "n.steps")
  check.probability(p.up, "p.up")
  steps = ifelse(runif(n.steps) < p.up, 1L, -1L)
  as.integer(c(start, start + cumsum(steps)))
}

#' Simple random walk on the square lattice
#'
#' Each step moves one unit north, south, east or west with equal chance.
#'
#' @param n.steps Number of steps to take.
#' @param start Starting position as a length-two vector `c(x, y)`.
#' @return An integer matrix with columns `x` and `y` and `n.steps + 1` rows.
#' @export
walk.2d = function(n.steps, start = c(0L, 0L)) {
  check.count(n.steps, "n.steps")
  moves = matrix(c(1L, 0L, -1L, 0L, 0L, 1L, 0L, -1L), ncol = 2, byrow = TRUE)
  pick = sample.int(4, n.steps, replace = TRUE)
  path = rbind(as.integer(start), moves[pick, , drop = FALSE])
  path = apply(path, 2, cumsum)
  colnames(path) = c("x", "y")
  path
}

#' Steps until a walk first reaches a level
#'
#' Walks one step at a time until `level` is hit or `max.steps` is used up.
#'
#' @param level Level to reach; must differ from `start`.
#' @param p.up Probability of a step up.
#' @param start Starting position.
#' @param max.steps Upper bound on the number of steps.
#' @return The number of steps taken, or `NA` if the level was not reached.
#' @export
walk.first.passage = function(level, p.up = 0.5, start = 0L, max.steps = 1e5) {
  if (level == start) stop("'level' must differ from 'start'", call. = FALSE)
  pos = start
  n = 0L
  while (pos != level) {
    if (n >= max.steps) return(NA_integer_)
    pos = pos + if (runif(1) < p.up) 1L else -1L
    n = n + 1L
  }
  n
}

#' Largest distance from the start along a path
#'
#' @param path An integer vector of positions as returned by [walk.1d()].
#' @return The largest absolute distance from the first position.
#' @export
walk.max.excursion = function(path) {
  max(abs(path - path[1]))
}

#' Number of returns to the starting point
#'
#' @param path An integer vector of positions as returned by [walk.1d()].
#' @return The number of times after the first step that the path is back at
#'   its first position.
#' @export
walk.return.count = function(path) {
  sum(path[-1] == path[1])
}

#' Play out the gambler's ruin
#'
#' A gambler bets one unit per round until the bankroll hits zero or the
#' target. Uses `repeat` so that the loop condition is checked after a bet.
#'
#' @param bankroll Starting number of units.
#' @param target Bankroll at which the gambler walks away.
#' @param p.win Probability of winning a single bet.
#' @return A list with `result` ("ruin" or "target") and `rounds`.
#' @export
gambler.ruin = function(bankroll = 10, target = 20, p.win = 0.5) {
  check.count(bankroll, "bankroll")
  check.probability(p.win, "p.win")
  if (target <= bankroll) stop("'target' must exceed 'bankroll'", call. = FALSE)
  rounds = 0L
  repeat {
    bankroll = bankroll + if (runif(1) < p.win) 1 else -1
    rounds = rounds + 1L
    if (bankroll == 0) {
      result = "ruin"
      break
    }
    if (bankroll == target) {
      result = "target"
      break
    }
  }
  list(result = result, rounds = rounds)
}

#' Summarise many random walks
#'
#' Repeats [walk.1d()] and collects the final position, the largest
#' excursion and the number of returns of each replicate.
#'
#' @param n.reps Number of walks to simulate.
#' @param n.steps Number of steps to take.
#' @param p.up Probability of a step up.
#' @return A data frame with one row per replicate.
#' @export
walk.summary.table = function(n.reps = 200, n.steps = 100, p.up = 0.5) {
  check.count(n.reps, "n.reps")
  final = integer(n.reps)
  excursion = integer(n.reps)
  returns = integer(n.reps)
  for (i in seq_len(n.reps)) {
    path = walk.1d(n.steps, p.up)
    final[i] = path[length(path)]
    excursion[i] = walk.max.excursion(path)
    returns[i] = walk.return.count(path)
  }
  data.frame(rep = seq_len(n.reps), final = final, excursion = excursion,
             returns = returns)
}
