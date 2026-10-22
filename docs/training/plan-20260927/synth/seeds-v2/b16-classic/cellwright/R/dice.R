#' Argument checks
#'
#' @description
#' `check.count()` stops unless the value is a single positive whole number;
#' `check.probability()` stops unless it is a single number between 0 and 1.
#'
#' @param x Value to check.
#' @param name Name of the argument, used in the error message.
#' @rdname checks
#' @keywords internal
check.count = function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("'%s' must be a single number", name), call. = FALSE)
  }
  if (x < 1 || x != round(x)) {
    stop(sprintf("'%s' must be a positive whole number", name), call. = FALSE)
  }
  invisible(x)
}

#' @rdname checks
#' @keywords internal
check.probability = function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("'%s' must be a single number", name), call. = FALSE)
  }
  if (x < 0 || x > 1) {
    stop(sprintf("'%s' must lie between 0 and 1", name), call. = FALSE)
  }
  invisible(x)
}

#' Roll a set of dice
#'
#' @description
#' Draws independent throws of fair dice. Each throw rolls `n.dice` dice
#' with `n.faces` faces.
#'
#' @param n.dice Number of dice rolled per throw.
#' @param n.faces Number of faces on each die.
#' @param n.throws Number of throws to simulate.
#' @return An integer matrix with one row per throw and one column per die.
#' @export
roll.dice = function(n.dice = 2, n.faces = 6, n.throws = 1) {
  check.count(n.dice, "n.dice")
  check.count(n.faces, "n.faces")
  check.count(n.throws, "n.throws")

  rolls = sample.int(n.faces, n.dice * n.throws, replace = TRUE)
  matrix(rolls, nrow = n.throws, ncol = n.dice, byrow = TRUE)
}

#' Totals of each throw
#'
#' Adds up the faces shown in each row of a roll matrix.
#'
#' @param rolls An integer matrix of throws as returned by [roll.dice()].
#' @return An integer vector with one total per throw.
#' @export
throw.totals = function(rolls) {
  if (!is.matrix(rolls)) rolls = matrix(rolls, nrow = 1)
  as.integer(apply(rolls, 1, sum))
}

#' Detect doubles in each throw
#'
#' Flags the throws in which every die shows the same face.
#'
#' @param rolls An integer matrix of throws as returned by [roll.dice()].
#' @return A logical vector with one value per throw.
#' @export
throw.doubles = function(rolls) {
  if (!is.matrix(rolls)) rolls = matrix(rolls, nrow = 1)
  apply(rolls, 1, function(r) length(unique(r)) == 1L)
}

#' Count how often each face appears
#'
#' Tabulates all faces in a roll matrix, including faces that never came up.
#'
#' @param rolls An integer matrix of throws as returned by [roll.dice()].
#' @param n.faces Number of faces on each die.
#' @return A named integer vector of counts, one per face.
#' @export
tally.faces = function(rolls, n.faces = 6) {
  check.count(n.faces, "n.faces")
  faces = as.vector(rolls)
  # tabulate() silently drops values above nbins, which would undercount
  if (any(faces > n.faces | faces < 1, na.rm = TRUE)) {
    stop("'rolls' contains faces outside 1 to n.faces", call. = FALSE)
  }

  counts = tabulate(faces, nbins = n.faces)
  names(counts) = seq_len(n.faces)
  counts
}

#' Play one round of craps
#'
#' Throws two dice until the round is decided. A first throw of 7 or 11 wins,
#' 2, 3 or 12 loses, anything else sets a point that must be thrown again
#' before a 7.
#'
#' @param verbose If `TRUE`, print each throw as it happens.
#' @return A list with elements `outcome` ("win" or "lose"), `point` and
#'   `throws`.
#' @export
craps.round = function(verbose = FALSE) {
  first = sum(roll.dice(2, 6, 1))
  if (verbose) cat(sprintf("come-out roll: %d\n", first))

  # The come-out roll can settle the round straight away
  if (first %in% c(7, 11)) {
    return(list(outcome = "win", point = NA_integer_, throws = 1L))
  }
  if (first %in% c(2, 3, 12)) {
    return(list(outcome = "lose", point = NA_integer_, throws = 1L))
  }

  # Otherwise keep throwing until the point or a seven comes up
  n.throws = 1L
  outcome = NA_character_
  while (is.na(outcome)) {
    total = sum(roll.dice(2, 6, 1))
    n.throws = n.throws + 1L
    if (verbose) cat(sprintf("throw %d: %d\n", n.throws, total))
    if (total == first) {
      outcome = "win"
    } else if (total == 7) {
      outcome = "lose"
    }
  }
  list(outcome = outcome, point = as.integer(first), throws = n.throws)
}

#' Play one turn of Pig
#'
#' Keeps rolling a single die until the turn total reaches `hold.at` or a 1
#' is rolled, in which case the turn scores nothing.
#'
#' @param hold.at Turn total at which the player stops rolling.
#' @param n.faces Number of faces on each die.
#' @return A list with elements `score` and `rolls`, the faces thrown.
#' @export
pig.turn = function(hold.at = 20, n.faces = 6) {
  check.count(hold.at, "hold.at")
  check.count(n.faces, "n.faces")

  turn.total = 0
  history = integer(0)
  repeat {
    face = sample.int(n.faces, 1)
    history = c(history, face)
    if (face == 1L) {
      turn.total = 0
      break
    }
    turn.total = turn.total + face
    if (turn.total >= hold.at) break
  }
  list(score = turn.total, rolls = history)
}

#' Estimate the chance of a fixed total
#'
#' Simulates many throws and returns the share whose total equals `target`.
#'
#' @param target Total to look for.
#' @param n.dice Number of dice rolled per throw.
#' @param n.throws Number of throws to simulate.
#' @param conf Confidence level for the normal-approximation interval.
#' @return A named numeric vector with `estimate`, `lower` and `upper`.
#' @export
total.chance = function(target, n.dice = 2, n.throws = 10000, conf = 0.95) {
  check.probability(conf, "conf")
  hits = throw.totals(roll.dice(n.dice, 6, n.throws)) == target
  p = mean(hits)

  # Wald interval, clipped to [0, 1]
  z = qnorm(1 - (1 - conf) / 2)
  half = z * sqrt(p * (1 - p) / n.throws)
  c(estimate = p, lower = max(0, p - half), upper = min(1, p + half))
}
