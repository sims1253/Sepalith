#' Check a transition matrix
#'
#' Verifies that a matrix is square, non-negative and row-stochastic, and
#' fills in state names if they are missing.
#'
#' @param P A square transition matrix with rows summing to one.
#' @param tol Tolerance used when checking row sums.
#' @return `P`, with row and column names set.
#' @keywords internal
check.transition.matrix <- function(P, tol = 1e-8) {
  if (!is.matrix(P) || nrow(P) != ncol(P)) {
    stop("`P` must be a square matrix.", call. = FALSE)
  }
  if (any(P < 0)) {
    stop("`P` must not contain negative probabilities.", call. = FALSE)
  }
  if (any(abs(rowSums(P) - 1) > tol)) {
    stop("Rows of `P` must sum to one.", call. = FALSE)
  }
  states <- rownames(P)
  if (is.null(states)) {
    states <- as.character(seq_len(nrow(P)))
  }
  dimnames(P) <- list(states, states)
  return(P)
}

#' Simulate a discrete-time Markov chain
#'
#' Generates a path of a finite Markov chain from a transition matrix and a
#' starting state.
#'
#' @param P A square transition matrix with rows summing to one.
#' @param n.steps Number of transitions to simulate.
#' @param start Starting state, given by name or index.
#' @return A character vector of length `n.steps + 1` with visited states.
#' @family markov chains
#' @export
mc.chain.path <- function(P, n.steps, start = 1) {
  P <- check.transition.matrix(P)
  states <- rownames(P)
  current <- if (is.character(start)) match(start, states) else as.integer(start)
  if (is.na(current) || current < 1 || current > length(states)) {
    stop("`start` is not a valid state.", call. = FALSE)
  }
  path <- integer(n.steps + 1)
  path[1] <- current
  for (i in seq_len(n.steps)) {
    current <- sample.int(length(states), 1, prob = P[current, ])
    path[i + 1] <- current
  }
  return(states[path])
}

#' Multi-step transition probabilities
#'
#' Computes `P^n` by repeated squaring.
#'
#' @param P A square transition matrix with rows summing to one.
#' @param n Number of steps.
#' @return The `n`-step transition matrix.
#' @family markov chains
#' @export
mc.n.step <- function(P, n) {
  P <- check.transition.matrix(P)
  result <- diag(nrow(P))
  dimnames(result) <- dimnames(P)
  base <- P
  while (n > 0) {
    if (n %% 2 == 1) {
      result <- result %*% base
    }
    base <- base %*% base
    n <- n %/% 2
  }
  return(result)
}

#' Stationary distribution of a Markov chain
#'
#' Solves `pi P = pi` with `sum(pi) = 1` via the left eigenvector for the
#' eigenvalue one.
#'
#' @param P A square transition matrix with rows summing to one.
#' @return A named numeric vector of stationary probabilities.
#' @family markov chains
#' @export
mc.stationary <- function(P) {
  P <- check.transition.matrix(P)
  eig <- eigen(t(P))
  idx <- which.min(abs(eig$values - 1))
  v <- Re(eig$vectors[, idx])
  v <- v / sum(v)
  v[abs(v) < 1e-12] <- 0
  return(stats::setNames(v, rownames(P)))
}

#' Estimate a transition matrix from an observed path
#'
#' Counts observed transitions and normalises each row. Rows for states
#' that were never left are filled with a self-loop.
#'
#' @param path Character or factor vector of visited states.
#' @param states Optional character vector of all possible states.
#' @return A list with elements `counts` and `P`.
#' @family markov chains
#' @export
mc.fit.path <- function(path, states = NULL) {
  path <- as.character(path)
  if (is.null(states)) {
    states <- sort(unique(path))
  }
  k <- length(states)
  counts <- matrix(0L, k, k, dimnames = list(states, states))
  from <- match(utils::head(path, -1), states)
  to <- match(path[-1], states)
  for (i in seq_along(from)) {
    counts[from[i], to[i]] <- counts[from[i], to[i]] + 1L
  }
  totals <- rowSums(counts)
  P <- counts / ifelse(totals == 0, 1, totals)
  for (s in which(totals == 0)) {
    P[s, s] <- 1
  }
  return(list(counts = counts, P = P))
}

#' Absorption probabilities and expected time to absorption
#'
#' For an absorbing chain, uses the fundamental matrix to compute the
#' probability of ending in each absorbing state and the expected number of
#' steps until absorption from every transient state.
#'
#' @param P A square transition matrix with rows summing to one.
#' @return A list with elements `absorbing`, `transient`, `B` (absorption
#'   probabilities) and `steps` (expected steps to absorption).
#' @family markov chains
#' @export
mc.absorption <- function(P) {
  P <- check.transition.matrix(P)
  absorbing <- which(diag(P) == 1)
  transient <- setdiff(seq_len(nrow(P)), absorbing)
  if (length(absorbing) == 0) {
    stop("Chain has no absorbing states.", call. = FALSE)
  }
  if (length(transient) == 0) {
    stop("Chain has no transient states.", call. = FALSE)
  }
  Q <- P[transient, transient, drop = FALSE]
  R <- P[transient, absorbing, drop = FALSE]
  N <- tryCatch(
    solve(diag(length(transient)) - Q),
    error = function(e) {
      stop("Some transient states cannot reach an absorbing state.", call. = FALSE)
    }
  )
  B <- N %*% R
  steps <- rowSums(N)
  names(steps) <- rownames(P)[transient]
  return(list(
    absorbing = rownames(P)[absorbing],
    transient = rownames(P)[transient],
    B = B,
    steps = steps
  ))
}

#' Empirical hitting time of a target state
#'
#' Simulates repeated chains from a start state and records the first step
#' at which the target state is visited.
#'
#' @param P A square transition matrix with rows summing to one.
#' @param start Starting state, given by name or index.
#' @param target Target state, given by name or index.
#' @param reps Number of simulated chains.
#' @param max.steps Maximum path length; chains that never hit the target
#'   are recorded as `NA`.
#' @return An integer vector of hitting times, one per replicate.
#' @family markov chains
#' @export
mc.hitting.times <- function(P, start, target, reps = 1000, max.steps = 1000) {
  P <- check.transition.matrix(P)
  states <- rownames(P)
  target.name <- if (is.character(target)) target else states[target]
  hits <- rep(NA_integer_, reps)
  for (r in seq_len(reps)) {
    path <- mc.chain.path(P, max.steps, start = start)
    first <- match(target.name, path[-1])
    if (!is.na(first)) {
      hits[r] <- first
    }
  }
  return(hits)
}
