#' Solve a 0/1 knapsack by dynamic programming
#'
#' Classic table-filling solution over integer capacities, followed by a
#' backtrack to recover the chosen items.
#'
#' @param value Numeric vector of item values.
#' @param weight Integer vector of item weights.
#' @param capacity Integer knapsack capacity.
#' @return A list with logical `chosen`, numeric `value` and integer `weight`.
#' @export
#' @family knapsack
knapsack.dp <- function(value, weight, capacity) {
  stopifnot(length(value) == length(weight), capacity >= 0)
  weight <- as.integer(weight)
  n <- length(value)
  tab <- matrix(0, nrow = n + 1L, ncol = capacity + 1L)
  for (i in seq_len(n)) {
    for (c in 0:capacity) {
      skip <- tab[i, c + 1L]
      if (weight[i] <= c) {
        take <- tab[i, c - weight[i] + 1L] + value[i]
        tab[i + 1L, c + 1L] <- max(skip, take)
      } else {
        tab[i + 1L, c + 1L] <- skip
      }
    }
  }
  chosen <- logical(n)
  c <- capacity
  for (i in n:1) {
    if (tab[i + 1L, c + 1L] != tab[i, c + 1L]) {
      chosen[i] <- TRUE
      c <- c - weight[i]
    }
  }
  list(chosen = chosen, value = sum(value[chosen]), weight = sum(weight[chosen]))
}

#' Solve a 0/1 knapsack greedily by value density
#'
#' Takes items in order of decreasing value per unit weight, then compares
#' against the single best item that fits, which bounds the error by half.
#'
#' @param value Numeric vector of item values.
#' @param weight Integer vector of item weights.
#' @param capacity Integer knapsack capacity.
#' @return A list with logical `chosen`, numeric `value` and integer `weight`.
#' @export
#' @family knapsack
knapsack.greedy <- function(value, weight, capacity) {
  stopifnot(length(value) == length(weight), capacity >= 0)
  n <- length(value)
  chosen <- logical(n)
  load <- 0
  for (i in order(value / weight, decreasing = TRUE)) {
    if (load + weight[i] <= capacity) {
      chosen[i] <- TRUE
      load <- load + weight[i]
    }
  }
  fits <- which(weight <= capacity)
  if (length(fits) > 0) {
    best.single <- fits[which.max(value[fits])]
    if (value[best.single] > sum(value[chosen])) {
      chosen <- logical(n)
      chosen[best.single] <- TRUE
    }
  }
  list(chosen = chosen, value = sum(value[chosen]), weight = sum(weight[chosen]))
}

#' Solve a 0/1 knapsack with lpSolve
#'
#' @param value Numeric vector of item values.
#' @param weight Integer vector of item weights.
#' @param capacity Integer knapsack capacity.
#' @return A list with logical `chosen`, numeric `value` and integer `weight`.
#' @export
#' @family knapsack
knapsack.lp <- function(value, weight, capacity) {
  stopifnot(length(value) == length(weight), capacity >= 0)
  sol <- lpSolve::lp(
    direction = "max",
    objective.in = value,
    const.mat = matrix(weight, nrow = 1),
    const.dir = "<=",
    const.rhs = capacity,
    all.bin = TRUE
  )
  if (sol$status != 0) {
    stop("lpSolve failed with status ", sol$status, call. = FALSE)
  }
  chosen <- sol$solution > 0.5
  list(chosen = chosen, value = sum(value[chosen]), weight = sum(weight[chosen]))
}

#' Pack items into bins with first fit
#'
#' Places each item in the first open bin with enough room, opening a new
#' bin when none fits.
#'
#' @param size Numeric vector of item sizes.
#' @param capacity Bin capacity.
#' @param decreasing Logical; sort items by decreasing size first (FFD).
#' @return An integer vector giving the bin of each item, in input order.
#' @export
#' @family binpacking
binpack.first.fit <- function(size, capacity, decreasing = TRUE) {
  if (any(size > capacity)) {
    stop("Some items are larger than the bin capacity.", call. = FALSE)
  }
  ord <- if (decreasing) order(size, decreasing = TRUE) else seq_along(size)
  bin <- integer(length(size))
  free <- numeric(0)
  for (i in ord) {
    fit <- which(free >= size[i])
    if (length(fit) > 0) {
      b <- fit[1]
    } else {
      free <- c(free, capacity)
      b <- length(free)
    }
    bin[i] <- b
    free[b] <- free[b] - size[i]
  }
  bin
}

#' Pack items into bins with best fit
#'
#' Places each item in the open bin that leaves the least room afterwards,
#' opening a new bin when none fits.
#'
#' @param size Numeric vector of item sizes.
#' @param capacity Bin capacity.
#' @param decreasing Logical; sort items by decreasing size first (BFD).
#' @return An integer vector giving the bin of each item, in input order.
#' @export
#' @family binpacking
binpack.best.fit <- function(size, capacity, decreasing = TRUE) {
  if (any(size > capacity)) {
    stop("Some items are larger than the bin capacity.", call. = FALSE)
  }
  ord <- if (decreasing) order(size, decreasing = TRUE) else seq_along(size)
  bin <- integer(length(size))
  free <- numeric(0)
  for (i in ord) {
    fit <- which(free >= size[i])
    if (length(fit) > 0) {
      b <- fit[which.min(free[fit] - size[i])]
    } else {
      free <- c(free, capacity)
      b <- length(free)
    }
    bin[i] <- b
    free[b] <- free[b] - size[i]
  }
  bin
}

#' Lower bound on the number of bins
#'
#' The larger of the volume bound and the count of items bigger than half
#' a bin.
#'
#' @param size Numeric vector of item sizes.
#' @param capacity Bin capacity.
#' @return An integer lower bound on the optimal bin count.
#' @export
#' @family binpacking
binpack.lower.bound <- function(size, capacity) {
  volume <- ceiling(sum(size) / capacity)
  big <- sum(size > capacity / 2)
  as.integer(max(volume, big))
}

#' Summarise a bin packing
#'
#' @param size Numeric vector of item sizes.
#' @param bin Integer vector of bin assignments as returned by
#'   [binpack.first.fit()].
#' @param capacity Bin capacity.
#' @return A data frame with one row per bin: `bin`, `n.items`, `load` and
#'   `fill`, with the lower bound and gap stored as attributes.
#' @export
#' @family binpacking
binpack.summary <- function(size, bin, capacity) {
  load <- tapply(size, bin, sum)
  n.items <- tapply(size, bin, length)
  out <- data.frame(
    bin = as.integer(names(load)),
    n.items = as.integer(n.items),
    load = as.numeric(load),
    fill = as.numeric(load) / capacity
  )
  lb <- binpack.lower.bound(size, capacity)
  attr(out, "lower.bound") <- lb
  attr(out, "gap") <- nrow(out) - lb
  out
}
