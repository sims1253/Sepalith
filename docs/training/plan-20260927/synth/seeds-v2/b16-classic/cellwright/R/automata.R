#' Lookup table for an elementary cellular automaton
#'
#' @description
#' Converts a Wolfram rule number into the eight output bits, ordered by
#' neighbourhood from `111` down to `000`.
#'
#' @param rule Rule number between 0 and 255.
#' @return An integer vector of length eight named by neighbourhood.
#' @export
ca.rule.table = function(rule) {
  if (rule < 0 || rule > 255 || rule != round(rule)) {
    stop("'rule' must be a whole number between 0 and 255", call. = FALSE)
  }
  # intToBits() is least significant first; reverse the low byte
  bits = as.integer(intToBits(rule))[8:1]
  names(bits) = c("111", "110", "101", "100", "011", "010", "001", "000")
  bits
}

#' Advance an elementary automaton by one generation
#'
#' @param state Integer vector of zeros and ones.
#' @param table Lookup table as returned by [ca.rule.table()].
#' @param wrap If `TRUE`, the row wraps around; otherwise cells beyond the
#'   edges count as zero.
#' @keywords internal
ca.step = function(state, table, wrap = TRUE) {
  n = length(state)
  if (wrap) {
    left = c(state[n], state[-n])
    right = c(state[-1], state[1])
  } else {
    left = c(0L, state[-n])
    right = c(state[-1], 0L)
  }

  # neighbourhood 111 sits at position 1, 000 at position 8
  idx = 8L - (4L * left + 2L * state + right)
  as.integer(table[idx])
}

#' Run an elementary automaton
#'
#' @param initial Integer vector of zeros and ones for the first generation.
#' @param rule Rule number between 0 and 255.
#' @param generations Number of generations to compute after the first.
#' @param wrap If `TRUE`, the row wraps around; otherwise cells beyond the
#'   edges count as zero.
#' @return An integer matrix with one row per generation.
#' @export
ca.run = function(initial, rule = 30, generations = 20, wrap = TRUE) {
  check.count(generations, "generations")
  table = ca.rule.table(rule)

  out = matrix(0L, nrow = generations + 1, ncol = length(initial))
  out[1, ] = as.integer(initial)
  for (g in seq_len(generations)) {
    out[g + 1, ] = ca.step(out[g, ], table, wrap = wrap)
  }
  out
}

#' Count live neighbours on a Life grid
#'
#' @description
#' Sums the eight surrounding cells of every cell on a torus.
#'
#' @param grid Logical or 0/1 matrix of live cells.
#' @keywords internal
life.neighbours = function(grid) {
  nr = nrow(grid)
  nc = ncol(grid)
  rows.up = c(nr, seq_len(nr - 1))
  rows.down = c(seq_len(nr)[-1], 1)
  cols.left = c(nc, seq_len(nc - 1))
  cols.right = c(seq_len(nc)[-1], 1)

  # Shift the whole grid once per direction and add the eight copies
  g = grid * 1L
  g[rows.up, , drop = FALSE] + g[rows.down, , drop = FALSE] +
    g[, cols.left, drop = FALSE] + g[, cols.right, drop = FALSE] +
    g[rows.up, cols.left, drop = FALSE] + g[rows.up, cols.right, drop = FALSE] +
    g[rows.down, cols.left, drop = FALSE] + g[rows.down, cols.right, drop = FALSE]
}

#' Advance a Life grid by one generation
#'
#' @param grid Logical or 0/1 matrix of live cells.
#' @param birth Neighbour counts that bring a dead cell to life.
#' @param survive Neighbour counts that keep a live cell alive.
#' @return A logical matrix the same size as `grid`.
#' @export
life.step = function(grid, birth = 3, survive = c(2, 3)) {
  n = life.neighbours(grid)
  alive = grid == 1
  (alive & n %in% survive) | (!alive & n %in% birth)
}

#' Run Life until it settles
#'
#' Steps the grid until it stops changing, dies out or reaches the
#' generation limit.
#'
#' @param grid Logical or 0/1 matrix of live cells.
#' @param max.gen Maximum number of generations.
#' @return A list with the final `grid`, the number of `generations` run and
#'   the `reason` for stopping.
#' @export
life.run = function(grid, max.gen = 500) {
  check.count(max.gen, "max.gen")
  grid = grid == 1
  gen = 0L
  reason = "limit"

  repeat {
    if (gen >= max.gen) break
    nxt = life.step(grid)
    gen = gen + 1L
    if (!any(nxt)) {
      reason = "extinct"
      grid = nxt
      break
    }
    # Only still lifes are detected; oscillators run to the limit
    if (identical(nxt, grid)) {
      reason = "still"
      break
    }
    grid = nxt
  }
  list(grid = grid, generations = gen, reason = reason)
}

#' Draw an automaton as text
#'
#' @param mat Matrix of zeros and ones, or a logical matrix.
#' @param on Character used for live cells.
#' @param off Character used for dead cells.
#' @return A character vector with one string per row, returned invisibly
#'   after being printed.
#' @export
ca.render = function(mat, on = "#", off = ".") {
  rows = apply(mat == 1, 1, function(r) paste(ifelse(r, on, off), collapse = ""))
  cat(rows, sep = "\n")
  invisible(rows)
}
