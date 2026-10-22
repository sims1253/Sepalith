#' Validate a cost matrix
#'
#' @param cost A numeric matrix with workers in rows and tasks in columns.
#' @return `cost`, invisibly, with dimnames filled in when absent.
#' @keywords internal
checkCostMatrix <- function(cost) {
  if (!is.matrix(cost) || !is.numeric(cost)) {
    stop("`cost` must be a numeric matrix.", call. = FALSE)
  }
  if (anyNA(cost)) {
    stop("`cost` must not contain missing values.", call. = FALSE)
  }
  if (is.null(rownames(cost))) {
    rownames(cost) <- paste0("w", seq_len(nrow(cost)))
  }
  if (is.null(colnames(cost))) {
    colnames(cost) <- paste0("t", seq_len(ncol(cost)))
  }
  invisible(cost)
}

#' Turn an assignment vector into a tidy table
#'
#' @param cost A numeric matrix with workers in rows and tasks in columns.
#' @param assigned Integer vector giving the column assigned to each row, or
#'   `NA` for unassigned rows.
#' @return A data frame with columns `worker`, `task` and `cost`.
#' @keywords internal
assignmentTable <- function(cost, assigned) {
  ok <- !is.na(assigned)
  data.frame(
    worker = rownames(cost)[ok],
    task = colnames(cost)[assigned[ok]],
    cost = cost[cbind(which(ok), assigned[ok])],
    stringsAsFactors = FALSE
  )
}

#' Greedy assignment by globally cheapest pair
#'
#' Repeatedly picks the cheapest remaining worker-task pair. Fast but not
#' optimal; useful as a starting point or a baseline.
#'
#' @param cost A numeric matrix with workers in rows and tasks in columns.
#' @return A data frame with columns `worker`, `task` and `cost`.
#' @export
#' @family assignment
assignGreedy <- function(cost) {
  cost <- checkCostMatrix(cost)
  n <- nrow(cost)
  assigned <- rep(NA_integer_, n)
  work <- cost
  for (step in seq_len(min(dim(cost)))) {
    idx <- which(work == min(work), arr.ind = TRUE)[1, ]
    assigned[idx[1]] <- idx[2]
    work[idx[1], ] <- Inf
    work[, idx[2]] <- Inf
  }
  return(assignmentTable(cost, assigned))
}

#' Greedy assignment by regret
#'
#' At each step assigns the worker whose gap between cheapest and second
#' cheapest remaining task is largest, so that costly misses happen less.
#'
#' @param cost A numeric matrix with workers in rows and tasks in columns.
#' @return A data frame with columns `worker`, `task` and `cost`.
#' @export
#' @family assignment
assignRegret <- function(cost) {
  cost <- checkCostMatrix(cost)
  n <- nrow(cost)
  assigned <- rep(NA_integer_, n)
  work <- cost
  for (step in seq_len(min(dim(cost)))) {
    open <- which(is.na(assigned))
    regret <- vapply(open, function(i) {
      row <- sort(work[i, is.finite(work[i, ])])
      if (length(row) < 2) Inf else row[2] - row[1]
    }, numeric(1))
    i <- open[which.max(regret)]
    j <- which.min(work[i, ])
    assigned[i] <- j
    work[i, ] <- Inf
    work[, j] <- Inf
  }
  return(assignmentTable(cost, assigned))
}

#' Improve an assignment with pairwise swaps
#'
#' Tries exchanging tasks between every pair of workers and keeps any swap
#' that lowers total cost, until no swap helps.
#'
#' @param cost A numeric matrix with workers in rows and tasks in columns.
#' @param assignment A data frame as returned by [assignGreedy()].
#' @param maxIter Maximum number of improving swaps to apply.
#' @return A data frame with columns `worker`, `task` and `cost`.
#' @export
#' @family assignment
improveAssignment <- function(cost, assignment, maxIter = 1000L) {
  cost <- checkCostMatrix(cost)
  assigned <- match(assignment$task, colnames(cost))[match(rownames(cost), assignment$worker)]
  rows <- which(!is.na(assigned))
  iter <- 0L
  repeat {
    improved <- FALSE
    for (a in rows) {
      for (b in rows) {
        if (b <= a) {
          next
        }
        now <- cost[a, assigned[a]] + cost[b, assigned[b]]
        swapped <- cost[a, assigned[b]] + cost[b, assigned[a]]
        if (swapped < now - 1e-12) {
          tmp <- assigned[a]
          assigned[a] <- assigned[b]
          assigned[b] <- tmp
          improved <- TRUE
          iter <- iter + 1L
        }
      }
    }
    if (!improved || iter >= maxIter) {
      break
    }
  }
  return(assignmentTable(cost, assigned))
}

#' Solve the assignment problem exactly with lpSolve
#'
#' @param cost A numeric matrix with workers in rows and tasks in columns.
#' @param direction Whether to minimise cost or maximise value.
#' @return A data frame with columns `worker`, `task` and `cost`.
#' @export
#' @family assignment
assignExact <- function(cost, direction = c("min", "max")) {
  cost <- checkCostMatrix(cost)
  direction <- match.arg(direction)
  sol <- lpSolve::lp.assign(cost, direction = direction)
  if (sol$status != 0) {
    stop("lp.assign did not find a solution (status ", sol$status, ").", call. = FALSE)
  }
  assigned <- apply(sol$solution, 1, function(r) {
    hit <- which(r > 0.5)
    if (length(hit) == 0) NA_integer_ else hit[1]
  })
  return(assignmentTable(cost, assigned))
}

#' Compare assignment heuristics against the exact optimum
#'
#' @param cost A numeric matrix with workers in rows and tasks in columns.
#' @return A data frame with one row per method and columns `method`,
#'   `total` and `gap` (relative to the exact solution).
#' @export
#' @family assignment
compareAssignments <- function(cost) {
  cost <- checkCostMatrix(cost)
  greedy <- assignGreedy(cost)
  results <- list(
    greedy = greedy,
    regret = assignRegret(cost),
    greedySwap = improveAssignment(cost, greedy),
    exact = assignExact(cost)
  )
  totals <- vapply(results, function(r) sum(r$cost), numeric(1))
  best <- totals[["exact"]]
  out <- data.frame(
    method = names(totals),
    total = unname(totals),
    gap = unname((totals - best) / abs(best)),
    stringsAsFactors = FALSE
  )
  return(out[order(out$total), , drop = FALSE])
}
