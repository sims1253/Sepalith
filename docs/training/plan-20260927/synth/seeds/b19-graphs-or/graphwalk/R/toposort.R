#' Topological order with Kahn's algorithm
#'
#' Repeatedly removes vertices with no remaining incoming edges. Ties are
#' broken alphabetically so the result is reproducible.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()],
#'   built with `directed = TRUE`.
#' @return A character vector of vertex names in topological order.
#' @export
#' @family toposort
topo_sort_kahn <- function(adj) {
  indeg <- in_degree(adj)
  ready <- sort(names(indeg)[indeg == 0])
  order <- character(0)
  while (length(ready) > 0) {
    v <- ready[1]
    ready <- ready[-1]
    order <- c(order, v)
    for (u in adj[[v]]) {
      indeg[u] <- indeg[u] - 1L
      if (indeg[u] == 0L) {
        ready <- sort(c(ready, u))
      }
    }
  }
  if (length(order) < length(adj)) {
    stuck <- setdiff(names(adj), order)
    stop("Graph has a cycle involving: ", paste(stuck, collapse = ", "), call. = FALSE)
  }
  return(order)
}

#' Topological order with depth-first search
#'
#' Emits vertices in reverse post-order. Uses three-state marking to detect
#' back edges, i.e. cycles.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()],
#'   built with `directed = TRUE`.
#' @return A character vector of vertex names in topological order.
#' @export
#' @family toposort
topo_sort_dfs <- function(adj) {
  state <- setNames(rep(0L, length(adj)), names(adj))
  order <- character(0)
  visit <- function(v) {
    if (state[v] == 2L) {
      return(invisible(NULL))
    }
    if (state[v] == 1L) {
      stop("Graph has a cycle through vertex ", v, call. = FALSE)
    }
    state[v] <<- 1L
    for (u in adj[[v]]) {
      visit(u)
    }
    state[v] <<- 2L
    order <<- c(v, order)
    invisible(NULL)
  }
  for (v in sort(names(adj))) {
    if (state[v] == 0L) {
      visit(v)
    }
  }
  return(order)
}

#' Check whether a directed graph is acyclic
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()],
#'   built with `directed = TRUE`.
#' @return `TRUE` if the graph has no directed cycle, otherwise `FALSE`.
#' @export
#' @family toposort
is_dag <- function(adj) {
  ok <- tryCatch(
    {
      topo_sort_kahn(adj)
      TRUE
    },
    error = function(e) FALSE
  )
  return(ok)
}

#' Earliest start times for a task network
#'
#' Forward pass of the critical path method. Each task can start once all of
#' its predecessors have finished.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()],
#'   built with `directed = TRUE`.
#' @param duration A named numeric vector of task durations.
#' @return A data frame with columns `task`, `duration`, `earliest_start` and
#'   `earliest_finish`, in topological order.
#' @export
#' @family critical-path
cpm_forward <- function(adj, duration) {
  missing_tasks <- setdiff(names(adj), names(duration))
  if (length(missing_tasks) > 0) {
    stop("No duration for: ", paste(missing_tasks, collapse = ", "), call. = FALSE)
  }
  order <- topo_sort_kahn(adj)
  es <- setNames(numeric(length(order)), order)
  ef <- setNames(numeric(length(order)), order)
  for (v in order) {
    ef[v] <- es[v] + duration[[v]]
    for (u in adj[[v]]) {
      if (ef[v] > es[u]) {
        es[u] <- ef[v]
      }
    }
  }
  out <- data.frame(
    task = order,
    duration = unname(duration[order]),
    earliest_start = unname(es),
    earliest_finish = unname(ef),
    stringsAsFactors = FALSE
  )
  return(out)
}

#' Latest start times for a task network
#'
#' Backward pass of the critical path method, anchored at the project
#' finish time from the forward pass.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()],
#'   built with `directed = TRUE`.
#' @param duration A named numeric vector of task durations.
#' @return A data frame with columns `task`, `duration`, `latest_start` and
#'   `latest_finish`, in topological order.
#' @export
#' @family critical-path
cpm_backward <- function(adj, duration) {
  fwd <- cpm_forward(adj, duration)
  order <- fwd$task
  horizon <- max(fwd$earliest_finish)
  lf <- setNames(rep(horizon, length(order)), order)
  ls <- setNames(numeric(length(order)), order)
  for (v in rev(order)) {
    succ <- adj[[v]]
    if (length(succ) > 0) {
      lf[v] <- min(ls[succ])
    }
    ls[v] <- lf[v] - duration[[v]]
  }
  out <- data.frame(
    task = order,
    duration = unname(duration[order]),
    latest_start = unname(ls),
    latest_finish = unname(lf),
    stringsAsFactors = FALSE
  )
  return(out)
}

#' Critical path schedule with slack
#'
#' Combines the forward and backward passes and flags zero-slack tasks.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()],
#'   built with `directed = TRUE`.
#' @param duration A named numeric vector of task durations.
#' @param tol Numeric tolerance below which slack is treated as zero.
#' @return A data frame with forward and backward times, `slack` and a logical
#'   `critical` column.
#' @export
#' @family critical-path
cpm_schedule <- function(adj, duration, tol = 1e-9) {
  fwd <- cpm_forward(adj, duration)
  bwd <- cpm_backward(adj, duration)
  out <- merge(fwd, bwd[, c("task", "latest_start", "latest_finish")], by = "task")
  out <- out[match(fwd$task, out$task), , drop = FALSE]
  out$slack <- out$latest_start - out$earliest_start
  out$critical <- abs(out$slack) < tol
  rownames(out) <- NULL
  return(out)
}
