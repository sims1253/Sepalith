#' Create a population of agents
#'
#' Builds a data frame with one row per agent, placed uniformly at random on
#' a rectangular grid. A fixed number of agents start out infectious.
#'
#' @param n Number of agents in the population.
#' @param n_infected Number of agents that start in the infectious state.
#' @param width,height Dimensions of the arena the agents live in.
#' @return A data frame of class `herd_population` with columns `id`, `x`,
#'   `y`, `state` and `days_in_state`.
#' @export
#' @examples
#' pop <- new_population(200, n_infected = 3)
#' table(pop$state)
new_population <- function(n, n_infected = 1L, width = 100, height = 100) {
  stopifnot(n >= 1, n_infected >= 0, n_infected <= n)
  state <- rep("S", n)
  if (n_infected > 0) {
    state[sample.int(n, n_infected)] <- "I"
  }
  pop <- data.frame(
    id = seq_len(n),
    x = stats::runif(n, 0, width),
    y = stats::runif(n, 0, height),
    state = state,
    days_in_state = integer(n),
    stringsAsFactors = FALSE
  )
  attr(pop, "arena") <- c(width = width, height = height)
  class(pop) <- c("herd_population", class(pop))
  pop
}

#' Check that a population is well formed
#'
#' Verifies the required columns, the allowed state labels and that agent
#' ids are unique. Errors early so step functions can assume a clean input.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @return `pop`, invisibly, if all checks pass.
#' @keywords internal
validate_population <- function(pop) {
  required <- c("id", "x", "y", "state", "days_in_state")
  missing_cols <- setdiff(required, names(pop))
  if (length(missing_cols) > 0) {
    stop("Population is missing columns: ", paste(missing_cols, collapse = ", "),
      call. = FALSE)
  }
  bad_states <- setdiff(unique(pop$state), c("S", "E", "I", "R"))
  if (length(bad_states) > 0) {
    stop("Unknown agent states: ", paste(bad_states, collapse = ", "), call. = FALSE)
  }
  if (anyDuplicated(pop$id)) {
    stop("Agent ids must be unique.", call. = FALSE)
  }
  if (is.null(attr(pop, "arena"))) {
    stop("Population has no `arena` attribute.", call. = FALSE)
  }
  invisible(pop)
}

#' Count agents by state
#'
#' Tabulates the number of agents in each compartment, always returning all
#' four states even when some are empty.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @return A named integer vector with elements `S`, `E`, `I` and `R`.
#' @export
population_counts <- function(pop) {
  counts <- table(factor(pop$state, levels = c("S", "E", "I", "R")))
  stats::setNames(as.integer(counts), names(counts))
}

#' Add new agents to a population
#'
#' Appends agents in a given state at random positions in the arena. New
#' ids continue from the current maximum id.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param n Number of agents to add.
#' @param state State label assigned to the new agents.
#' @return The population with `n` extra rows.
#' @export
add_agents <- function(pop, n, state = "S") {
  validate_population(pop)
  if (n <= 0) {
    return(pop)
  }
  arena <- attr(pop, "arena")
  start_id <- max(pop$id, 0L)
  newcomers <- data.frame(
    id = start_id + seq_len(n),
    x = stats::runif(n, 0, arena[["width"]]),
    y = stats::runif(n, 0, arena[["height"]]),
    state = rep(state, n),
    days_in_state = integer(n),
    stringsAsFactors = FALSE
  )
  out <- rbind(as.data.frame(pop), newcomers)
  attr(out, "arena") <- arena
  class(out) <- class(pop)
  out
}

#' Remove agents from a population
#'
#' Drops the agents with the given ids, for example to model deaths or
#' emigration. Unknown ids are silently ignored.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param ids Integer vector of agent ids to remove.
#' @return The population without the removed rows.
#' @export
remove_agents <- function(pop, ids) {
  validate_population(pop)
  if (length(ids) == 0) {
    return(pop)
  }
  arena <- attr(pop, "arena")
  keep <- !(pop$id %in% ids)
  out <- as.data.frame(pop)[keep, , drop = FALSE]
  rownames(out) <- NULL
  attr(out, "arena") <- arena
  class(out) <- class(pop)
  out
}

#' Find the neighbours of an agent
#'
#' Returns the ids of all other agents within a Euclidean radius of the
#' focal agent.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param id Id of the focal agent.
#' @param radius Contact radius in arena units.
#' @return An integer vector of neighbour ids, possibly empty.
#' @export
agent_neighbours <- function(pop, id, radius = 2) {
  focal <- which(pop$id == id)
  if (length(focal) != 1) {
    stop("Agent ", id, " not found in population.", call. = FALSE)
  }
  dx <- pop$x - pop$x[focal]
  dy <- pop$y - pop$y[focal]
  within <- sqrt(dx^2 + dy^2) <= radius
  within[focal] <- FALSE
  pop$id[within]
}

#' Print a population
#'
#' Shows the population size, arena dimensions and compartment counts.
#'
#' @param x A `herd_population` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.herd_population <- function(x, ...) {
  arena <- attr(x, "arena")
  counts <- population_counts(x)
  cat("<herd_population> ", nrow(x), " agents on a ",
    arena[["width"]], " x ", arena[["height"]], " arena\n", sep = "")
  for (state in names(counts)) {
    cat("  ", state, ": ", counts[[state]], "\n", sep = "")
  }
  invisible(x)
}

#' Summarise a population
#'
#' Computes compartment counts, proportions and the mean time spent in the
#' current state.
#'
#' @param object A `herd_population` object.
#' @param ... Ignored.
#' @return A data frame with one row per state.
#' @export
summary.herd_population <- function(object, ...) {
  counts <- population_counts(object)
  mean_days <- vapply(names(counts), function(s) {
    days <- object$days_in_state[object$state == s]
    if (length(days) == 0) NA_real_ else mean(days)
  }, numeric(1))
  data.frame(
    state = names(counts),
    n = unname(counts),
    prop = unname(counts) / nrow(object),
    mean_days = unname(mean_days),
    stringsAsFactors = FALSE
  )
}
