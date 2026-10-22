#' Vaccinate susceptible agents at random
#'
#' Moves a fraction of the susceptible agents directly into the recovered
#' compartment, chosen uniformly at random.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param coverage Fraction of susceptible agents to vaccinate.
#' @return The population with vaccinated agents in state `"R"`.
#' @export
vaccinate_random <- function(pop, coverage = 0.5) {
  stopifnot(coverage >= 0, coverage <= 1)
  sus <- which(pop$state == "S")
  n_vacc <- round(coverage * length(sus))
  if (n_vacc == 0) {
    return(pop)
  }

  chosen <- sus[sample.int(length(sus), n_vacc)]
  pop$state[chosen] <- "R"
  pop$days_in_state[chosen] <- 0L
  return(pop)
}

#' Vaccinate susceptible agents near a hotspot
#'
#' Ring vaccination: moves the susceptible agents closest to a given point
#' into the recovered compartment, nearest first.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param coverage Fraction of susceptible agents to vaccinate.
#' @param centre Numeric vector of length two giving the hotspot position.
#' @return The population with vaccinated agents in state `"R"`.
#' @export
vaccinate_ring <- function(pop, coverage = 0.5, centre = c(50, 50)) {
  stopifnot(coverage >= 0, coverage <= 1, length(centre) == 2)
  sus <- which(pop$state == "S")
  n_vacc <- round(coverage * length(sus))
  if (n_vacc == 0) {
    return(pop)
  }

  dist <- sqrt((pop$x[sus] - centre[[1]])^2 + (pop$y[sus] - centre[[2]])^2)
  chosen <- sus[order(dist)[seq_len(n_vacc)]]
  pop$state[chosen] <- "R"
  pop$days_in_state[chosen] <- 0L
  return(pop)
}

#' Isolate infectious agents
#'
#' Moves infectious agents that have been symptomatic for at least a given
#' number of days to a quarantine corner far outside the contact radius of
#' everyone else.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param delay Days in the infectious state before isolation starts.
#' @param compliance Probability that an eligible agent actually isolates.
#' @return The population with isolated agents relocated.
#' @export
quarantine_infectious <- function(pop, delay = 2L, compliance = 0.8) {
  eligible <- which(pop$state == "I" & pop$days_in_state >= delay)
  if (length(eligible) == 0) {
    return(pop)
  }

  isolate <- eligible[stats::runif(length(eligible)) < compliance]
  # Far enough away that no kernel radius can reach them.
  arena <- attr(pop, "arena")
  pop$x[isolate] <- -10 * arena[["width"]]
  pop$y[isolate] <- -10 * arena[["height"]]
  return(pop)
}

#' Apply a named intervention
#'
#' Dispatches to one of the intervention functions so that scenarios can be
#' described as plain lists in a configuration file.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param type Intervention type: `"none"`, `"random"`, `"ring"` or
#'   `"quarantine"`.
#' @param ... Passed to the matching intervention function.
#' @return The population after the intervention.
#' @export
apply_intervention <- function(pop, type = c("none", "random", "ring", "quarantine"), ...) {
  type <- match.arg(type)
  out <- switch(type,
    none = pop,
    random = vaccinate_random(pop, ...),
    ring = vaccinate_ring(pop, ...),
    quarantine = quarantine_infectious(pop, ...)
  )
  return(out)
}

#' Outbreak metrics
#'
#' Summaries of a single simulated outbreak. `peak_infections()` finds the
#' day with the most infectious agents, `attack_rate()` the fraction of the
#' population that was ever infected and `outbreak_duration()` the last day
#' with any exposed or infectious agent (`NA` if the outbreak never started).
#'
#' @param counts A data frame of daily counts as returned in the `counts`
#'   element of [run_herd()].
#' @return `peak_infections()` returns a list with elements `day` and
#'   `size`; the others return a single number.
#' @export
peak_infections <- function(counts) {
  i <- which.max(counts$I)
  return(list(day = counts$day[i], size = counts$I[i]))
}

#' @rdname peak_infections
#' @export
attack_rate <- function(counts) {
  first <- counts[1, ]
  last <- counts[nrow(counts), ]
  total <- first$S + first$E + first$I + first$R
  return((first$S - last$S) / total)
}

#' @rdname peak_infections
#' @export
outbreak_duration <- function(counts) {
  active <- which(counts$E + counts$I > 0)
  if (length(active) == 0) {
    return(NA_integer_)
  }
  return(counts$day[max(active)])
}

#' Compare intervention scenarios
#'
#' Runs the same starting population under several intervention scenarios
#' and collects the outbreak metrics for each replicate.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param scenarios A named list of scenarios; each element is a list with
#'   `type` and further arguments for [apply_intervention()].
#' @param reps Number of replicate runs per scenario.
#' @param days Maximum number of days to simulate.
#' @param params A named list of model parameters, see [run_herd()].
#' @return A data frame with one row per scenario and replicate.
#' @export
compare_scenarios <- function(pop, scenarios, reps = 10, days = 150, params = list()) {
  if (is.null(names(scenarios)) || any(names(scenarios) == "")) {
    stop("`scenarios` must be a fully named list.", call. = FALSE)
  }

  # Preallocate one row per scenario x replicate and fill in place.
  n_rows <- length(scenarios) * reps
  result <- data.frame(
    scenario = character(n_rows),
    rep = integer(n_rows),
    peak_day = integer(n_rows),
    peak_size = integer(n_rows),
    attack_rate = numeric(n_rows),
    duration = integer(n_rows),
    stringsAsFactors = FALSE
  )

  row <- 0L
  for (name in names(scenarios)) {
    spec <- scenarios[[name]]
    for (r in seq_len(reps)) {
      # Random interventions are redrawn for every replicate.
      start <- do.call(apply_intervention, c(list(pop), spec))
      run <- run_herd(start, days = days, params = params)
      peak <- peak_infections(run$counts)

      row <- row + 1L
      result$scenario[row] <- name
      result$rep[row] <- r
      result$peak_day[row] <- peak$day
      result$peak_size[row] <- peak$size
      result$attack_rate[row] <- attack_rate(run$counts)
      result$duration[row] <- outbreak_duration(run$counts)
    }
  }

  return(result)
}
