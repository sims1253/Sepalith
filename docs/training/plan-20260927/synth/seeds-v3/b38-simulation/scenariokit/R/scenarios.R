#' Toy stochastic epidemic
#'
#' A chain-binomial SIR model used as the default model of the experiment
#' runners.
#'
#' @param params A list with elements `beta`, `gamma`, `immune` (fraction
#'   immune at the start) and `seed.cases`.
#' @param days Number of days to simulate.
#' @param N Population size.
#' @return A data frame with columns `time`, `S`, `I`, `R` and `incidence`.
#' @export
toy.epidemic <- function(params, days = 100L, N = 1000L) {
  immune <- if (is.null(params$immune)) 0 else params$immune
  R0 <- round(N * immune)
  I0 <- if (is.null(params$seed.cases)) 5L else params$seed.cases
  S <- N - R0 - I0
  I <- I0
  R <- R0
  out <- matrix(0, nrow = days + 1L, ncol = 4L,
    dimnames = list(NULL, c("S", "I", "R", "incidence")))
  out[1L, ] <- c(S, I, R, 0)

  # Chain-binomial update, one row per day
  for (day in seq_len(days)) {
    infections <- stats::rbinom(1L, S, 1 - exp(-params$beta * I / N))
    recoveries <- stats::rbinom(1L, I, 1 - exp(-params$gamma))
    S <- S - infections
    I <- I + infections - recoveries
    R <- R + recoveries
    out[day + 1L, ] <- c(S, I, R, infections)
  }
  data.frame(time = 0:days, out)
}

#' Define intervention scenarios
#'
#' Each scenario is a copy of the baseline parameters with one effect
#' applied: `lockdown` and `school.closure` scale `beta` by their effect,
#' `vaccination` makes `1 - effect` of the population immune, `combined`
#' applies both `lockdown` and `vaccination`.
#'
#' @param base A list of baseline parameters, see [toy.epidemic()].
#' @param scenarios Scenarios to build, any of `"baseline"`, `"lockdown"`,
#'   `"vaccination"`, `"school.closure"` and `"combined"`.
#' @param effects A named vector of effect sizes for `lockdown`,
#'   `vaccination` and `school.closure`.
#' @return A named list of parameter lists, one per scenario, each with an
#'   extra element `label`.
#' @export
define.scenarios <- function(base,
                             scenarios = c("baseline", "lockdown", "vaccination",
                                           "school.closure", "combined"),
                             effects = c(lockdown = 0.5, vaccination = 0.6,
                                         school.closure = 0.8)) {
  scenarios <- match.arg(scenarios, several.ok = TRUE)
  out <- list()

  # Scenario `baseline`
  if ("baseline" %in% scenarios) {
    p <- base
    p$label <- "baseline"
    out[["baseline"]] <- p
  }
  # Scenario `lockdown`
  if ("lockdown" %in% scenarios) {
    p <- base
    p$beta <- base$beta * effects[["lockdown"]]
    p$label <- "lockdown"
    out[["lockdown"]] <- p
  }
  # Scenario `vaccination`
  if ("vaccination" %in% scenarios) {
    p <- base
    p$immune <- 1 - effects[["vaccination"]]
    p$label <- "vaccination"
    out[["vaccination"]] <- p
  }
  # Scenario `school.closure`
  if ("school.closure" %in% scenarios) {
    p <- base
    p$beta <- base$beta * effects[["school.closure"]]
    p$label <- "school.closure"
    out[["school.closure"]] <- p
  }
  # Scenario `combined`
  if ("combined" %in% scenarios) {
    p <- base
    p$beta <- base$beta * effects[["lockdown"]]
    p$immune <- 1 - effects[["vaccination"]]
    p$label <- "combined"
    out[["combined"]] <- p
  }
  out
}

#' Build a parameter grid
#'
#' @param beta,gamma Numeric vectors of transmission and recovery rates.
#' @param immune Numeric vector of initially immune fractions.
#' @return A list of parameter lists, one per combination, each with an
#'   extra element `label` of the form `"beta=0.3/gamma=0.1/immune=0"`.
#' @export
scenario.grid <- function(beta, gamma, immune = 0) {
  grid <- expand.grid(beta = beta, gamma = gamma, immune = immune,
    KEEP.OUT.ATTRS = FALSE)
  out <- vector("list", nrow(grid))
  for (i in seq_len(nrow(grid))) {
    p <- as.list(grid[i, ])
    p$label <- sprintf("beta=%g/gamma=%g/immune=%g", p$beta, p$gamma, p$immune)
    out[[i]] <- p
  }
  names(out) <- vapply(out, `[[`, character(1), "label")
  out
}

#' Run the intervention experiment
#'
#' Runs `n.reps` replicates of every scenario from [define.scenarios()].
#' Each scenario starts from the same seed, so scenarios share their random
#' numbers as far as the model allows (common random numbers).
#'
#' @param base A list of baseline parameters, see [toy.epidemic()].
#' @param n.reps Number of replicates per scenario.
#' @param days Number of days to simulate.
#' @param seed Seed set before each scenario.
#' @param model A function `(params, days)` returning a trajectory.
#' @return A named list with one element per scenario, each a list of
#'   `n.reps` trajectories.
#' @export
run.policy.experiment <- function(base, n.reps = 20L, days = 100L, seed = 1L,
                                  model = toy.epidemic) {
  scen <- define.scenarios(base)
  results <- list()

  # Run `baseline`
  set.seed(seed)
  results$baseline <- lapply(seq_len(n.reps), function(i) model(scen$baseline, days))
  # Run `lockdown`
  set.seed(seed)
  results$lockdown <- lapply(seq_len(n.reps), function(i) model(scen$lockdown, days))
  # Run `vaccination`
  set.seed(seed)
  results$vaccination <- lapply(seq_len(n.reps), function(i) model(scen$vaccination, days))
  # Run `school.closure`
  set.seed(seed)
  results$school.closure <- lapply(seq_len(n.reps),
    function(i) model(scen$school.closure, days))
  # Run `combined`
  set.seed(seed)
  results$combined <- lapply(seq_len(n.reps), function(i) model(scen$combined, days))
  results
}

#' Run a grid experiment
#'
#' @param grid A list of parameter lists, see [scenario.grid()].
#' @param n.reps Number of replicates per grid point.
#' @param days Number of days to simulate.
#' @param seed Seed set before each grid point.
#' @param model A function `(params, days)` returning a trajectory.
#' @return A named list with one element per grid point, each a list of
#'   `n.reps` trajectories.
#' @export
run.grid.experiment <- function(grid, n.reps = 10L, days = 100L, seed = 1L,
                                model = toy.epidemic) {
  if (!is.list(grid) || length(grid) == 0L) {
    stop("`grid` must be a non-empty list of parameter lists.", call. = FALSE)
  }
  results <- vector("list", length(grid))
  names(results) <- names(grid)
  for (i in seq_along(grid)) {
    set.seed(seed)
    results[[i]] <- lapply(seq_len(n.reps), function(r) model(grid[[i]], days))
  }
  results
}

#' Contrast scenarios against a reference
#'
#' @param results A named list of replicate lists, as returned by
#'   [run.policy.experiment()].
#' @param metric A function of one trajectory returning a single number.
#' @param reference Name of the reference scenario.
#' @return A data frame with columns `scenario`, `mean`, `se` (standard
#'   error of the mean), `difference` (to the reference mean), `ratio` and
#'   `paired.se` (standard error of the replicate-wise differences, valid
#'   under common random numbers).
#' @export
scenario.contrast <- function(results, metric, reference = "baseline") {
  if (!reference %in% names(results)) {
    stop(sprintf("Reference scenario `%s` not found.", reference), call. = FALSE)
  }
  values <- lapply(results, function(runs) vapply(runs, metric, numeric(1)))
  ref <- values[[reference]]
  out <- data.frame(scenario = names(values))

  # Column `mean`
  out$mean <- vapply(values, mean, numeric(1), USE.NAMES = FALSE)
  # Column `se`
  out$se <- vapply(values, function(v) stats::sd(v) / sqrt(length(v)), numeric(1),
    USE.NAMES = FALSE)
  # Column `difference`
  out$difference <- out$mean - mean(ref)
  # Column `ratio`
  out$ratio <- out$mean / mean(ref)
  # Column `paired.se`
  out$paired.se <- vapply(values, function(v) stats::sd(v - ref) / sqrt(length(v)),
    numeric(1), USE.NAMES = FALSE)
  out
}
