#' Build a Monte Carlo design grid
#'
#' Crosses all factor levels into a grid of scenarios and assigns each a
#' reproducible seed.
#'
#' @param ... Named vectors of factor levels, e.g. `n = c(20, 50)`.
#' @param base.seed Seed from which per-scenario seeds are derived.
#' @return A tibble with one row per scenario and columns `scenario`,
#'   `seed` plus one column per factor.
#' @family monte carlo
#' @export
mc.design <- function(..., base.seed = 1L) {
  factors <- list(...)
  if (length(factors) == 0 || is.null(names(factors)) || any(names(factors) == "")) {
    stop("All design factors must be named.", call. = FALSE)
  }
  grid <- expand.grid(factors, stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
  grid <- tibble::as_tibble(grid) |>
    dplyr::mutate(
      scenario = dplyr::row_number(),
      seed = base.seed + 1000L * scenario,
      .before = 1
    )
  return(grid)
}

#' Run one Monte Carlo replicate safely
#'
#' Evaluates a simulation function and captures errors and warnings so a
#' single failing replicate does not abort the whole experiment.
#'
#' @param fun Simulation function returning a named numeric vector or list.
#' @param args Named list of arguments passed to `fun`.
#' @param seed Seed set before calling `fun`.
#' @return A list with elements `value`, `error`, `warning` and `elapsed`.
#' @keywords internal
mc.run.one <- function(fun, args, seed) {
  set.seed(seed)
  warn <- NA_character_
  started <- proc.time()[["elapsed"]]
  value <- tryCatch(
    withCallingHandlers(
      do.call(fun, args),
      warning = function(w) {
        warn <<- conditionMessage(w)
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) structure(conditionMessage(e), class = "mc.error")
  )
  failed <- inherits(value, "mc.error")
  return(list(
    value = if (failed) NULL else value,
    error = if (failed) unclass(value) else NA_character_,
    warning = warn,
    elapsed = proc.time()[["elapsed"]] - started
  ))
}

#' Run a Monte Carlo experiment
#'
#' Runs `reps` replicates of a simulation function for every scenario in a
#' design grid and stacks the results in long format.
#'
#' @param design A tibble as returned by [mc.design()].
#' @param fun Simulation function returning a named numeric vector or list.
#' @param reps Number of replicates per scenario.
#' @param progress Whether to print a line after each scenario.
#' @return A tibble with one row per scenario, replicate and statistic,
#'   with columns `scenario`, `rep`, `stat`, `value`, `error` and `warning`.
#' @family monte carlo
#' @export
mc.run <- function(design, fun, reps = 100, progress = interactive()) {
  factor.cols <- setdiff(names(design), c("scenario", "seed"))
  pieces <- vector("list", nrow(design) * reps)
  k <- 0L
  for (i in seq_len(nrow(design))) {
    args <- as.list(design[i, factor.cols])
    for (r in seq_len(reps)) {
      res <- mc.run.one(fun, args, seed = design$seed[i] + r)
      k <- k + 1L
      if (is.null(res$value)) {
        pieces[[k]] <- tibble::tibble(
          scenario = design$scenario[i], rep = r, stat = NA_character_,
          value = NA_real_, error = res$error, warning = res$warning
        )
      } else {
        vals <- unlist(res$value)
        pieces[[k]] <- tibble::tibble(
          scenario = design$scenario[i], rep = r, stat = names(vals),
          value = unname(vals), error = NA_character_, warning = res$warning
        )
      }
    }
    if (progress) {
      message("Scenario ", i, "/", nrow(design), " done")
    }
  }
  return(dplyr::bind_rows(pieces))
}

#' Monte Carlo bias
#'
#' @param estimate Numeric vector of replicate estimates.
#' @param truth True parameter value.
#' @return The mean deviation of the estimates from the truth.
#' @family performance measures
#' @export
mc.bias <- function(estimate, truth) {
  return(mean(estimate - truth, na.rm = TRUE))
}

#' Monte Carlo root mean squared error
#'
#' @param estimate Numeric vector of replicate estimates.
#' @param truth True parameter value.
#' @return The root mean squared deviation of the estimates from the truth.
#' @family performance measures
#' @export
mc.rmse <- function(estimate, truth) {
  return(sqrt(mean((estimate - truth)^2, na.rm = TRUE)))
}

#' Monte Carlo interval coverage
#'
#' @param lower,upper Numeric vectors of replicate interval bounds.
#' @param truth True parameter value.
#' @return The fraction of intervals that contain the truth.
#' @family performance measures
#' @export
mc.coverage <- function(lower, upper, truth) {
  return(mean(lower <= truth & truth <= upper, na.rm = TRUE))
}

#' Aggregate Monte Carlo results
#'
#' Summarises each statistic within each scenario by its mean, standard
#' deviation and Monte Carlo standard error, and joins the design factors
#' back on.
#'
#' @param results A tibble as returned by [mc.run()].
#' @param design A tibble as returned by [mc.design()].
#' @return A tibble with one row per scenario and statistic.
#' @family monte carlo
#' @export
mc.aggregate <- function(results, design) {
  failures <- results |>
    dplyr::group_by(scenario) |>
    dplyr::summarise(
      n.failed = dplyr::n_distinct(rep[!is.na(error)]),
      n.warned = dplyr::n_distinct(rep[!is.na(warning)]),
      .groups = "drop"
    )
  summary <- results |>
    dplyr::filter(!is.na(stat)) |>
    dplyr::group_by(scenario, stat) |>
    dplyr::summarise(
      n = dplyr::n(),
      mean = mean(value, na.rm = TRUE),
      sd = stats::sd(value, na.rm = TRUE),
      mcse = sd / sqrt(n),
      .groups = "drop"
    )
  out <- summary |>
    dplyr::left_join(failures, by = "scenario") |>
    dplyr::left_join(dplyr::select(design, -seed), by = "scenario") |>
    dplyr::arrange(scenario, stat)
  return(out)
}

#' Replicates needed for a target Monte Carlo error
#'
#' Uses pilot standard deviations to estimate how many replicates each
#' scenario needs so that the Monte Carlo standard error of the mean falls
#' below a target.
#'
#' @param aggregated A tibble as returned by [mc.aggregate()].
#' @param target.mcse Desired Monte Carlo standard error.
#' @return The input tibble with an added column `reps.needed`.
#' @family monte carlo
#' @export
mc.reps.needed <- function(aggregated, target.mcse) {
  stopifnot(target.mcse > 0)
  out <- aggregated |>
    dplyr::mutate(
      reps.needed = ceiling((sd / target.mcse)^2),
      reps.needed = pmax(reps.needed, n)
    )
  return(out)
}
