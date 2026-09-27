config_woa <- function(
  iterations,
  population_size,
  iterations_same_cost = NULL,
  absolute_tol = NULL
) {
  p <- new("WOAConfig")
  commonOpt <- checkCommonConfigOptions(
    iterations,
    population_size,
    iterations_same_cost,
    absolute_tol
  )
  p@iterations <- commonOpt$iterations
  p@population_size <- commonOpt$population_size
  p@iterations_same_cost <- commonOpt$iterations_same_cost
  p@absolute_tol <- commonOpt$absolute_tol

  return(p)
}