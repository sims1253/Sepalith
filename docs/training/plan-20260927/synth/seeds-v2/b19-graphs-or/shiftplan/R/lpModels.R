#' Build a transportation model with ompr
#'
#' Minimises shipping cost from supply sites to demand sites subject to
#' supply limits and demand requirements.
#'
#' @param supply Named numeric vector of capacities per source.
#' @param demand Named numeric vector of requirements per destination.
#' @param cost Numeric matrix of unit costs, sources in rows and destinations
#'   in columns.
#' @return An unsolved `ompr` model.
#' @export
transportModel <- function(supply, demand, cost) {
  m <- length(supply)
  n <- length(demand)
  stopifnot(identical(dim(cost), c(m, n)))
  if (sum(supply) < sum(demand)) {
    stop("Total supply is less than total demand.", call. = FALSE)
  }

  ompr::MIPModel() |>
    ompr::add_variable(x[i, j], i = 1:m, j = 1:n, type = "continuous", lb = 0) |>
    ompr::set_objective(
      ompr::sum_over(cost[i, j] * x[i, j], i = 1:m, j = 1:n),
      sense = "min"
    ) |>
    ompr::add_constraint(ompr::sum_over(x[i, j], j = 1:n) <= supply[i], i = 1:m) |>
    ompr::add_constraint(ompr::sum_over(x[i, j], i = 1:m) >= demand[j], j = 1:n)
}

#' Build a facility location model with ompr
#'
#' Chooses which facilities to open and how to serve each customer, trading
#' fixed opening costs against service costs.
#'
#' @param fixedCost Named numeric vector of opening costs per facility.
#' @param capacity Named numeric vector of capacities per facility.
#' @param demand Named numeric vector of requirements per customer.
#' @param cost Numeric matrix of unit service costs, facilities in rows and
#'   customers in columns.
#' @return An unsolved `ompr` model.
#' @export
facilityModel <- function(fixedCost, capacity, demand, cost) {
  m <- length(fixedCost)
  n <- length(demand)
  stopifnot(length(capacity) == m, identical(dim(cost), c(m, n)))

  # Capacity is only available at a facility that is open
  ompr::MIPModel() |>
    ompr::add_variable(open[i], i = 1:m, type = "binary") |>
    ompr::add_variable(x[i, j], i = 1:m, j = 1:n, type = "continuous", lb = 0) |>
    ompr::set_objective(
      ompr::sum_over(fixedCost[i] * open[i], i = 1:m) +
        ompr::sum_over(cost[i, j] * x[i, j], i = 1:m, j = 1:n),
      sense = "min"
    ) |>
    ompr::add_constraint(ompr::sum_over(x[i, j], i = 1:m) == demand[j], j = 1:n) |>
    ompr::add_constraint(ompr::sum_over(x[i, j], j = 1:n) <= capacity[i] * open[i], i = 1:m)
}

#' Build a shift staffing model with ompr
#'
#' Covers hourly staffing requirements with shifts of fixed length starting
#' at any hour, minimising total staff-hours.
#'
#' @param required Integer vector of staff needed in each hour of the day.
#' @param shiftLength Length of one shift in hours.
#' @param wrap Logical; whether shifts may wrap past the last hour.
#' @return An unsolved `ompr` model.
#' @export
staffingModel <- function(required, shiftLength = 8L, wrap = TRUE) {
  h <- length(required)

  # covers[t, s] is 1 when a shift starting at hour s is on duty at hour t
  covers <- matrix(0L, nrow = h, ncol = h)
  for (s in seq_len(h)) {
    hours <- s + seq_len(shiftLength) - 1L
    if (wrap) {
      hours <- (hours - 1L) %% h + 1L
    } else {
      hours <- hours[hours <= h]
    }
    covers[hours, s] <- 1L
  }

  ompr::MIPModel() |>
    ompr::add_variable(y[s], s = 1:h, type = "integer", lb = 0) |>
    ompr::set_objective(ompr::sum_over(y[s], s = 1:h), sense = "min") |>
    ompr::add_constraint(ompr::sum_over(covers[t, s] * y[s], s = 1:h) >= required[t], t = 1:h)
}

#' Solve an ompr model and check its status
#'
#' @param model An `ompr` model.
#' @param solver Which ROI solver plugin to use.
#' @param verbose Logical; print solver output.
#' @return An `ompr` solution object.
#' @export
solveModel <- function(model, solver = c("glpk", "symphony", "highs"), verbose = FALSE) {
  solver <- match.arg(solver)

  res <- tryCatch(
    ompr::solve_model(model, ompr.roi::with_ROI(solver = solver, verbose = verbose)),
    error = function(e) {
      stop("Solver '", solver, "' failed: ", conditionMessage(e), call. = FALSE)
    }
  )

  # Infeasible or unbounded models still return a result; surface that
  status <- ompr::solver_status(res)
  if (status != "success") {
    warning("Solver returned status '", status, "'.", call. = FALSE)
  }
  res
}

#' Extract the shipment plan from a solved transport model
#'
#' @param result An `ompr` solution object.
#' @param sourceNames,destNames Character labels for the model indices.
#' @param tol Values below this are treated as zero and dropped.
#' @return A data frame with columns `source`, `dest` and `amount`.
#' @export
extractShipments <- function(result, sourceNames, destNames, tol = 1e-6) {
  sol <- ompr::get_solution(result, x[i, j])
  # Solvers report tiny non-zero flows; keep only real shipments
  sol <- sol[sol$value > tol, , drop = FALSE]
  data.frame(
    source = sourceNames[sol$i],
    dest = destNames[sol$j],
    amount = sol$value,
    stringsAsFactors = FALSE
  )
}

#' Extract opened facilities from a solved facility model
#'
#' @param result An `ompr` solution object.
#' @param facilityNames Character labels for the facility index.
#' @param tol Binary values within this distance of one count as open.
#' @return A data frame with columns `facility` and `open`.
#' @export
extractOpened <- function(result, facilityNames, tol = 1e-6) {
  sol <- ompr::get_solution(result, open[i])
  data.frame(
    facility = facilityNames[sol$i],
    open = sol$value > 1 - tol,
    stringsAsFactors = FALSE
  )
}

#' Extract shift starts from a solved staffing model
#'
#' @param result An `ompr` solution object.
#' @param hourLabels Character labels for the hour index.
#' @param tol Values below this are treated as zero and dropped.
#' @return A data frame with columns `start` and `staff`.
#' @export
extractShiftStarts <- function(result, hourLabels, tol = 1e-6) {
  sol <- ompr::get_solution(result, y[s])
  sol <- sol[sol$value > tol, , drop = FALSE]
  data.frame(
    start = hourLabels[sol$s],
    staff = as.integer(round(sol$value)),
    stringsAsFactors = FALSE
  )
}
