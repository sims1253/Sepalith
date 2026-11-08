#' Check a production plan against per-resource capacity
#'
#' @param plan A data frame with columns `period`, `machine_hours`,
#'   `labour_hours`, `tooling_sets` and `storage_m3`.
#' @param machine_hours,labour_hours,tooling_sets,storage_m3 Capacity of each
#'   resource per period, recycled over periods.
#' @return A data frame with one row per period and a utilisation column per
#'   resource; warns for every resource that is over capacity.
#' @export
check_capacity <- function(plan, machine_hours, labour_hours, tooling_sets, storage_m3) {
  periods <- sort(unique(plan$period))
  n <- length(periods)

  # Machine hours
  machine_load <- tapply(plan$machine_hours, plan$period, sum)[as.character(periods)]
  machine_util <- unname(machine_load) / rep_len(machine_hours, n)
  if (any(machine_util > 1)) {
    warning(sprintf("`%s` over capacity in %d period(s).", "machine_hours",
      sum(machine_util > 1)), call. = FALSE)
  }
  # Labour hours
  labour_load <- tapply(plan$labour_hours, plan$period, sum)[as.character(periods)]
  labour_util <- unname(labour_load) / rep_len(labour_hours, n)
  if (any(labour_util > 1)) {
    warning(sprintf("`%s` over capacity in %d period(s).", "labour_hours",
      sum(labour_util > 1)), call. = FALSE)
  }
  # Tooling sets
  tooling_load <- tapply(plan$tooling_sets, plan$period, sum)[as.character(periods)]
  tooling_util <- unname(tooling_load) / rep_len(tooling_sets, n)
  if (any(tooling_util > 1)) {
    warning(sprintf("`%s` over capacity in %d period(s).", "tooling_sets",
      sum(tooling_util > 1)), call. = FALSE)
  }
  # Storage volume
  storage_load <- tapply(plan$storage_m3, plan$period, sum)[as.character(periods)]
  storage_util <- unname(storage_load) / rep_len(storage_m3, n)
  if (any(storage_util > 1)) {
    warning(sprintf("`%s` over capacity in %d period(s).", "storage_m3",
      sum(storage_util > 1)), call. = FALSE)
  }

  data.frame(
    period = periods,
    machine_util = machine_util,
    labour_util = labour_util,
    tooling_util = tooling_util,
    storage_util = storage_util
  )
}

#' Validate per-resource capacity arguments
#'
#' @param machine_hours,labour_hours,tooling_sets,storage_m3 Capacity of each
#'   resource per period.
#' @return The capacities as a named list, invisibly.
#' @keywords internal
validate_capacity_args <- function(machine_hours, labour_hours, tooling_sets, storage_m3) {
  # machine_hours
  if (!is.numeric(machine_hours) || length(machine_hours) == 0) {
    stop(sprintf("`%s` must be a non-empty numeric vector.", "machine_hours"), call. = FALSE)
  }
  if (anyNA(machine_hours) || any(machine_hours < 0)) {
    stop(sprintf("`%s` must be non-negative and not missing.", "machine_hours"), call. = FALSE)
  }
  # labour_hours
  if (!is.numeric(labour_hours) || length(labour_hours) == 0) {
    stop(sprintf("`%s` must be a non-empty numeric vector.", "labour_hours"), call. = FALSE)
  }
  if (anyNA(labour_hours) || any(labour_hours < 0)) {
    stop(sprintf("`%s` must be non-negative and not missing.", "labour_hours"), call. = FALSE)
  }
  # tooling_sets
  if (!is.numeric(tooling_sets) || length(tooling_sets) == 0) {
    stop(sprintf("`%s` must be a non-empty numeric vector.", "tooling_sets"), call. = FALSE)
  }
  if (anyNA(tooling_sets) || any(tooling_sets < 0)) {
    stop(sprintf("`%s` must be non-negative and not missing.", "tooling_sets"), call. = FALSE)
  }
  if (any(tooling_sets != round(tooling_sets))) {
    stop(sprintf("`%s` must be whole numbers.", "tooling_sets"), call. = FALSE)
  }
  # storage_m3
  if (!is.numeric(storage_m3) || length(storage_m3) == 0) {
    stop(sprintf("`%s` must be a non-empty numeric vector.", "storage_m3"), call. = FALSE)
  }
  if (anyNA(storage_m3) || any(storage_m3 < 0)) {
    stop(sprintf("`%s` must be non-negative and not missing.", "storage_m3"), call. = FALSE)
  }

  invisible(list(
    machine_hours = machine_hours,
    labour_hours = labour_hours,
    tooling_sets = tooling_sets,
    storage_m3 = storage_m3
  ))
}

#' Resource requirement of an order book
#'
#' @param orders A data frame with columns `period` and `quantity`.
#' @param resource Resource to load, one of `"machine"`, `"labour"`, `"oven"`
#'   or `"dock"`.
#' @param rates Requirement per unit for each resource.
#' @param setup Fixed requirement per order for each resource.
#' @return A data frame with columns `period`, `resource`, `orders`, `units`
#'   and `required`.
#' @export
resource_requirement <- function(orders, resource = c("machine", "labour", "oven", "dock"),
                                 rates = c(machine = 0.5, labour = 0.75, oven = 0.25,
                                           dock = 0.05),
                                 setup = c(machine = 1, labour = 0.5, oven = 2, dock = 0.25)) {
  resource <- match.arg(resource)
  by_period <- split(orders$quantity, orders$period)
  units <- vapply(by_period, sum, numeric(1))
  n_orders <- lengths(by_period)

  required <- switch(resource,
    machine = {
      # Each order needs one changeover plus run time
      run <- units * rates[["machine"]]
      run + n_orders * setup[["machine"]]
    },
    labour = {
      # Operators attend the run and every changeover
      run <- units * rates[["labour"]]
      run + n_orders * setup[["labour"]]
    },
    oven = {
      # Ovens load in batches, so round the run up to whole cycles
      run <- ceiling(units * rates[["oven"]])
      run + n_orders * setup[["oven"]]
    },
    dock = {
      # Dock time follows outbound units plus one slot per order
      run <- units * rates[["dock"]]
      run + n_orders * setup[["dock"]]
    }
  )

  data.frame(
    period = names(by_period),
    resource = resource,
    orders = unname(n_orders),
    units = unname(units),
    required = unname(required)
  )
}

#' Level a load profile against capacity
#'
#' Excess load is first pushed forward into later periods with spare
#' capacity; whatever remains at the horizon is pulled back into earlier
#' periods.
#'
#' @param load Numeric load per period.
#' @param capacity Capacity per period, recycled to the length of `load`.
#' @return A list with elements `levelled`, `moved_forward`, `moved_back` and
#'   `unplaced`.
#' @export
level_load <- function(load, capacity) {
  n <- length(load)
  capacity <- rep_len(capacity, n)
  levelled <- load
  moved_forward <- 0
  moved_back <- 0

  # Forward pass: carry excess into the next period
  carry <- 0
  for (i in seq_len(n)) {
    levelled[i] <- levelled[i] + carry
    carry <- max(levelled[i] - capacity[i], 0)
    levelled[i] <- levelled[i] - carry
    if (i < n) moved_forward <- moved_forward + carry
  }

  # Backward pass: pull what is left into spare room of earlier periods
  for (i in rev(seq_len(n))) {
    room <- max(capacity[i] - levelled[i], 0)
    placed <- min(room, carry)
    levelled[i] <- levelled[i] + placed
    carry <- carry - placed
    moved_back <- moved_back + placed
  }

  list(
    levelled = levelled,
    moved_forward = moved_forward,
    moved_back = moved_back,
    unplaced = carry
  )
}

#' Summarise utilisation per resource
#'
#' @param util A data frame as returned by [check_capacity()].
#' @param threshold Utilisation above which a period counts as critical.
#' @return A data frame with one row per resource and columns `resource`,
#'   `mean`, `peak`, `peak_period`, `critical` and `over`.
#' @export
bottleneck_report <- function(util, threshold = 0.9) {
  resources <- c("machine", "labour", "tooling", "storage")
  out <- data.frame(
    resource = resources,
    mean = NA_real_,
    peak = NA_real_,
    peak_period = NA,
    critical = NA_integer_,
    over = NA_integer_
  )

  for (i in seq_along(resources)) {
    column <- util[[paste0(resources[[i]], "_util")]]
    out$mean[i] <- mean(column)
    out$peak[i] <- max(column)
    out$peak_period[i] <- util$period[which.max(column)]
    out$critical[i] <- sum(column > threshold)
    out$over[i] <- sum(column > 1)
  }

  # The bottleneck is the resource with the highest average utilisation
  out$bottleneck <- out$mean == max(out$mean)
  out[order(-out$mean), , drop = FALSE]
}
