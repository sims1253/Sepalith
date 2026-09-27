aib_build <- function(
  data,
  id,
  time,
  group,
  decision,
  outcome = NULL,
  verbose = TRUE
) {
  # ---- Input validation ------------------------------------------------
  data <- as_tibble(data)
  cols <- c(id, time, group, decision)
  if (!is.null(outcome)) {
    cols <- c(cols, outcome)
  }

  missing_cols <- setdiff(cols, names(data))
  if (length(missing_cols) > 0) {
    abort(c(
      "Missing columns in `data`:",
      paste0("  x ", missing_cols)
    ))
  }

  # Coerce decision to integer 0/1
  dec_vals <- unique(data[[decision]])
  if (!all(dec_vals %in% c(0L, 1L, 0, 1, TRUE, FALSE, NA))) {
    abort(c(
      "`decision` column must be binary (0/1).",
      paste0("  Found values: ", paste(dec_vals, collapse = ", "))
    ))
  }
  data[[decision]] <- as.integer(as.logical(data[[decision]]))

  # Coerce time to sorted integer ranks
  raw_times <- sort(unique(data[[time]]))
  time_map <- setNames(seq_along(raw_times), as.character(raw_times))
  data$.time_int <- time_map[as.character(data[[time]])]

  # Factor-encode group
  data[[group]] <- as.factor(data[[group]])
  groups <- levels(data[[group]])

  # Basic panel summary
  n_units <- length(unique(data[[id]]))
  n_times <- length(raw_times)
  n_groups <- length(groups)

  if (verbose) {
    cli_h1("AIBias: Building audit object")
    cli_alert_info("Units       : {n_units}")
    cli_alert_info(
      "Time points : {n_times} ({paste(raw_times, collapse = ', ')})"
    )
    cli_alert_info(
      "Groups      : {n_groups} ({paste(groups, collapse = ', ')})"
    )
    cli_alert_info(
      "Decision    : '{decision}' (mean = {round(mean(data[[decision]], na.rm=TRUE), 3)})"
    )
    if (!is.null(outcome)) {
      cli_alert_info("Outcome     : '{outcome}'")
    }
  }

  # ---- Compute base rates by group x time --------------------------------
  rates <- .compute_rates(data, group, decision, ".time_int")

  # ---- Build object ------------------------------------------------------
  obj <- structure(
    list(
      data = data,
      meta = list(
        id = id,
        time = time,
        time_int = ".time_int",
        time_map = time_map,
        group = group,
        decision = decision,
        outcome = outcome,
        groups = groups,
        n_units = n_units,
        n_times = n_times,
        raw_times = raw_times
      ),
      rates = rates,
      bias = NULL,
      transitions = NULL,
      amplification = NULL,
      adjusted = NULL,
      bootstrap = NULL,
      diagnostics = list()
    ),
    class = "aibias"
  )

  if (verbose) {
    cli_alert_success("Object built successfully.")
  }
  obj
}