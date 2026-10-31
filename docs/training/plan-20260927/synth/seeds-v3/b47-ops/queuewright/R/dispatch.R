#' Order a job queue by a dispatching rule
#'
#' @param jobs A data frame with columns `id`, `release`, `proc_time` and `due`.
#' @param rule Dispatching rule, one of `"fifo"`, `"spt"`, `"lpt"`, `"edd"`,
#'   `"slack"` or `"cr"` (critical ratio).
#' @param now Current time used by the time-dependent rules.
#' @return `jobs` reordered, with an added `position` column and a `rule`
#'   attribute.
#' @export
dispatch_order <- function(jobs, rule = c("fifo", "spt", "lpt", "edd", "slack", "cr"),
                           now = 0) {
  rule <- match.arg(rule)
  required <- c("id", "release", "proc_time", "due")
  missing_cols <- setdiff(required, names(jobs))
  if (length(missing_cols) > 0) {
    stop("`jobs` is missing columns: ", paste(missing_cols, collapse = ", "),
      call. = FALSE)
  }

  key <- switch(rule,
    fifo = {
      # Earliest release first, ties broken by release then id
      primary <- jobs$release
      secondary <- jobs$id
      order(primary, secondary)
    },
    spt = {
      # Shortest processing time first, ties broken by release
      primary <- jobs$proc_time
      secondary <- jobs$release
      order(primary, secondary)
    },
    lpt = {
      # Longest processing time first, ties broken by release
      primary <- -jobs$proc_time
      secondary <- jobs$release
      order(primary, secondary)
    },
    edd = {
      # Earliest due date first, ties broken by processing time
      primary <- jobs$due
      secondary <- jobs$proc_time
      order(primary, secondary)
    },
    slack = {
      # Least remaining slack first, ties broken by due date
      primary <- jobs$due - now - jobs$proc_time
      secondary <- jobs$due
      order(primary, secondary)
    },
    cr = {
      # Smallest critical ratio first, ties broken by due date
      primary <- (jobs$due - now) / jobs$proc_time
      secondary <- jobs$due
      order(primary, secondary)
    }
  )

  out <- jobs[key, , drop = FALSE]
  out$position <- seq_len(nrow(out))
  rownames(out) <- NULL
  attr(out, "rule") <- rule
  out
}

#' Simulate a single machine under a dispatching rule
#'
#' @param jobs A data frame with columns `id`, `release`, `proc_time` and `due`.
#' @param rule Dispatching rule, one of `"fifo"`, `"spt"`, `"lpt"`, `"edd"`,
#'   `"slack"` or `"cr"`.
#' @return A data frame with columns `id`, `start`, `finish`, `flow`,
#'   `lateness` and `tardiness`, in the input order.
#' @export
simulate_single_machine <- function(jobs,
                                    rule = c("fifo", "spt", "lpt", "edd", "slack", "cr")) {
  rule <- match.arg(rule)
  n <- nrow(jobs)
  start <- rep(NA_real_, n)
  finish <- rep(NA_real_, n)
  done <- logical(n)
  clock <- min(jobs$release)

  for (step in seq_len(n)) {
    ready <- which(!done & jobs$release <= clock)
    if (length(ready) == 0) {
      # Machine idles until the next release
      clock <- min(jobs$release[!done])
      ready <- which(!done & jobs$release <= clock)
    }
    score <- switch(rule,
      fifo = jobs$release[ready],
      spt = jobs$proc_time[ready],
      lpt = -jobs$proc_time[ready],
      edd = jobs$due[ready],
      slack = jobs$due[ready] - clock - jobs$proc_time[ready],
      cr = (jobs$due[ready] - clock) / jobs$proc_time[ready]
    )
    pick <- ready[which.min(score)]
    start[pick] <- clock
    finish[pick] <- clock + jobs$proc_time[pick]
    clock <- finish[pick]
    done[pick] <- TRUE
  }

  data.frame(
    id = jobs$id,
    start = start,
    finish = finish,
    flow = finish - jobs$release,
    lateness = finish - jobs$due,
    tardiness = pmax(finish - jobs$due, 0)
  )
}

#' Performance measures of a single-machine schedule
#'
#' @param schedule A data frame as returned by [simulate_single_machine()].
#' @param metrics Measures to compute, any of `"makespan"`, `"mean_flow"`,
#'   `"max_flow"`, `"mean_tardiness"`, `"max_lateness"`, `"n_tardy"` and
#'   `"idle"`.
#' @return A named numeric vector with one element per entry of `metrics`.
#' @export
schedule_metrics <- function(schedule,
                             metrics = c("makespan", "mean_flow", "max_flow",
                                         "mean_tardiness", "max_lateness",
                                         "n_tardy", "idle")) {
  metrics <- match.arg(metrics, several.ok = TRUE)
  out <- setNames(numeric(length(metrics)), metrics)
  busy <- sum(schedule$finish - schedule$start)

  for (metric in metrics) {
    out[[metric]] <- switch(metric,
      makespan = {
        # Span from the first start to the last finish
        span <- max(schedule$finish) - min(schedule$start)
        span
      },
      mean_flow = {
        # Average time a job spends in the system
        flow <- schedule$flow
        mean(flow)
      },
      max_flow = {
        # Longest time a job spends in the system
        flow <- schedule$flow
        max(flow)
      },
      mean_tardiness = {
        # Average positive lateness over all jobs
        tardiness <- schedule$tardiness
        mean(tardiness)
      },
      max_lateness = {
        # Largest lateness, negative when every job is early
        lateness <- schedule$lateness
        max(lateness)
      },
      n_tardy = {
        # Number of jobs that finish after their due date
        tardiness <- schedule$tardiness
        sum(tardiness > 0)
      },
      idle = {
        # Machine time inside the span that is not busy
        span <- max(schedule$finish) - min(schedule$start)
        span - busy
      }
    )
  }
  out
}

#' Compare dispatching rules on one job set
#'
#' @param jobs A data frame with columns `id`, `release`, `proc_time` and `due`.
#' @param rules Dispatching rules to compare.
#' @param metrics Measures passed to [schedule_metrics()].
#' @return A data frame with one row per rule, one column per measure and a
#'   `best` column naming the measures on which the rule ranks first.
#' @export
compare_rules <- function(jobs, rules = c("fifo", "spt", "lpt", "edd", "slack", "cr"),
                          metrics = c("makespan", "mean_flow", "mean_tardiness", "n_tardy")) {
  rows <- vector("list", length(rules))
  for (i in seq_along(rules)) {
    schedule <- simulate_single_machine(jobs, rules[[i]])
    values <- schedule_metrics(schedule, metrics)
    rows[[i]] <- as.data.frame(as.list(values))
  }
  out <- do.call(rbind, rows)
  out <- cbind(rule = rules, out)

  # Mark the rule(s) with the smallest value of each measure
  best <- character(length(rules))
  for (metric in metrics) {
    winners <- out[[metric]] == min(out[[metric]])
    best[winners] <- ifelse(best[winners] == "", metric,
      paste(best[winners], metric, sep = ", "))
  }
  out$best <- best
  out
}

#' Composite priority score for a job queue
#'
#' Each component is rescaled to the unit interval before weighting, so the
#' weights are comparable across components.
#'
#' @param jobs A data frame with columns `id`, `release`, `proc_time`, `due`
#'   and `weight`.
#' @param weights Named weights for the components `proc_time`, `due`,
#'   `slack`, `release` and `weight`.
#' @param now Current time used by the slack component.
#' @return `jobs` sorted by increasing `score`, with the `score` column added.
#' @export
weighted_dispatch <- function(jobs,
                              weights = c(proc_time = 1, due = 1, slack = 0,
                                          release = 0, weight = 0),
                              now = 0) {
  rescale <- function(x) {
    spread <- diff(range(x))
    if (spread == 0) return(rep(0, length(x)))
    (x - min(x)) / spread
  }
  score <- numeric(nrow(jobs))

  # Shorter jobs score lower
  if (weights[["proc_time"]] > 0) {
    component <- rescale(jobs$proc_time)
    score <- score + weights[["proc_time"]] * component
  }
  # Earlier due dates score lower
  if (weights[["due"]] > 0) {
    component <- rescale(jobs$due)
    score <- score + weights[["due"]] * component
  }
  # Less slack scores lower
  if (weights[["slack"]] > 0) {
    component <- rescale(jobs$due - now - jobs$proc_time)
    score <- score + weights[["slack"]] * component
  }
  # Earlier releases score lower
  if (weights[["release"]] > 0) {
    component <- rescale(jobs$release)
    score <- score + weights[["release"]] * component
  }
  # Heavier customer weights score lower
  if (weights[["weight"]] > 0) {
    component <- rescale(-jobs$weight)
    score <- score + weights[["weight"]] * component
  }

  jobs$score <- score
  out <- jobs[order(score, jobs$id), , drop = FALSE]
  rownames(out) <- NULL
  out
}
