#' Validate a job table
#'
#' Checks the columns every single-machine heuristic in this file relies on.
#'
#' @param jobs A data frame of jobs with columns `job`, `processing`, `due`
#'   and optionally `weight` and `release`.
#' @return `jobs`, invisibly, with `weight` and `release` filled in when absent.
#' @keywords internal
checkJobs <- function(jobs) {
  required <- c("job", "processing", "due")
  missingCols <- setdiff(required, names(jobs))
  if (length(missingCols) > 0) {
    stop("Job table is missing: ", paste(missingCols, collapse = ", "), call. = FALSE)
  }
  if (any(jobs$processing <= 0)) {
    stop("Processing times must be positive.", call. = FALSE)
  }
  if (!"weight" %in% names(jobs)) {
    jobs$weight <- 1
  }
  if (!"release" %in% names(jobs)) {
    jobs$release <- 0
  }
  invisible(jobs)
}

#' Evaluate a job sequence on a single machine
#'
#' Simulates the sequence without idle insertion beyond release dates and
#' computes start, completion, lateness and tardiness per job.
#'
#' @param jobs A data frame of jobs with columns `job`, `processing`, `due`
#'   and optionally `weight` and `release`.
#' @param sequence Integer row indices of `jobs` in processing order.
#' @return A data frame with one row per job in sequence order.
#' @export
#' @family singleMachine
evaluateSequence <- function(jobs, sequence) {
  jobs <- checkJobs(jobs)
  n <- length(sequence)
  start <- numeric(n)
  completion <- numeric(n)
  clock <- 0
  for (k in seq_len(n)) {
    j <- sequence[k]
    start[k] <- max(clock, jobs$release[j])
    completion[k] <- start[k] + jobs$processing[j]
    clock <- completion[k]
  }
  out <- jobs[sequence, c("job", "processing", "due", "weight"), drop = FALSE]
  out$position <- seq_len(n)
  out$start <- start
  out$completion <- completion
  out$lateness <- completion - out$due
  out$tardiness <- pmax(out$lateness, 0)
  rownames(out) <- NULL
  return(out)
}

#' Sequence jobs by earliest due date
#'
#' @param jobs A data frame of jobs with columns `job`, `processing`, `due`
#'   and optionally `weight` and `release`.
#' @return A data frame as returned by [evaluateSequence()].
#' @export
#' @family singleMachine
scheduleEdd <- function(jobs) {
  jobs <- checkJobs(jobs)
  sequence <- order(jobs$due, jobs$processing)
  return(evaluateSequence(jobs, sequence))
}

#' Sequence jobs by shortest processing time
#'
#' @param jobs A data frame of jobs with columns `job`, `processing`, `due`
#'   and optionally `weight` and `release`.
#' @return A data frame as returned by [evaluateSequence()].
#' @export
#' @family singleMachine
scheduleSpt <- function(jobs) {
  jobs <- checkJobs(jobs)
  sequence <- order(jobs$processing, jobs$due)
  return(evaluateSequence(jobs, sequence))
}

#' Sequence jobs by weighted shortest processing time
#'
#' @param jobs A data frame of jobs with columns `job`, `processing`, `due`
#'   and optionally `weight` and `release`.
#' @return A data frame as returned by [evaluateSequence()].
#' @export
#' @family singleMachine
scheduleWspt <- function(jobs) {
  jobs <- checkJobs(jobs)
  sequence <- order(jobs$processing / jobs$weight, jobs$due)
  return(evaluateSequence(jobs, sequence))
}

#' Minimise the number of late jobs with Moore-Hodgson
#'
#' Builds an EDD sequence and, whenever a job finishes late, drops the longest
#' job scheduled so far to the end of the sequence.
#'
#' @param jobs A data frame of jobs with columns `job`, `processing`, `due`
#'   and optionally `weight` and `release`.
#' @return A data frame as returned by [evaluateSequence()], with on-time jobs
#'   first and rejected jobs appended.
#' @export
#' @family singleMachine
scheduleMooreHodgson <- function(jobs) {
  jobs <- checkJobs(jobs)
  edd <- order(jobs$due, jobs$processing)
  kept <- integer(0)
  rejected <- integer(0)
  clock <- 0
  for (j in edd) {
    kept <- c(kept, j)
    clock <- clock + jobs$processing[j]
    if (clock > jobs$due[j]) {
      longest <- kept[which.max(jobs$processing[kept])]
      kept <- setdiff(kept, longest)
      rejected <- c(rejected, longest)
      clock <- clock - jobs$processing[longest]
    }
  }
  return(evaluateSequence(jobs, c(kept, rejected)))
}

#' Improve a sequence with adjacent pairwise interchange
#'
#' Repeatedly swaps neighbouring jobs while doing so lowers the chosen
#' objective. Stops at a local optimum or after `maxPasses` passes.
#'
#' @param jobs A data frame of jobs with columns `job`, `processing`, `due`
#'   and optionally `weight` and `release`.
#' @param sequence Integer row indices of `jobs` in processing order.
#' @param objective Objective to minimise.
#' @param maxPasses Maximum number of full sweeps over the sequence.
#' @return A list with the improved `sequence`, its `value` and the number of
#'   `passes` used.
#' @export
#' @family singleMachine
improveByInterchange <- function(jobs, sequence,
                                 objective = c("totalWeightedTardiness",
                                               "maxLateness", "totalCompletion"),
                                 maxPasses = 50L) {
  objective <- match.arg(objective)
  score <- function(seq) {
    ev <- evaluateSequence(jobs, seq)
    switch(objective,
      totalWeightedTardiness = sum(ev$weight * ev$tardiness),
      maxLateness = max(ev$lateness),
      totalCompletion = sum(ev$completion)
    )
  }
  best <- score(sequence)
  passes <- 0L
  repeat {
    passes <- passes + 1L
    improved <- FALSE
    for (k in seq_len(length(sequence) - 1L)) {
      cand <- sequence
      cand[c(k, k + 1L)] <- cand[c(k + 1L, k)]
      val <- score(cand)
      if (val < best - 1e-12) {
        sequence <- cand
        best <- val
        improved <- TRUE
      }
    }
    if (!improved || passes >= maxPasses) {
      break
    }
  }
  return(list(sequence = sequence, value = best, passes = passes))
}

#' Summarise a single-machine schedule
#'
#' @param schedule A data frame as returned by [evaluateSequence()].
#' @return A one-row data frame of standard scheduling metrics.
#' @export
#' @family singleMachine
summariseSchedule <- function(schedule) {
  data.frame(
    nJobs = nrow(schedule),
    makespan = max(schedule$completion),
    totalCompletion = sum(schedule$completion),
    maxLateness = max(schedule$lateness),
    nLate = sum(schedule$tardiness > 0),
    totalTardiness = sum(schedule$tardiness),
    weightedTardiness = sum(schedule$weight * schedule$tardiness)
  )
}
