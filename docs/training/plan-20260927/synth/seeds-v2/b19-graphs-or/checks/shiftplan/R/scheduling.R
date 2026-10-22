jobs <- data.frame(
  job = c("A", "B", "C", "D"),
  processing = c(4, 2, 6, 3),
  due = c(5, 4, 15, 7),
  weight = c(1, 3, 1, 2)
)

# checkJobs
filled <- checkJobs(jobs[, c("job", "processing", "due")])
stopifnot(all(filled$weight == 1), all(filled$release == 0))
bad <- jobs
bad$processing[2] <- NA
stopifnot(inherits(try(checkJobs(bad), silent = TRUE), "try-error"))

# evaluateSequence honours release dates
rel <- jobs
rel$release <- c(0, 10, 0, 0)
ev <- evaluateSequence(rel, c(1, 2))
stopifnot(identical(ev$start, c(0, 10)), identical(ev$completion, c(4, 12)))
stopifnot(identical(ev$tardiness, c(0, 8)))

# dispatching rules
stopifnot(identical(scheduleEdd(jobs)$job, c("B", "A", "D", "C")))
stopifnot(identical(scheduleSpt(jobs)$job, c("B", "D", "A", "C")))
stopifnot(identical(scheduleWspt(jobs)$job, c("B", "D", "A", "C")))

# scheduleMooreHodgson minimises the number of late jobs
mh <- scheduleMooreHodgson(jobs)
stopifnot(sum(mh$tardiness > 0) == 1, mh$job[4] == "A")

# improveByInterchange never worsens the start sequence
start <- c(3, 1, 2, 4)
imp <- improveByInterchange(jobs, start)
before <- evaluateSequence(jobs, start)
stopifnot(imp$value <= sum(before$weight * before$tardiness), imp$passes >= 1L)
stopifnot(setequal(imp$sequence, 1:4))

# summariseSchedule
sm <- summariseSchedule(scheduleEdd(jobs))
stopifnot(sm$nJobs == 4, sm$makespan == 15, sm$nLate == 2)
