make_job <- function(name, priority = 5L, value = name) {
  Job$new(function() value, name = name, priority = priority)
}

test_that("push and pop serve the highest priority first", {
  q <- PriorityQueue$new()
  q$push(make_job("low", 2L))$push(make_job("high", 9L))$push(make_job("mid", 5L))
  expect_equal(q$size, 3L)
  expect_equal(q$pop()$name, "high")
  expect_equal(q$pop()$name, "mid")
  expect_equal(q$pop()$name, "low")
  expect_null(q$pop())
})

test_that("jobs with equal priority keep insertion order", {
  q <- PriorityQueue$new()
  for (nm in c("a", "b", "c")) {
    q$push(make_job(nm, 4L))
  }
  served <- vapply(seq_len(3), function(i) q$pop()$name, character(1))
  expect_equal(served, c("a", "b", "c"))
})

test_that("capacity is enforced", {
  q <- PriorityQueue$new(capacity = 2)
  q$push(make_job("one"))$push(make_job("two"))
  expect_error(q$push(make_job("three")), "Queue is full")
  expect_equal(q$size, 2L)
})

test_that("active bindings are read-only", {
  q <- PriorityQueue$new()
  expect_error(q$size <- 10L, "read-only")
  expect_error(q$is_empty <- FALSE, "read-only")
  expect_error(q$jobs <- list(), "read-only")
})

test_that("remove drops a job by id and warns on unknown ids", {
  q <- PriorityQueue$new()
  keep <- make_job("keep")
  drop <- make_job("drop")
  q$push(keep)$push(drop)
  q$remove(drop$id)
  expect_equal(q$size, 1L)
  expect_equal(q$peek()$name, "keep")
  expect_warning(q$remove("no-such-id"), "No job with id")
})

test_that("queue_from_list accepts functions, formulas and jobs", {
  q <- queue_from_list(list(
    first = function() 1,
    second = ~ 2 + 2,
    make_job("third", 8L)
  ))
  expect_equal(q$size, 3L)
  expect_equal(q$peek()$name, "third")
  expect_setequal(queue_summary(q)$name, c("first", "second", "third"))
})

test_that("queue_drain runs every job and records failures", {
  q <- PriorityQueue$new()
  q$push(make_job("ok", 6L))
  q$push(Job$new(function() stop("boom"), name = "bad", priority = 3L))
  done <- queue_drain(q)
  expect_length(done, 2L)
  expect_true(q$is_empty)
  expect_equal(vapply(done, function(j) j$status, character(1)), c("succeeded", "failed"))
  expect_error(done[[2]]$result(), "boom")
})

test_that("scheduler retries failed jobs up to max_attempts", {
  calls <- 0L
  flaky <- function() {
    calls <<- calls + 1L
    if (calls < 3L) stop("not yet")
    "done"
  }
  sched <- Scheduler$new(
    max_attempts = 3L,
    backoff = backoff_constant(0),
    sleep = function(s) NULL
  )
  sched$submit(flaky, name = "flaky")
  sched$run_all()
  report <- scheduler_report(sched)
  expect_equal(report$attempts, 3L)
  expect_equal(report$status, "succeeded")
  expect_equal(calls, 3L)
})
