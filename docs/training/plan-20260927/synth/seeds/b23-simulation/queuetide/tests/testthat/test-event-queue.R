test_that("newEvent validates its inputs", {
  expect_error(newEvent("a", "arrival"))
  expect_error(newEvent(1, c("arrival", "departure")))
  ev = newEvent(2.5, "arrival", list(id = 1))
  expect_true(isEvent(ev))
  expect_equal(ev$payload$id, 1)
})

test_that("findInsertPos keeps ties first in, first out", {
  times = c(1, 2, 2, 5)
  expect_equal(findInsertPos(times, 0), 1L)
  expect_equal(findInsertPos(times, 2), 4L)
  expect_equal(findInsertPos(times, 3), 4L)
  expect_equal(findInsertPos(times, 9), 5L)
  expect_equal(findInsertPos(numeric(0), 1), 1L)
})

test_that("EventQueue pops events in time order", {
  queue = eventQueueFromTimes(c(3, 1, 2), type = "arrival")
  expect_equal(queue$size(), 3L)
  popped = numeric(0)
  while (!queue$isEmpty()) {
    popped = c(popped, queue$pop()$time)
  }
  expect_equal(popped, c(1, 2, 3))
  expect_null(queue$pop())
})

test_that("EventQueue peek does not remove the event", {
  queue = EventQueue$new()
  queue$push(newEvent(4, "departure"))
  queue$push(newEvent(1, "arrival"))
  expect_equal(queue$peek()$time, 1)
  expect_equal(queue$size(), 2L)
  expect_equal(queue$pop()$type, "arrival")
  expect_equal(queue$size(), 1L)
})

test_that("EventQueue cancel drops only the matching type", {
  queue = EventQueue$new()
  queue$push(newEvent(1, "arrival"))
  queue$push(newEvent(2, "departure"))
  queue$push(newEvent(3, "departure"))
  expect_equal(queue$cancel("departure"), 2L)
  expect_equal(queue$size(), 1L)
  expect_equal(queue$peek()$type, "arrival")
})

test_that("EventQueue rejects objects that are not events", {
  queue = EventQueue$new()
  expect_error(queue$push(list(time = 1, type = "arrival")), "newEvent")
  expect_true(queue$isEmpty())
})

test_that("runQueueSim serves everyone with ample capacity", {
  set.seed(21)
  arrivals = arrivalsPoisson(rate = 1, horizon = 50)
  sim = runQueueSim(arrivals, servers = 50, sampler = serviceFixed(0.5))
  expect_true(all(sim$customers$wait == 0))
  expect_true(all(abs(sim$customers$sojourn - 0.5) < 1e-12))
  expect_false(any(sim$customers$balked))
})

test_that("runQueueSim records balks when the line is full", {
  arrivals = c(0, 0.1, 0.2, 0.3)
  sim = runQueueSim(arrivals, servers = 1, sampler = serviceFixed(10), queueLimit = 1)
  expect_equal(sim$customers$balked, c(FALSE, FALSE, TRUE, TRUE))
  expect_equal(sim$customers$start[2], 10)
  expect_equal(sim$endTime, 20)
})
