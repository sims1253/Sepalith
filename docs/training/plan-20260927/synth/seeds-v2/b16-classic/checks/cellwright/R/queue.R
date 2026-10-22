# check.count() lives in R/dice.R; a copy is needed to run this file alone
check.count = function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("'%s' must be a single number", name), call. = FALSE)
  }
  if (x < 1 || x != round(x)) {
    stop(sprintf("'%s' must be a positive whole number", name), call. = FALSE)
  }
  invisible(x)
}
set.seed(3)

# new.queue / queue.length
q = new.queue(2L)
stopifnot(inherits(q, "cw_queue"), queue.length(q) == 0L, is.null(queue.peek(q)))

# queue.push grows the buffer and keeps NULL items
q = queue.push(q, "a")
q = queue.push(q, NULL)
q = queue.push(q, 3)
stopifnot(queue.length(q) == 3L, queue.peek(q) == "a", length(q$items) >= 3)
q0 = queue.push(new.queue(0L), "x")
stopifnot(queue.length(q0) == 1L, queue.peek(q0) == "x")

# queue.pop is first in, first out
p1 = queue.pop(q)
p2 = queue.pop(p1$queue)
p3 = queue.pop(p2$queue)
stopifnot(p1$item == "a", is.null(p2$item), p3$item == 3, queue.length(p3$queue) == 0L)
stopifnot(is.null(queue.pop(p3$queue)$item))

# print.cw_queue
stopifnot(identical(capture.output(print(q)), "<cw_queue> 3 items waiting"))

# run.bank.queue
sim = run.bank.queue(200)
stopifnot(nrow(sim) == 200, all(sim$wait >= 0), all(sim$start >= sim$arrive))
stopifnot(all(diff(sim$start) >= 0), all(sim$start[-1] >= sim$leave[-200] - 1e-9))

# queue.wait.summary
ws = queue.wait.summary(sim, threshold = 1)
stopifnot(identical(names(ws), c("mean", "median", "max", "long.share")))
stopifnot(ws[["max"]] >= ws[["mean"]])
stopifnot(inherits(try(queue.wait.summary(data.frame(x = 1)), silent = TRUE), "try-error"))
