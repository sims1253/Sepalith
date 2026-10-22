#' Create an empty FIFO queue
#'
#' @description
#' The queue is a plain list holding a vector of items and the index of the
#' current head, so pops never copy the whole buffer.
#'
#' @param capacity Initial length of the item buffer.
#' @return A list of class `cw_queue`.
#' @export
new.queue = function(capacity = 16L) {
  structure(
    list(items = vector("list", capacity), head = 1L, tail = 0L),
    class = "cw_queue"
  )
}

#' Number of items waiting in a queue
#'
#' @param q A queue created by [new.queue()].
#' @return A single integer.
#' @export
queue.length = function(q) {
  q$tail - q$head + 1L
}

#' Add an item to the back of a queue
#'
#' @description
#' Doubles the buffer when it is full. `NULL` is a valid item.
#'
#' @param q A queue created by [new.queue()].
#' @param item Any R object to enqueue.
#' @return The updated queue.
#' @export
queue.push = function(q, item) {
  if (q$tail == length(q$items)) {
    q$items = c(q$items, vector("list", max(1L, length(q$items))))
  }
  q$tail = q$tail + 1L
  # Single-bracket assignment: `[[<-` with NULL would delete the slot
  q$items[q$tail] = list(item)
  q
}

#' Look at the front item of a queue
#'
#' @param q A queue created by [new.queue()].
#' @return The front item, or `NULL` when the queue is empty.
#' @export
queue.peek = function(q) {
  if (queue.length(q) == 0L) return(NULL)
  q$items[[q$head]]
}

#' Remove the front item of a queue
#'
#' Because R copies on modify, the caller must keep the returned queue.
#'
#' @param q A queue created by [new.queue()].
#' @return A list with elements `item` (the removed item, or `NULL`) and
#'   `queue` (the updated queue).
#' @export
queue.pop = function(q) {
  if (queue.length(q) == 0L) return(list(item = NULL, queue = q))

  item = q$items[[q$head]]
  # Clear the slot so the popped object can be garbage collected
  q$items[q$head] = list(NULL)
  q$head = q$head + 1L
  list(item = item, queue = q)
}

#' Print a queue
#'
#' @param x A queue created by [new.queue()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.cw_queue = function(x, ...) {
  n = queue.length(x)
  cat(sprintf("<cw_queue> %d item%s waiting\n", n, if (n == 1L) "" else "s"))
  invisible(x)
}

#' Simulate a single-teller bank queue
#'
#' Customers arrive with exponential gaps and are served one at a time with
#' exponential service times. The simulation stops after `n.customers` have
#' been served.
#'
#' @param n.customers Number of customers to serve.
#' @param arrival.rate Mean arrivals per minute.
#' @param service.rate Mean services per minute.
#' @return A data frame with one row per customer: `id`, `arrive`, `start`,
#'   `leave` and `wait`.
#' @export
run.bank.queue = function(n.customers = 100, arrival.rate = 1, service.rate = 1.2) {
  check.count(n.customers, "n.customers")
  arrive = cumsum(rexp(n.customers, arrival.rate))
  service = rexp(n.customers, service.rate)
  start = numeric(n.customers)
  leave = numeric(n.customers)

  q = new.queue()
  clock = 0
  next.in = 1L
  served = 0L
  while (served < n.customers) {
    # admit everyone who has arrived by the time the teller is free
    while (next.in <= n.customers && arrive[next.in] <= clock) {
      q = queue.push(q, next.in)
      next.in = next.in + 1L
    }
    # idle teller: jump the clock to the next arrival
    if (queue.length(q) == 0L) {
      clock = arrive[next.in]
      next
    }

    popped = queue.pop(q)
    q = popped$queue
    id = popped$item
    start[id] = max(clock, arrive[id])
    leave[id] = start[id] + service[id]
    clock = leave[id]
    served = served + 1L
  }

  data.frame(
    id = seq_len(n.customers),
    arrive = arrive,
    start = start,
    leave = leave,
    wait = start - arrive
  )
}

#' Summarise waiting times from a queue simulation
#'
#' @param sim A data frame returned by [run.bank.queue()].
#' @param threshold Waiting time, in minutes, counted as a long wait.
#' @return A named numeric vector with the mean, median and maximum wait and
#'   the share of customers waiting longer than `threshold`.
#' @export
queue.wait.summary = function(sim, threshold = 5) {
  if (!all(c("wait", "leave") %in% names(sim))) {
    stop("'sim' must come from run.bank.queue()", call. = FALSE)
  }

  w = sim$wait
  c(
    mean = mean(w),
    median = median(w),
    max = max(w),
    long.share = mean(w > threshold)
  )
}
