#' Service time samplers
#'
#' Each constructor returns a function that draws service times.
#' `serviceExponential()` draws from an exponential distribution.
#'
#' @param mean Mean service time.
#' @return A function of one argument `n` returning `n` service times.
#' @export
serviceExponential = function(mean) {
  stopifnot(mean > 0)
  function(n) stats::rexp(n, rate = 1 / mean)
}

#' @description `serviceLognormal()` draws from a lognormal distribution
#'   parameterised by its mean and coefficient of variation.
#' @param cv Coefficient of variation of the service time.
#' @rdname serviceExponential
#' @export
serviceLognormal = function(mean, cv = 0.5) {
  stopifnot(mean > 0, cv > 0)

  # Match the first two moments on the log scale.
  sdlog = sqrt(log(1 + cv^2))
  meanlog = log(mean) - sdlog^2 / 2
  function(n) stats::rlnorm(n, meanlog = meanlog, sdlog = sdlog)
}

#' @description `serviceFixed()` always yields the same service time.
#' @rdname serviceExponential
#' @export
serviceFixed = function(mean) {
  stopifnot(mean > 0)
  function(n) rep(mean, n)
}

#' Build a service time sampler by name
#'
#' @param dist Distribution name: `"exponential"`, `"lognormal"` or
#'   `"fixed"`.
#' @param mean Mean service time.
#' @param ... Passed to the matching sampler constructor.
#' @return A function of one argument `n` returning `n` service times.
#' @export
makeServiceSampler = function(dist = c("exponential", "lognormal", "fixed"), mean, ...) {
  dist = match.arg(dist)
  switch(dist,
    exponential = serviceExponential(mean),
    lognormal = serviceLognormal(mean, ...),
    fixed = serviceFixed(mean)
  )
}

#' Pool of identical servers with a shared waiting line
#'
#' Tracks which servers are busy, the customers waiting in line and the
#' accumulated busy time needed to compute utilisation.
#'
#' @export
ServerPool = R6::R6Class("ServerPool",
  public = list(
    #' @field capacity Number of parallel servers.
    capacity = NULL,
    #' @field queueLimit Maximum number of waiting customers, `Inf` for none.
    queueLimit = NULL,
    #' @description Create a server pool.
    #' @param capacity Number of parallel servers.
    #' @param queueLimit Maximum number of waiting customers.
    #' @return A new `ServerPool` object.
    initialize = function(capacity = 1L, queueLimit = Inf) {
      stopifnot(capacity >= 1, queueLimit >= 0)
      self$capacity = as.integer(capacity)
      self$queueLimit = queueLimit
      private$busyWith = rep(NA_integer_, self$capacity)
      private$busySince = rep(NA_real_, self$capacity)
      private$busyTotal = numeric(self$capacity)
      private$line = list()
      invisible(self)
    },

    #' @description Index of a free server, or `NA` if all are busy.
    #' @return An integer.
    freeServer = function() {
      idle = which(is.na(private$busyWith))
      if (length(idle) == 0) NA_integer_ else idle[[1]]
    },

    #' @description Start serving a customer on a given server.
    #' @param server Index of the server.
    #' @param customer Customer id.
    #' @param now Current simulation time.
    #' @return The pool, invisibly.
    startService = function(server, customer, now) {
      if (!is.na(private$busyWith[server])) {
        stop("Server ", server, " is already busy.", call. = FALSE)
      }
      private$busyWith[server] = customer
      private$busySince[server] = now
      invisible(self)
    },

    #' @description Finish service on a given server and record busy time.
    #' @param server Index of the server.
    #' @param now Current simulation time.
    #' @return The id of the customer that was being served.
    endService = function(server, now) {
      customer = private$busyWith[server]
      if (is.na(customer)) {
        stop("Server ", server, " is idle.", call. = FALSE)
      }

      private$busyTotal[server] = private$busyTotal[server] + (now - private$busySince[server])
      private$busyWith[server] = NA_integer_
      private$busySince[server] = NA_real_
      customer
    },

    #' @description Add a customer to the waiting line.
    #' @param customer Customer id.
    #' @param now Current simulation time.
    #' @return `TRUE` if the customer joined, `FALSE` if the line was full.
    enqueue = function(customer, now) {
      if (length(private$line) >= self$queueLimit) {
        return(FALSE)
      }
      private$line[[length(private$line) + 1L]] = list(id = customer, since = now)
      TRUE
    },

    #' @description Remove the customer at the head of the waiting line.
    #' @return A list with `id` and `since`, or `NULL` if the line is empty.
    dequeue = function() {
      if (length(private$line) == 0) {
        return(NULL)
      }
      head = private$line[[1]]
      private$line = private$line[-1]
      head
    },

    #' @description Current number of waiting customers.
    #' @return An integer.
    lineLength = function() {
      length(private$line)
    },

    #' @description Per-server utilisation up to a given time.
    #' @param now Current simulation time.
    #' @return A numeric vector of busy fractions, one per server.
    utilisation = function(now) {
      if (now <= 0) {
        return(rep(0, self$capacity))
      }
      # Count the time spent on services that are still running.
      openBusy = ifelse(is.na(private$busySince), 0, now - private$busySince)
      (private$busyTotal + openBusy) / now
    }
  ),
  private = list(
    busyWith = NULL,
    busySince = NULL,
    busyTotal = NULL,
    line = NULL
  )
)

#' Theoretical M/M/c performance measures
#'
#' Uses the Erlang C formula to compute the probability of waiting, the
#' mean waiting time and the server utilisation of a stable M/M/c queue.
#' Handy as a reference for validating simulation output.
#'
#' @param lambda Arrival rate.
#' @param mu Service rate of a single server.
#' @param servers Number of parallel servers.
#' @return A list with elements `rho`, `pWait`, `meanWait` and
#'   `meanQueue`.
#' @export
erlangC = function(lambda, mu, servers = 1L) {
  stopifnot(lambda > 0, mu > 0, servers >= 1)

  # Offered load in Erlangs and per-server utilisation.
  a = lambda / mu
  rho = a / servers
  if (rho >= 1) {
    stop("Queue is unstable: utilisation must be below 1.", call. = FALSE)
  }

  k = 0:(servers - 1)
  idleTerms = sum(a^k / factorial(k))
  busyTerm = a^servers / (factorial(servers) * (1 - rho))
  pWait = busyTerm / (idleTerms + busyTerm)
  meanWait = pWait / (servers * mu - lambda)

  # Little's law gives the mean queue length from the mean wait.
  list(
    rho = rho,
    pWait = pWait,
    meanWait = meanWait,
    meanQueue = lambda * meanWait
  )
}
