#' Classify failed requests
#'
#' Maps HTTP status codes and transport error messages to an error class.
#'
#' @param status Integer HTTP status codes; `NA` when no response arrived.
#' @param message Character vector of transport error messages, recycled to
#'   the length of `status`.
#' @return A character vector with values `"timeout"`, `"rate_limit"`,
#'   `"server"`, `"network"`, `"auth"`, `"client"` or `"ok"`.
#' @export
classify_http_error <- function(status, message = "") {
  message <- rep_len(tolower(message), length(status))
  out <- rep("client", length(status))
  # Class `ok`: 1xx to 3xx responses
  is_ok <- !is.na(status) & status < 400L
  out[is_ok] <- "ok"
  # Class `auth`: missing or rejected credentials
  is_auth <- !is.na(status) & status %in% c(401L, 403L)
  out[is_auth] <- "auth"
  # Class `rate_limit`: too many requests
  is_rate_limit <- !is.na(status) & status == 429L
  out[is_rate_limit] <- "rate_limit"
  # Class `server`: 5xx except gateway timeouts
  is_server <- !is.na(status) & status >= 500L & status != 504L
  out[is_server] <- "server"
  # Class `timeout`: gateway timeouts and timed-out transfers
  is_timeout <- (!is.na(status) & status %in% c(408L, 504L)) |
    (is.na(status) & grepl("timed? ?out", message))
  out[is_timeout] <- "timeout"
  # Class `network`: no response for any other reason
  is_network <- is.na(status) & !is_timeout
  out[is_network] <- "network"
  return(out)
}

#' Delay before the next attempt
#'
#' @param attempt Number of attempts made so far (1 after the first failure).
#' @param strategy One of `"constant"`, `"linear"`, `"exponential"`,
#'   `"jitter"` or `"fibonacci"`.
#' @param base Base delay in seconds.
#' @param cap Maximum delay in seconds.
#' @return The delay in seconds, at most `cap`.
#' @export
backoff_delay <- function(attempt,
                          strategy = c("constant", "linear", "exponential", "jitter",
                                       "fibonacci"),
                          base = 1, cap = 60) {
  strategy <- match.arg(strategy)
  if (!is.numeric(attempt) || length(attempt) != 1L || attempt < 1) {
    stop("`attempt` must be a single number of at least 1.", call. = FALSE)
  }

  delay <- switch(strategy,
    constant = {
      base
    },
    linear = {
      base * attempt
    },
    exponential = {
      base * 2^(attempt - 1)
    },
    jitter = {
      # Full jitter: uniform between zero and the exponential delay
      stats::runif(1, 0, base * 2^(attempt - 1))
    },
    fibonacci = {
      fib <- c(1, 1)
      while (length(fib) < attempt) {
        fib <- c(fib, sum(utils::tail(fib, 2)))
      }
      base * fib[[attempt]]
    }
  )
  return(min(delay, cap))
}

#' Default retry policy per error class
#'
#' @param error_class One of `"timeout"`, `"rate_limit"`, `"server"`,
#'   `"network"` or `"auth"`, as returned by [classify_http_error()].
#' @return A list with elements `retry`, `max_attempts`, `strategy`, `base`
#'   and `respect_retry_after`.
#' @export
retry_policy <- function(error_class = c("timeout", "rate_limit", "server", "network", "auth")) {
  error_class <- match.arg(error_class)
  policy <- switch(error_class,
    timeout = {
      list(
        retry = TRUE, max_attempts = 3L, strategy = "exponential", base = 2,
        respect_retry_after = FALSE
      )
    },
    rate_limit = {
      list(
        retry = TRUE, max_attempts = 5L, strategy = "exponential", base = 5,
        respect_retry_after = TRUE
      )
    },
    server = {
      list(
        retry = TRUE, max_attempts = 4L, strategy = "jitter", base = 1,
        respect_retry_after = TRUE
      )
    },
    network = {
      list(
        retry = TRUE, max_attempts = 3L, strategy = "linear", base = 1,
        respect_retry_after = FALSE
      )
    },
    auth = {
      list(
        retry = FALSE, max_attempts = 1L, strategy = "constant", base = 0,
        respect_retry_after = FALSE
      )
    }
  )
  return(policy)
}

#' Signal a classed HTTP error
#'
#' @param status HTTP status code, or `NA` when no response arrived.
#' @param message Error message.
#' @param retry_after Seconds from a `Retry-After` header, or `NA`.
#' @return Does not return; signals a condition with classes
#'   `quaryl_<error class>`, `quaryl_http_error`, `error` and `condition`.
#' @keywords internal
http_condition <- function(status, message = "", retry_after = NA_real_) {
  error_class <- classify_http_error(status, message)
  if (identical(error_class, "ok")) {
    stop("`status` does not describe an error.", call. = FALSE)
  }
  text <- if (is.na(status)) message else sprintf("HTTP %d: %s", status, message)
  cond <- structure(
    class = c(paste0("quaryl_", error_class), "quaryl_http_error", "error", "condition"),
    list(
      message = text,
      call = NULL,
      status = status,
      error_class = error_class,
      retry_after = retry_after
    )
  )
  stop(cond)
}

#' Call a function with retries
#'
#' Retries on the error classes signalled by [http_condition()], following
#' the [retry_policy()] for each class.
#'
#' @param fun A function with no arguments that performs the request.
#' @param sleep Function used to wait; replace it in tests.
#' @param verbose If `TRUE`, report each retry.
#' @return A list with elements `value` (the result of `fun`) and `log` (a
#'   data frame with columns `attempt`, `error_class` and `delay`).
#' @export
with_retry <- function(fun, sleep = Sys.sleep, verbose = FALSE) {
  log <- data.frame(attempt = integer(0), error_class = character(0), delay = numeric(0))
  attempt <- 0L
  repeat {
    attempt <- attempt + 1L
    result <- tryCatch(
      list(ok = TRUE, value = fun()),
      quaryl_timeout = function(e) list(ok = FALSE, error = e),
      quaryl_rate_limit = function(e) list(ok = FALSE, error = e),
      quaryl_server = function(e) list(ok = FALSE, error = e),
      quaryl_network = function(e) list(ok = FALSE, error = e)
    )
    if (result$ok) {
      return(list(value = result$value, log = log))
    }
    err <- result$error
    policy <- retry_policy(err$error_class)
    if (!policy$retry || attempt >= policy$max_attempts) {
      stop(err)
    }
    delay <- backoff_delay(attempt, policy$strategy, base = policy$base)
    if (policy$respect_retry_after && !is.na(err$retry_after)) {
      delay <- max(delay, err$retry_after)
    }
    log[nrow(log) + 1L, ] <- list(attempt, err$error_class, delay)
    if (verbose) {
      message(sprintf("Attempt %d failed (%s); retrying in %.1fs.", attempt,
        err$error_class, delay))
    }
    sleep(delay)
  }
}

#' Summarise a retry log
#'
#' @param log A data frame as returned in `with_retry()$log`.
#' @return A data frame with one row per error class (`timeout`,
#'   `rate_limit`, `server`, `network`) and columns `error_class`, `n` and
#'   `total_delay`.
#' @export
retry_summary <- function(log) {
  classes <- c("timeout", "rate_limit", "server", "network")
  n <- integer(length(classes))
  total_delay <- numeric(length(classes))
  names(n) <- names(total_delay) <- classes
  # Class `timeout`
  rows <- log$error_class == "timeout"
  n[["timeout"]] <- sum(rows)
  total_delay[["timeout"]] <- sum(log$delay[rows])
  # Class `rate_limit`
  rows <- log$error_class == "rate_limit"
  n[["rate_limit"]] <- sum(rows)
  total_delay[["rate_limit"]] <- sum(log$delay[rows])
  # Class `server`
  rows <- log$error_class == "server"
  n[["server"]] <- sum(rows)
  total_delay[["server"]] <- sum(log$delay[rows])
  # Class `network`
  rows <- log$error_class == "network"
  n[["network"]] <- sum(rows)
  total_delay[["network"]] <- sum(log$delay[rows])
  return(data.frame(
    error_class = classes,
    n = unname(n),
    total_delay = unname(total_delay)
  ))
}
