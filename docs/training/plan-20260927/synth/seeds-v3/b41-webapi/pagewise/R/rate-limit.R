#' Read rate-limit headers
#'
#' Services report their limits with different header conventions.
#'
#' @param headers Named character vector of response headers, lower-case names.
#' @param vendor One of `"standard"` (`x-ratelimit-*`, reset as epoch
#'   seconds), `"draft"` (`ratelimit-*`, reset as seconds from now),
#'   `"shopify"` (`x-shopify-shop-api-call-limit` as `used/limit`) or
#'   `"retryafter"` (`retry-after` only).
#' @param now `POSIXct` time the response was received.
#' @return A list with elements `limit`, `remaining` and `resetAt`; unknown
#'   values are `NA`.
#' @export
parseRateHeaders = function(headers, vendor = c("standard", "draft", "shopify", "retryafter"),
                            now = Sys.time()) {
  vendor = match.arg(vendor)
  num = function(name) {
    if (name %in% names(headers)) suppressWarnings(as.numeric(headers[[name]])) else NA_real_
  }

  info = switch(vendor,
    standard = {
      reset = num("x-ratelimit-reset")
      list(
        limit = num("x-ratelimit-limit"),
        remaining = num("x-ratelimit-remaining"),
        resetAt = as.POSIXct(reset, origin = "1970-01-01", tz = "UTC")
      )
    },
    draft = {
      reset = num("ratelimit-reset")
      list(
        limit = num("ratelimit-limit"),
        remaining = num("ratelimit-remaining"),
        resetAt = now + reset
      )
    },
    shopify = {
      value = if ("x-shopify-shop-api-call-limit" %in% names(headers)) {
        as.numeric(strsplit(headers[["x-shopify-shop-api-call-limit"]], "/", fixed = TRUE)[[1]])
      } else {
        c(NA_real_, NA_real_)
      }
      list(
        limit = value[2],
        remaining = value[2] - value[1],
        resetAt = now + 1
      )
    },
    retryafter = {
      reset = num("retry-after")
      list(
        limit = NA_real_,
        remaining = if (is.na(reset)) NA_real_ else 0,
        resetAt = now + reset
      )
    }
  )
  return(info)
}

#' Update the tracked rate-limit state
#'
#' Values missing from `info` keep their previous state.
#'
#' @param state A list with elements `limit`, `remaining`, `resetAt`,
#'   `requests` and `lastSeen`, or `NULL` to start fresh.
#' @param info Parsed headers from [parseRateHeaders()].
#' @param now `POSIXct` time the response was received.
#' @return The updated state list.
#' @export
updateRateState = function(state, info, now = Sys.time()) {
  if (is.null(state)) {
    state = list(limit = NA_real_, remaining = NA_real_, resetAt = as.POSIXct(NA),
      requests = 0L, lastSeen = as.POSIXct(NA))
  }

  # Field `limit`
  if (!is.na(info$limit)) {
    state$limit = info$limit
  }

  # Field `remaining`
  if (!is.na(info$remaining)) {
    state$remaining = info$remaining
  } else if (!is.na(state$remaining)) {
    state$remaining = max(state$remaining - 1, 0)
  }

  # Field `resetAt`
  if (!is.na(info$resetAt)) {
    state$resetAt = info$resetAt
  }

  # Field `requests`
  if (!is.na(state$resetAt) && now >= state$resetAt) {
    state$requests = 1L
  } else {
    state$requests = state$requests + 1L
  }

  # Field `lastSeen`
  state$lastSeen = now
  return(state)
}

#' Seconds to wait before the next request
#'
#' @param state Rate-limit state from [updateRateState()].
#' @param now `POSIXct` current time.
#' @param strategy One of `"block"` (wait for the reset only when nothing
#'   remains), `"spread"` (space the remaining requests evenly until the
#'   reset), `"reserve"` (keep `reserve` requests back) or `"none"`.
#' @param reserve Number of requests to keep back under `"reserve"`.
#' @return A single non-negative number of seconds.
#' @export
rateLimitWait = function(state, now = Sys.time(),
                         strategy = c("block", "spread", "reserve", "none"), reserve = 5) {
  strategy = match.arg(strategy)
  untilReset = if (is.na(state$resetAt)) {
    NA_real_
  } else {
    max(as.numeric(difftime(state$resetAt, now, units = "secs")), 0)
  }
  if (is.na(untilReset) || is.na(state$remaining)) {
    return(0)
  }

  wait = switch(strategy,
    block = {
      if (state$remaining <= 0) untilReset else 0
    },
    spread = {
      if (state$remaining <= 0) untilReset else untilReset / state$remaining
    },
    reserve = {
      if (state$remaining <= reserve) untilReset else 0
    },
    none = {
      0
    }
  )
  return(wait)
}

#' Advance a token bucket
#'
#' Refills the bucket for the time elapsed since the last call, then takes
#' `cost` tokens if enough are available.
#'
#' @param bucket A list with elements `tokens`, `capacity`, `rate` (tokens
#'   per second) and `updatedAt`.
#' @param now `POSIXct` current time.
#' @param cost Number of tokens the request needs.
#' @return A list with elements `bucket` (the updated bucket), `allowed`
#'   and `wait` (seconds until `cost` tokens are available).
#' @export
takeTokens = function(bucket, now = Sys.time(), cost = 1) {
  if (cost > bucket$capacity) {
    stop("`cost` is larger than the bucket capacity.", call. = FALSE)
  }

  # Refill: add tokens for the elapsed time, up to the capacity
  elapsed = as.numeric(difftime(now, bucket$updatedAt, units = "secs"))
  if (elapsed > 0) {
    bucket$tokens = min(bucket$capacity, bucket$tokens + elapsed * bucket$rate)
    bucket$updatedAt = now
  }

  # Consume: take the tokens, or report how long until there are enough
  if (bucket$tokens >= cost) {
    bucket$tokens = bucket$tokens - cost
    allowed = TRUE
    wait = 0
  } else {
    allowed = FALSE
    wait = (cost - bucket$tokens) / bucket$rate
  }

  return(list(bucket = bucket, allowed = allowed, wait = wait))
}

#' Summarise rate-limit states across endpoints
#'
#' @param states A named list of rate-limit states, one per endpoint, from
#'   [updateRateState()].
#' @param now `POSIXct` current time.
#' @return A data frame with columns `endpoint`, `limit`, `remaining`,
#'   `usedPct`, `resetIn` and `requests`, ordered by `usedPct` descending.
#' @export
rateLimitReport = function(states, now = Sys.time()) {
  endpoint = names(states)
  n = length(states)
  limit = numeric(n)
  remaining = numeric(n)
  usedPct = numeric(n)
  resetIn = numeric(n)
  requests = integer(n)
  for (i in seq_len(n)) {
    s = states[[i]]
    # Column `limit`
    limit[i] = s$limit
    # Column `remaining`
    remaining[i] = s$remaining
    # Column `usedPct`
    usedPct[i] = if (is.na(s$limit) || s$limit == 0) NA_real_ else
      round(100 * (s$limit - s$remaining) / s$limit, 1)
    # Column `resetIn`
    resetIn[i] = if (is.na(s$resetAt)) NA_real_ else
      max(as.numeric(difftime(s$resetAt, now, units = "secs")), 0)
    # Column `requests`
    requests[i] = s$requests
  }

  out = data.frame(endpoint = endpoint, limit = limit, remaining = remaining,
    usedPct = usedPct, resetIn = resetIn, requests = requests)
  out = out[order(-out$usedPct, na.last = TRUE), , drop = FALSE]
  rownames(out) = NULL
  return(out)
}
