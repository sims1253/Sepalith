#' Reverse-key items by response format
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param agree5,agree7,freq4,binary Character vectors naming the reversed
#'   items on each response format.
#' @return `data` with the named items reversed and an attribute `reversed`
#'   listing the reversed items.
#' @export
apply_keys <- function(data, agree5 = character(), agree7 = character(),
                       freq4 = character(), binary = character()) {
  ranges <- list(agree5 = c(1, 5), agree7 = c(1, 7), freq4 = c(0, 3), binary = c(0, 1))
  reversed <- character()

  # Reverse `agree5` items
  if (length(agree5) > 0) {
    bounds <- ranges[["agree5"]]
    for (item in agree5) {
      if (any(data[[item]] < bounds[1] | data[[item]] > bounds[2], na.rm = TRUE)) {
        stop(sprintf("Item `%s` is outside the `%s` range.", item, "agree5"), call. = FALSE)
      }
      data[[item]] <- sum(bounds) - data[[item]]
    }
    reversed <- c(reversed, agree5)
  }

  # Reverse `agree7` items
  if (length(agree7) > 0) {
    bounds <- ranges[["agree7"]]
    for (item in agree7) {
      if (any(data[[item]] < bounds[1] | data[[item]] > bounds[2], na.rm = TRUE)) {
        stop(sprintf("Item `%s` is outside the `%s` range.", item, "agree7"), call. = FALSE)
      }
      data[[item]] <- sum(bounds) - data[[item]]
    }
    reversed <- c(reversed, agree7)
  }

  # Reverse `freq4` items
  if (length(freq4) > 0) {
    bounds <- ranges[["freq4"]]
    for (item in freq4) {
      if (any(data[[item]] < bounds[1] | data[[item]] > bounds[2], na.rm = TRUE)) {
        stop(sprintf("Item `%s` is outside the `%s` range.", item, "freq4"), call. = FALSE)
      }
      data[[item]] <- sum(bounds) - data[[item]]
    }
    reversed <- c(reversed, freq4)
  }

  # Reverse `binary` items
  if (length(binary) > 0) {
    bounds <- ranges[["binary"]]
    for (item in binary) {
      if (any(data[[item]] < bounds[1] | data[[item]] > bounds[2], na.rm = TRUE)) {
        stop(sprintf("Item `%s` is outside the `%s` range.", item, "binary"), call. = FALSE)
      }
      data[[item]] <- sum(bounds) - data[[item]]
    }
    reversed <- c(reversed, binary)
  }

  attr(data, "reversed") <- reversed
  return(data)
}

#' Reverse a vector of responses
#'
#' @param x A numeric vector of responses.
#' @param method One of `"flip"`, `"negate"`, `"center"` or `"recode"`.
#' @param min_value,max_value The lowest and highest response option.
#' @param map For `"recode"`, a named vector mapping old values (names) to new
#'   values.
#' @return A numeric vector the length of `x`.
#' @keywords internal
reverse_responses <- function(x, method = c("flip", "negate", "center", "recode"),
                              min_value = 1, max_value = 5, map = NULL) {
  method <- match.arg(method)
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }
  midpoint <- (min_value + max_value) / 2

  out <- switch(method,
    flip = {
      if (any(x < min_value | x > max_value, na.rm = TRUE)) {
        stop("`x` must lie within `min_value` and `max_value`.", call. = FALSE)
      }
      min_value + max_value - x
    },
    negate = {
      -x
    },
    center = {
      if (any(x < min_value | x > max_value, na.rm = TRUE)) {
        stop("`x` must lie within `min_value` and `max_value`.", call. = FALSE)
      }
      midpoint - x
    },
    recode = {
      if (is.null(map) || is.null(names(map))) {
        stop("`map` must be a named vector for `method = \"recode\"`.", call. = FALSE)
      }
      unname(map[as.character(x)])
    }
  )
  return(as.numeric(out))
}

#' Build a scoring key
#'
#' @param items Character vector naming all items of the scale.
#' @param plus,minus,skip Character vectors naming the positively keyed,
#'   negatively keyed and unscored items.
#' @return A named integer vector over `items` with `1`, `-1` or `0`.
#' @export
build_key <- function(items, plus = character(), minus = character(), skip = character()) {
  key <- rep(NA_integer_, length(items))
  names(key) <- items

  # Items keyed `plus`
  unknown <- setdiff(plus, items)
  if (length(unknown) > 0) {
    stop(sprintf("`%s` names unknown items: %s", "plus", toString(unknown)), call. = FALSE)
  }
  key[plus] <- 1L

  # Items keyed `minus`
  unknown <- setdiff(minus, items)
  if (length(unknown) > 0) {
    stop(sprintf("`%s` names unknown items: %s", "minus", toString(unknown)), call. = FALSE)
  }
  if (any(!is.na(key[minus]))) {
    stop(sprintf("`%s` overlaps an earlier item set.", "minus"), call. = FALSE)
  }
  key[minus] <- -1L

  # Items keyed `skip`
  unknown <- setdiff(skip, items)
  if (length(unknown) > 0) {
    stop(sprintf("`%s` names unknown items: %s", "skip", toString(unknown)), call. = FALSE)
  }
  if (any(!is.na(key[skip]))) {
    stop(sprintf("`%s` overlaps an earlier item set.", "skip"), call. = FALSE)
  }
  key[skip] <- 0L

  # Every item must be keyed exactly once
  if (anyNA(key)) {
    stop(sprintf("Items without a key: %s", toString(items[is.na(key)])), call. = FALSE)
  }
  return(key)
}

#' Recode responses between response formats
#'
#' @param x A numeric vector of responses.
#' @param scheme One of `"likert5_to_0"`, `"likert5_to_100"`,
#'   `"likert7_to_5"`, `"yesno"` or `"frequency"`.
#' @return A numeric vector the length of `x`.
#' @export
recode_responses <- function(x, scheme = c("likert5_to_0", "likert5_to_100",
                                           "likert7_to_5", "yesno", "frequency")) {
  scheme <- match.arg(scheme)
  check_range <- function(lower, upper) {
    if (any(x < lower | x > upper, na.rm = TRUE)) {
      stop(sprintf("`x` must lie in [%d, %d] for `%s`.", lower, upper, scheme), call. = FALSE)
    }
  }

  out <- switch(scheme,
    likert5_to_0 = {
      check_range(1, 5)
      x - 1
    },
    likert5_to_100 = {
      check_range(1, 5)
      (x - 1) / 4 * 100
    },
    likert7_to_5 = {
      check_range(1, 7)
      1 + (x - 1) / 6 * 4
    },
    yesno = {
      check_range(1, 2)
      as.numeric(x == 1)
    },
    frequency = {
      check_range(0, 3)
      c(0, 1, 3, 7)[x + 1]
    }
  )
  return(out)
}

#' Audit a scoring key against the data
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param key A named integer vector as returned by [build_key()].
#' @return A data frame with columns `item`, `key`, `r_raw`, `r_keyed` and
#'   `suspect`, one row per scored item.
#' @keywords internal
audit_keys <- function(data, key) {
  scored <- names(key)[key != 0L]
  if (length(scored) < 3) {
    stop("At least three scored items are needed.", call. = FALSE)
  }
  responses <- as.matrix(data[scored])
  keyed <- sweep(responses, 2, key[scored], `*`)

  out <- data.frame(item = scored, key = unname(key[scored]))
  out$r_raw <- NA_real_
  out$r_keyed <- NA_real_
  for (i in seq_along(scored)) {
    # Correlate each item with the sum of the other items
    rest_raw <- rowSums(responses[, -i, drop = FALSE])
    rest_keyed <- rowSums(keyed[, -i, drop = FALSE])
    out$r_raw[i] <- stats::cor(responses[, i], rest_raw, use = "complete.obs")
    out$r_keyed[i] <- stats::cor(keyed[, i], rest_keyed, use = "complete.obs")
  }

  out$suspect <- out$r_keyed < 0
  rownames(out) <- NULL
  return(out)
}
