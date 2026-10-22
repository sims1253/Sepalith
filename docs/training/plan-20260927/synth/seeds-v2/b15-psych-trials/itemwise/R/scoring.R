#' Reverse-key a vector of item responses
#'
#' Maps each response onto the opposite end of the response scale, so that
#' `min` becomes `max` and vice versa.
#'
#' @param x A numeric vector of item responses.
#' @param min The lowest admissible response category.
#' @param max The highest admissible response category.
#' @return A numeric vector the same length as `x`.
#' @export
#' @examples
#' reverse_key(c(1, 2, 5), min = 1, max = 5)
reverse_key <- function(x, min, max) {
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }
  return(max + min - x)
}

#' Reverse-key selected columns of a response data frame
#'
#' Applies [reverse_key()] to every column named in `reverse`, leaving the
#' remaining columns untouched.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param reverse A character vector of item names to reverse-key.
#' @param min The lowest admissible response category.
#' @param max The highest admissible response category.
#' @return `data` with the reverse-keyed columns replaced.
#' @export
reverse_key_items <- function(data, reverse, min, max) {
  missing_cols <- setdiff(reverse, names(data))
  if (length(missing_cols) > 0) {
    stop("Unknown items: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  for (item in reverse) {
    data[[item]] <- reverse_key(data[[item]], min = min, max = max)
  }
  return(data)
}

#' Check that item responses fall inside the response scale
#'
#' Flags any non-missing response outside `[min, max]` or not a whole number.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items A character vector of item names making up the scale.
#' @param min The lowest admissible response category.
#' @param max The highest admissible response category.
#' @noRd
check_item_range <- function(data, items, min, max) {
  values <- unlist(data[items], use.names = FALSE)
  values <- values[!is.na(values)]

  bad <- values < min | values > max | values != round(values)
  if (any(bad)) {
    stop(
      sum(bad), " response(s) outside the scale ", min, "-", max, ".",
      call. = FALSE
    )
  }
  return(invisible(data))
}

#' Check per-respondent missingness against a threshold
#'
#' Flags respondents who skipped more than `max_missing` of the scale items.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items A character vector of item names making up the scale.
#' @param max_missing Maximum number of missing items tolerated per respondent.
#' @noRd
check_item_missing <- function(data, items, max_missing) {
  n_missing <- rowSums(is.na(data[items]))
  too_many <- n_missing > max_missing
  # An empty data frame is not "everyone failed", so only warn with rows
  if (length(too_many) > 0 && all(too_many)) {
    warning("Every respondent exceeds `max_missing`.", call. = FALSE)
  }
  return(too_many)
}

#' Compute a sum score for a scale
#'
#' Reverse-keys the requested items, then sums the responses for each
#' respondent. Respondents with too many missing items get `NA`.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items A character vector of item names making up the scale.
#' @param reverse A character vector of item names to reverse-key.
#' @param min The lowest admissible response category.
#' @param max The highest admissible response category.
#' @param max_missing Maximum number of missing items tolerated per respondent.
#' @return A numeric vector of scale scores, one per row of `data`.
#' @export
score_sum <- function(data, items, reverse = character(), min = 1, max = 5,
                      max_missing = 0) {
  check_item_range(data, items, min, max)
  data <- reverse_key_items(data, intersect(reverse, items), min, max)

  # A respondent with no answers at all must not score 0, even when
  # `max_missing` is generous enough to let them through.
  too_many <- check_item_missing(data, items, max_missing)
  none <- rowSums(!is.na(data[items])) == 0

  scores <- rowSums(data[items], na.rm = TRUE)
  scores[too_many | none] <- NA_real_
  return(scores)
}

#' Compute a mean score for a scale
#'
#' Reverse-keys the requested items, then averages the available responses
#' for each respondent. Respondents with too many missing items get `NA`.
#'
#' @inheritParams score_sum
#' @return A numeric vector of scale scores, one per row of `data`.
#' @export
score_mean <- function(data, items, reverse = character(), min = 1, max = 5,
                       max_missing = 0) {
  check_item_range(data, items, min, max)
  data <- reverse_key_items(data, intersect(reverse, items), min, max)

  too_many <- check_item_missing(data, items, max_missing)
  none <- rowSums(!is.na(data[items])) == 0

  scores <- rowMeans(data[items], na.rm = TRUE)
  scores[too_many | none] <- NA_real_
  return(scores)
}

#' Compute a prorated sum score for a scale
#'
#' Averages the available responses and rescales the result to the full
#' number of items, which keeps scores comparable when a few items are skipped.
#'
#' @inheritParams score_sum
#' @return A numeric vector of scale scores, one per row of `data`.
#' @export
score_prorated <- function(data, items, reverse = character(), min = 1, max = 5,
                           max_missing = 1) {
  scores <- score_mean(
    data, items,
    reverse = reverse, min = min, max = max, max_missing = max_missing
  )
  return(scores * length(items))
}

#' Score several scales from a scoring key
#'
#' Loops over a scoring key and computes one score column per scale, using
#' the chosen scoring rule throughout.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param key A named list; each element is a list with `items` and,
#'   optionally, `reverse`.
#' @param method Scoring rule, one of `"sum"`, `"mean"` or `"prorated"`.
#' @param ... Passed to [score_sum()], [score_mean()] or [score_prorated()].
#' @return A data frame with one column per scale in `key`.
#' @export
#' @examples
#' key <- list(anx = list(items = c("a1", "a2", "a3"), reverse = "a2"))
#' resp <- data.frame(a1 = c(1, 4), a2 = c(5, 2), a3 = c(2, 3))
#' score_scales(resp, key, method = "mean")
score_scales <- function(data, key, method = c("sum", "mean", "prorated"), ...) {
  method <- match.arg(method)
  if (is.null(names(key)) || any(names(key) == "")) {
    stop("`key` must be a fully named list.", call. = FALSE)
  }

  scorer <- switch(method,
    sum = score_sum,
    mean = score_mean,
    prorated = score_prorated
  )

  out <- vector("list", length(key))
  names(out) <- names(key)
  for (scale in names(key)) {
    spec <- key[[scale]]
    reverse <- if (is.null(spec$reverse)) character() else spec$reverse
    out[[scale]] <- scorer(data, spec$items, reverse = reverse, ...)
  }
  return(as.data.frame(out, check.names = FALSE))
}
