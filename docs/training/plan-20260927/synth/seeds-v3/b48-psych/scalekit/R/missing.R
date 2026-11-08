#' Replace special missing codes with NA
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param refused,dont_know,not_applicable,skipped The numeric code used for
#'   each kind of missing response.
#' @return `data` with all codes set to `NA` and an attribute `codes` counting
#'   the replaced values per kind.
#' @export
recode_missing <- function(data, refused = -9, dont_know = -8,
                           not_applicable = -7, skipped = -6) {
  numeric_cols <- names(data)[vapply(data, is.numeric, logical(1))]
  counts <- c(refused = 0, dont_know = 0, not_applicable = 0, skipped = 0)

  for (col in numeric_cols) {
    x <- data[[col]]

    # Code `refused`
    hit <- !is.na(x) & x == refused
    counts[["refused"]] <- counts[["refused"]] + sum(hit)
    x[hit] <- NA

    # Code `dont_know`
    hit <- !is.na(x) & x == dont_know
    counts[["dont_know"]] <- counts[["dont_know"]] + sum(hit)
    x[hit] <- NA

    # Code `not_applicable`
    hit <- !is.na(x) & x == not_applicable
    counts[["not_applicable"]] <- counts[["not_applicable"]] + sum(hit)
    x[hit] <- NA

    # Code `skipped`
    hit <- !is.na(x) & x == skipped
    counts[["skipped"]] <- counts[["skipped"]] + sum(hit)
    x[hit] <- NA

    data[[col]] <- x
  }

  attr(data, "codes") <- counts
  return(data)
}

#' Impute missing item responses
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items Character vector naming the items to impute.
#' @param method One of `"person_mean"`, `"item_mean"`, `"item_median"`,
#'   `"midpoint"` or `"zero"`.
#' @param min_value,max_value The lowest and highest response option.
#' @return `data` with missing values in `items` filled in and an attribute
#'   `n_imputed`.
#' @export
impute_items <- function(data, items,
                         method = c("person_mean", "item_mean", "item_median",
                                    "midpoint", "zero"),
                         min_value = 1, max_value = 5) {
  method <- match.arg(method)
  responses <- as.matrix(data[items])
  holes <- is.na(responses)

  filled <- switch(method,
    person_mean = {
      person <- rowMeans(responses, na.rm = TRUE)
      person[is.nan(person)] <- NA_real_
      responses[holes] <- person[row(responses)[holes]]
      responses
    },
    item_mean = {
      item <- colMeans(responses, na.rm = TRUE)
      item[is.nan(item)] <- NA_real_
      responses[holes] <- item[col(responses)[holes]]
      responses
    },
    item_median = {
      item <- apply(responses, 2, stats::median, na.rm = TRUE)
      responses[holes] <- item[col(responses)[holes]]
      responses
    },
    midpoint = {
      responses[holes] <- (min_value + max_value) / 2
      responses
    },
    zero = {
      responses[holes] <- 0
      responses
    }
  )

  data[items] <- as.data.frame(filled)
  attr(data, "n_imputed") <- sum(holes & !is.na(filled))
  return(data)
}

#' Flag respondents with too many missing items per subscale
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param depression,anxiety,stress Character vectors naming the items of each
#'   subscale.
#' @param max_missing The largest number of missing items allowed per
#'   subscale.
#' @return A data frame with a `_missing` count and a `_valid` flag per
#'   subscale, plus `all_valid`.
#' @export
flag_missing <- function(data, depression, anxiety, stress, max_missing = 1) {
  if (max_missing < 0) {
    stop("`max_missing` must not be negative.", call. = FALSE)
  }
  out <- data.frame(row.names = seq_len(nrow(data)))

  # Subscale `depression`
  if (!all(depression %in% names(data))) {
    stop(sprintf("Some `%s` items are not columns of `data`.", "depression"), call. = FALSE)
  }
  out$depression_missing <- rowSums(is.na(data[depression]))
  out$depression_valid <- out$depression_missing <= max_missing

  # Subscale `anxiety`
  if (!all(anxiety %in% names(data))) {
    stop(sprintf("Some `%s` items are not columns of `data`.", "anxiety"), call. = FALSE)
  }
  out$anxiety_missing <- rowSums(is.na(data[anxiety]))
  out$anxiety_valid <- out$anxiety_missing <= max_missing

  # Subscale `stress`
  if (!all(stress %in% names(data))) {
    stop(sprintf("Some `%s` items are not columns of `data`.", "stress"), call. = FALSE)
  }
  out$stress_missing <- rowSums(is.na(data[stress]))
  out$stress_valid <- out$stress_missing <= max_missing

  out$all_valid <- out$depression_valid & out$anxiety_valid & out$stress_valid
  return(out)
}

#' Profile missing responses
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items Character vector naming the items to profile.
#' @return A list with elements `n_respondents`, `n_items`, `item_rate`,
#'   `person_rate`, `n_complete` and `worst_item`.
#' @export
missing_profile <- function(data, items = names(data)) {
  if (!all(items %in% names(data))) {
    stop("All `items` must be columns of `data`.", call. = FALSE)
  }
  holes <- is.na(as.matrix(data[items]))

  # 1. Respondents
  n_respondents <- nrow(holes)
  # 2. Items
  n_items <- ncol(holes)
  # 3. Missing share per item
  item_rate <- colMeans(holes)
  # 4. Missing share per respondent
  person_rate <- unname(rowMeans(holes))
  # 5. Respondents without any missing item
  n_complete <- sum(rowSums(holes) == 0)
  # 6. Item with the highest missing share
  worst_item <- if (any(holes)) names(which.max(item_rate)) else NA_character_

  return(list(
    n_respondents = n_respondents,
    n_items = n_items,
    item_rate = item_rate,
    person_rate = person_rate,
    n_complete = n_complete,
    worst_item = worst_item
  ))
}

#' Check whether a subscale can be scored under a missing-data rule
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items Character vector naming the items of the subscale.
#' @param rule One of `"complete"`, `"half"`, `"majority"`, `"count"` or
#'   `"any"`.
#' @param max_missing For `rule = "count"`, the largest number of missing
#'   items allowed.
#' @return A logical vector, one value per respondent.
#' @keywords internal
scorable <- function(data, items, rule = c("complete", "half", "majority", "count", "any"),
                     max_missing = 1) {
  rule <- match.arg(rule)
  n_items <- length(items)
  n_missing <- rowSums(is.na(data[items]))
  n_answered <- n_items - n_missing

  ok <- switch(rule,
    complete = {
      n_missing == 0
    },
    half = {
      n_answered >= n_items / 2
    },
    majority = {
      n_answered > n_items / 2
    },
    count = {
      if (max_missing < 0) {
        stop("`max_missing` must not be negative.", call. = FALSE)
      }
      n_missing <= max_missing
    },
    any = {
      n_answered > 0
    }
  )
  return(unname(ok))
}
