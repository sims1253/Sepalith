#' Score a scale from its items
#'
#' `score_mean()` averages the answered items, `score_sum()` adds them up
#' and `score_prorated()` imputes missing items by the respondent's own item
#' mean, so the score stays on the scale of a complete sum. Respondents who
#' answered fewer than `min_items` of the items receive `NA` instead of a
#' score based on too little information.
#'
#' @param data A data frame of item-level survey responses.
#' @param items <[`tidy-select`][dplyr::dplyr_tidy_select]> Items belonging to the scale.
#' @param name Name of the new score column.
#' @param min_items Minimum number of answered items required for a score.
#' @return `data` with an added score column.
#' @export
score_mean <- function(data, items, name, min_items = 1L) {
  item_data <- dplyr::select(data, {{ items }})
  answered <- rowSums(!is.na(item_data))
  score <- rowMeans(item_data, na.rm = TRUE)
  score[answered < min_items] <- NA_real_
  data[[name]] <- score
  return(data)
}

#' @rdname score_mean
#' @export
score_sum <- function(data, items, name, min_items = 1L) {
  item_data <- dplyr::select(data, {{ items }})
  answered <- rowSums(!is.na(item_data))
  score <- rowSums(item_data, na.rm = TRUE)
  score[answered < min_items] <- NA_real_
  data[[name]] <- score
  return(data)
}

#' @rdname score_mean
#' @export
score_prorated <- function(data, items, name, min_items = 1L) {
  item_data <- dplyr::select(data, {{ items }})
  answered <- rowSums(!is.na(item_data))
  score <- rowMeans(item_data, na.rm = TRUE) * ncol(item_data)
  score[answered < min_items] <- NA_real_
  data[[name]] <- score
  return(data)
}

#' Score a scale with a chosen method
#'
#' A thin dispatcher over the individual scorers, convenient when the
#' scoring rule comes from a codebook.
#'
#' @param data A data frame of item-level survey responses.
#' @param items <[`tidy-select`][dplyr::dplyr_tidy_select]> Items belonging to the scale.
#' @param name Name of the new score column.
#' @param method One of `"mean"`, `"sum"` or `"prorated"`.
#' @param min_items Minimum number of answered items required for a score.
#' @return `data` with an added score column.
#' @export
score_scale <- function(data, items, name, method = c("mean", "sum", "prorated"),
                        min_items = 1L) {
  method <- match.arg(method)
  scored <- switch(method,
    mean = score_mean(data, {{ items }}, name, min_items),
    sum = score_sum(data, {{ items }}, name, min_items),
    prorated = score_prorated(data, {{ items }}, name, min_items)
  )
  return(scored)
}

#' Score every scale listed in a codebook
#'
#' The codebook must have columns `scale`, `item` and `method`, with one
#' row per item. Scales are scored in the order they first appear.
#'
#' @param data A data frame of item-level survey responses.
#' @param codebook A data frame describing which items belong to which scale.
#' @param min_items Minimum number of answered items required for a score.
#' @return `data` with one added column per scale.
#' @export
score_codebook <- function(data, codebook, min_items = 1L) {
  required <- c("scale", "item", "method")
  missing <- setdiff(required, names(codebook))
  if (length(missing) > 0) {
    stop("Codebook is missing columns: ", paste(missing, collapse = ", "),
      call. = FALSE)
  }

  # summarise() sorts the groups, so restore codebook order afterwards.
  scales <- codebook %>%
    dplyr::group_by(.data$scale) %>%
    dplyr::summarise(
      items = list(.data$item),
      method = dplyr::first(.data$method),
      .groups = "drop"
    )
  scales <- scales[match(unique(codebook$scale), scales$scale), ]

  for (i in seq_len(nrow(scales))) {
    data <- score_scale(
      data,
      items = dplyr::all_of(scales$items[[i]]),
      name = scales$scale[[i]],
      method = scales$method[[i]],
      min_items = min_items
    )
  }
  return(data)
}

#' Estimate Cronbach's alpha for a set of items
#'
#' Uses complete cases only. Returns `NA` when fewer than two items or
#' fewer than three complete respondents are available.
#'
#' @param data A data frame of item-level survey responses.
#' @param items <[`tidy-select`][dplyr::dplyr_tidy_select]> Items belonging to the scale.
#' @return A single numeric value.
#' @export
scale_alpha <- function(data, items) {
  item_data <- data %>%
    dplyr::select({{ items }}) %>%
    tidyr::drop_na()

  k <- ncol(item_data)
  if (k < 2 || nrow(item_data) < 3) {
    return(NA_real_)
  }

  item_vars <- vapply(item_data, stats::var, numeric(1))
  total_var <- stats::var(rowSums(item_data))
  alpha <- (k / (k - 1)) * (1 - sum(item_vars) / total_var)
  return(alpha)
}

#' Compute item-level statistics for a scale
#'
#' Reports, for each item, the mean, standard deviation, missing share,
#' corrected item-total correlation and the alpha obtained when the item
#' is dropped. Useful for spotting items that should be reverse-scored.
#'
#' @param data A data frame of item-level survey responses.
#' @param items <[`tidy-select`][dplyr::dplyr_tidy_select]> Items belonging to the scale.
#' @return A tibble with one row per item.
#' @export
item_statistics <- function(data, items) {
  item_data <- dplyr::select(data, {{ items }})
  k <- ncol(item_data)
  if (k < 3) {
    stop("Item statistics need at least three items.", call. = FALSE)
  }

  complete <- tidyr::drop_na(item_data)
  item_names <- names(item_data)
  n <- length(item_names)
  stats <- tibble::tibble(
    item = item_names,
    mean = numeric(n),
    sd = numeric(n),
    prop_missing = numeric(n),
    item_rest_r = numeric(n),
    alpha_if_dropped = numeric(n)
  )

  for (i in seq_len(n)) {
    # Descriptives use every answer; the scale-level figures need complete
    # cases so that all items are compared on the same respondents.
    x <- item_data[[i]]
    stats$mean[i] <- mean(x, na.rm = TRUE)
    stats$sd[i] <- stats::sd(x, na.rm = TRUE)
    stats$prop_missing[i] <- mean(is.na(x))

    rest <- rowSums(complete[, -i, drop = FALSE])
    stats$item_rest_r[i] <- if (nrow(complete) > 2) {
      stats::cor(complete[[i]], rest)
    } else {
      NA_real_
    }
    stats$alpha_if_dropped[i] <- scale_alpha(complete, -dplyr::all_of(item_names[i]))
  }

  stats %>%
    dplyr::mutate(
      flag = dplyr::case_when(
        item_rest_r < 0 ~ "negative: reverse-score?",
        item_rest_r < 0.3 ~ "weak",
        .default = ""
      )
    )
}
