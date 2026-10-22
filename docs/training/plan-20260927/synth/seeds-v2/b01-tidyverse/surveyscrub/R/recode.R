#' Recode five-point agreement items
#'
#' Converts agreement labels ("Strongly disagree" to "Strongly agree") into
#' integer scores from 1 to 5. Unrecognised labels become `NA`.
#'
#' @param data A data frame of survey responses.
#' @param cols <[`tidy-select`][dplyr::dplyr_tidy_select]> Item columns to recode.
#' @return `data` with the selected columns converted to integers.
#' @export
recode_agree5 <- function(data, cols) {
  lookup <- c(
    "strongly disagree" = 1L,
    "disagree" = 2L,
    "neither agree nor disagree" = 3L,
    "agree" = 4L,
    "strongly agree" = 5L
  )
  data %>%
    dplyr::mutate(dplyr::across({{ cols }}, ~ unname(lookup[normalise_label(.x)])))
}

#' Recode five-point frequency items
#'
#' Converts frequency labels ("Never" to "Always") into integer scores
#' from 1 to 5. Unrecognised labels become `NA`.
#'
#' @param data A data frame of survey responses.
#' @param cols <[`tidy-select`][dplyr::dplyr_tidy_select]> Item columns to recode.
#' @return `data` with the selected columns converted to integers.
#' @export
recode_freq5 <- function(data, cols) {
  lookup <- c(
    "never" = 1L,
    "rarely" = 2L,
    "sometimes" = 3L,
    "often" = 4L,
    "always" = 5L
  )
  data %>%
    dplyr::mutate(dplyr::across({{ cols }}, ~ unname(lookup[normalise_label(.x)])))
}

#' Recode yes/no items
#'
#' Converts yes/no labels into logical values. Unrecognised labels,
#' including "Don't know", become `NA`.
#'
#' @param data A data frame of survey responses.
#' @param cols <[`tidy-select`][dplyr::dplyr_tidy_select]> Item columns to recode.
#' @return `data` with the selected columns converted to logicals.
#' @export
recode_yesno <- function(data, cols) {
  # Single letters show up when respondents type instead of ticking a box.
  lookup <- c(
    "yes" = TRUE,
    "y" = TRUE,
    "no" = FALSE,
    "n" = FALSE
  )
  data %>%
    dplyr::mutate(dplyr::across({{ cols }}, ~ unname(lookup[normalise_label(.x)])))
}

#' Reverse-score Likert items
#'
#' Flips the scale of negatively worded items so that high scores always
#' mean the same thing, e.g. 1 becomes 5 on a five-point scale.
#'
#' @param data A data frame of survey responses.
#' @param cols <[`tidy-select`][dplyr::dplyr_tidy_select]> Item columns to recode.
#' @param points Number of points on the response scale.
#' @return `data` with the selected columns reverse-scored.
#' @export
reverse_items <- function(data, cols, points = 5L) {
  if (!is.numeric(points) || length(points) != 1 || is.na(points) || points < 2) {
    stop("`points` must be a single number of at least 2.", call. = FALSE)
  }

  data %>%
    dplyr::mutate(dplyr::across({{ cols }}, ~ (points + 1L) - .x))
}

#' Replace numeric missing-value codes with NA
#'
#' Survey tools often store refusals and skips as sentinel values such as
#' -99 or 999. This replaces those codes with `NA` in the selected columns.
#'
#' @param data A data frame of survey responses.
#' @param cols <[`tidy-select`][dplyr::dplyr_tidy_select]> Item columns to recode.
#' @param codes Values to treat as missing.
#' @return `data` with missing-value codes replaced by `NA`.
#' @export
recode_missing_codes <- function(data, cols, codes = c(-99, -98, 999)) {
  data %>%
    dplyr::mutate(dplyr::across({{ cols }}, function(x) {
      x[x %in% codes] <- NA
      x
    }))
}

#' Collapse rare categories of a factor column
#'
#' Lumps levels that make up less than `min_prop` of the non-missing answers
#' into a single "Other" level, keeping the remaining levels in frequency
#' order.
#'
#' @param data A data frame of survey responses.
#' @param col <[`data-masking`][dplyr::dplyr_data_masking]> A single column to collapse.
#' @param min_prop Minimum share a level needs to be kept.
#' @param other_level Label used for the lumped level.
#' @return `data` with `col` converted to a collapsed factor.
#' @export
collapse_rare <- function(data, col, min_prop = 0.05, other_level = "Other") {
  data %>%
    dplyr::mutate(
      {{ col }} := {{ col }} %>%
        forcats::as_factor() %>%
        forcats::fct_lump_prop(prop = min_prop, other_level = other_level) %>%
        forcats::fct_infreq()
    )
}

#' Normalise response labels before lookup
#'
#' @param x Character vector of response labels.
#' @noRd
normalise_label <- function(x) {
  x <- as.character(x)

  # Curly apostrophes come from labels pasted out of word processors.
  x <- stringr::str_replace_all(x, "[\u2018\u2019]", "'")
  stringr::str_squish(stringr::str_to_lower(x))
}
