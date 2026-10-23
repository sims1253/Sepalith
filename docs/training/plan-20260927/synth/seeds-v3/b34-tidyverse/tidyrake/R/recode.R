#' Band order amounts
#'
#' Assigns each order to the band whose lower bound it reaches. The bands
#' are `"micro"`, `"small"`, `"medium"`, `"large"` and `"bulk"`.
#'
#' @param data A data frame of cleaned orders.
#' @param col Name of the amount column.
#' @param bounds A named numeric vector of lower bounds for the bands
#'   `small`, `medium`, `large` and `bulk`.
#' @return `data` with an added factor column `amount_band` and a numeric
#'   column `amount_floor` holding the lower bound of the band (zero for
#'   `"micro"`).
#' @export
band_amount <- function(data, col = "amount",
                        bounds = c(small = 20, medium = 100, large = 500, bulk = 2000)) {
  check_known_cols(data, col, "col")
  bands <- c("micro", "small", "medium", "large", "bulk")

  # Walk the ladder from the highest bound down
  data %>%
    dplyr::mutate(
      amount_band = dplyr::case_when(
        is.na(.data[[col]]) ~ NA_character_,
        .data[[col]] >= bounds[["bulk"]] ~ "bulk",
        .data[[col]] >= bounds[["large"]] ~ "large",
        .data[[col]] >= bounds[["medium"]] ~ "medium",
        .data[[col]] >= bounds[["small"]] ~ "small",
        TRUE ~ "micro"
      ),
      amount_floor = dplyr::case_when(
        amount_band == "bulk" ~ bounds[["bulk"]],
        amount_band == "large" ~ bounds[["large"]],
        amount_band == "medium" ~ bounds[["medium"]],
        amount_band == "small" ~ bounds[["small"]],
        amount_band == "micro" ~ 0,
        TRUE ~ NA_real_
      ),
      amount_band = factor(amount_band, levels = bands)
    )
}

#' Classify free-text order statuses
#'
#' Maps raw status strings onto the levels `"open"`, `"shipped"`,
#' `"delivered"`, `"cancelled"` and `"returned"`. Anything that matches none
#' of the patterns becomes `"unknown"`.
#'
#' @param data A data frame of cleaned orders.
#' @param col Name of the status column.
#' @return `data` with an added factor column `status_class` and a logical
#'   column `status_final`, which is `TRUE` for `"delivered"`,
#'   `"cancelled"` and `"returned"`, `FALSE` for `"open"` and `"shipped"`
#'   and `NA` for `"unknown"`.
#' @export
classify_status <- function(data, col = "status") {
  check_known_cols(data, col, "col")
  levels <- c("open", "shipped", "delivered", "cancelled", "returned", "unknown")

  # The first matching pattern wins
  data %>%
    dplyr::mutate(
      status_class = dplyr::case_when(
        grepl("^(open|new|pending)", .data[[col]]) ~ "open",
        grepl("^(ship|dispatch)", .data[[col]]) ~ "shipped",
        grepl("^(deliver|complete)", .data[[col]]) ~ "delivered",
        grepl("^(cancel|void)", .data[[col]]) ~ "cancelled",
        grepl("^(return|refund)", .data[[col]]) ~ "returned",
        TRUE ~ "unknown"
      ),
      status_final = dplyr::case_when(
        status_class == "open" ~ FALSE,
        status_class == "shipped" ~ FALSE,
        status_class == "delivered" ~ TRUE,
        status_class == "cancelled" ~ TRUE,
        status_class == "returned" ~ TRUE,
        TRUE ~ NA
      ),
      status_class = factor(status_class, levels = levels)
    )
}

#' Label numeric risk scores
#'
#' @param score A numeric vector of scores between 0 and 100.
#' @param thresholds A named numeric vector of lower bounds for the labels
#'   `critical`, `high`, `medium` and `low`. Scores below `low` are
#'   labelled `"none"`.
#' @return A character vector the same length as `score`.
#' @export
label_priority <- function(score,
                           thresholds = c(critical = 90, high = 70, medium = 40, low = 10)) {
  if (!is.numeric(score)) {
    stop("`score` must be numeric.", call. = FALSE)
  }
  out <- rep(NA_character_, length(score))
  open <- !is.na(score)

  # Walk the ladder from the top so each score gets its highest label
  hit <- open & score >= thresholds[["critical"]]
  out[hit] <- "critical"
  open <- open & !hit

  hit <- open & score >= thresholds[["high"]]
  out[hit] <- "high"
  open <- open & !hit

  hit <- open & score >= thresholds[["medium"]]
  out[hit] <- "medium"
  open <- open & !hit

  hit <- open & score >= thresholds[["low"]]
  out[hit] <- "low"
  open <- open & !hit

  out[open] <- "none"
  out
}

#' Recode survey answer columns
#'
#' @param data A data frame.
#' @param cols Columns to recode.
#' @param scale The answer scale used by `cols`: `"yes_no"` gives a logical,
#'   `"agree5"` and `"freq4"` give ordered factors and `"stars"` gives an
#'   integer from 1 to 5.
#' @return `data` with `cols` recoded. Unmatched answers become `NA`.
#' @export
recode_answers <- function(data, cols, scale = c("yes_no", "agree5", "freq4", "stars")) {
  scale <- match.arg(scale)
  check_known_cols(data, cols, "cols")

  for (col in cols) {
    # Answers are compared after trimming and lower-casing
    value <- tolower(trimws(as.character(data[[col]])))
    data[[col]] <- switch(scale,
      yes_no = {
        map <- c(yes = TRUE, y = TRUE, no = FALSE, n = FALSE)
        unname(map[value])
      },
      agree5 = {
        levels <- c("strongly disagree", "disagree", "neutral", "agree", "strongly agree")
        factor(value, levels = levels, ordered = TRUE)
      },
      freq4 = {
        levels <- c("never", "sometimes", "often", "always")
        factor(value, levels = levels, ordered = TRUE)
      },
      stars = {
        stars <- nchar(gsub("[^*]", "", value))
        ifelse(stars >= 1L & stars <= 5L, as.integer(stars), NA_integer_)
      }
    )
  }
  data
}

#' Collapse rare levels into one
#'
#' @param x A character vector or factor.
#' @param min_n Levels seen fewer than `min_n` times are collapsed.
#' @param other Label for the collapsed level.
#' @return A factor whose levels are the kept levels in order of frequency,
#'   followed by `other` when anything was collapsed.
#' @export
collapse_levels <- function(x, min_n = 5L, other = "other") {
  x <- as.character(x)
  counts <- sort(table(x), decreasing = TRUE)
  keep <- names(counts)[counts >= min_n]

  rare <- !is.na(x) & !(x %in% keep)
  x[rare] <- other
  levels <- if (any(rare)) c(keep, other) else keep
  factor(x, levels = levels)
}

#' Flag suspicious orders
#'
#' @param data A data frame of cleaned orders with columns `amount`,
#'   `status`, `order_date` and `customer_id`.
#' @param as_of Reference date for future-dated orders.
#' @return `data` with the logical columns `flag_amount`, `flag_status`,
#'   `flag_date` and `flag_customer`, an integer column `flag_count`, a
#'   logical column `flag_any` and a character column `flag_reason` naming
#'   the first flag that fired, or `NA`.
#' @export
flag_orders <- function(data, as_of = Sys.Date()) {
  check_known_cols(data, c("amount", "status", "order_date", "customer_id"), "data")
  as_of <- as.Date(as_of)

  data %>%
    dplyr::mutate(
      flag_amount = is.na(amount) | amount <= 0,
      flag_status = is.na(status) | status == "",
      flag_date = is.na(order_date) | order_date > as_of,
      flag_customer = is.na(customer_id) | customer_id == "",
      flag_count = as.integer(flag_amount) + as.integer(flag_status) +
        as.integer(flag_date) + as.integer(flag_customer),
      flag_any = flag_count > 0L,
      flag_reason = dplyr::case_when(
        flag_amount ~ "amount",
        flag_status ~ "status",
        flag_date ~ "date",
        flag_customer ~ "customer",
        TRUE ~ NA_character_
      )
    )
}
