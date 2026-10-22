#' Simple returns from a price series
#'
#' @param prices Numeric vector of prices in time order.
#' @return A numeric vector the same length as `prices`, starting with `NA`.
#' @family return calculators
#' @export
simple_returns <- function(prices) {
  if (length(prices) < 2) {
    return(rep(NA_real_, length(prices)))
  }
  c(NA_real_, prices[-1] / prices[-length(prices)] - 1)
}

#' Log returns from a price series
#'
#' @param prices Numeric vector of prices in time order.
#' @return A numeric vector the same length as `prices`, starting with `NA`.
#' @family return calculators
#' @export
log_returns <- function(prices) {
  if (length(prices) < 2) {
    return(rep(NA_real_, length(prices)))
  }
  c(NA_real_, diff(log(prices)))
}

#' Add per-asset returns to a long price table
#'
#' Sorts by date within each asset and appends a `return` column computed
#' with [simple_returns()] or [log_returns()].
#'
#' @param prices_df A data frame of prices with `asset`, `date` and `price`
#'   columns.
#' @param type Either `"simple"` or `"log"`.
#' @return `prices_df` with an added `return` column.
#' @family return calculators
#' @export
add_asset_returns <- function(prices_df, type = c("simple", "log")) {
  type <- match.arg(type)
  ret_fun <- switch(type,
    simple = simple_returns,
    log = log_returns
  )
  prices_df |>
    dplyr::arrange(.data$asset, .data$date) |>
    dplyr::group_by(.data$asset) |>
    dplyr::mutate(return = ret_fun(.data$price)) |>
    dplyr::ungroup()
}

#' Compound a return series over a period
#'
#' @param returns Numeric vector of periodic simple returns.
#' @param na_rm Logical; drop missing returns before compounding?
#' @return A single cumulative return.
#' @export
compound_return <- function(returns, na_rm = TRUE) {
  if (na_rm) {
    returns <- returns[!is.na(returns)]
  }
  prod(1 + returns) - 1
}

#' Annualise a periodic return series
#'
#' @param returns Numeric vector of periodic simple returns.
#' @param periods_per_year Number of return periods in a year.
#' @return A single annualised (geometric) return.
#' @export
annualised_return <- function(returns, periods_per_year = 252) {
  returns <- returns[!is.na(returns)]
  if (length(returns) == 0) {
    return(NA_real_)
  }
  (1 + compound_return(returns))^(periods_per_year / length(returns)) - 1
}

#' Portfolio returns from asset returns and weights
#'
#' Joins fixed target weights onto per-asset returns and sums the weighted
#' returns by date. Weights are renormalised over the assets that have a
#' return on each date.
#'
#' @param returns_df A data frame of returns with `asset`, `date` and
#'   `return` columns.
#' @param weights A named numeric vector of portfolio weights by asset.
#' @return A data frame with `date`, `return` and `coverage` columns.
#' @family portfolio aggregators
#' @export
portfolio_returns <- function(returns_df, weights) {
  if (is.null(names(weights))) {
    stop("`weights` must be named by asset.", call. = FALSE)
  }
  if (abs(sum(weights) - 1) > 1e-6) {
    warning("`weights` do not sum to one; they will be rescaled.", call. = FALSE)
  }
  weight_df <- tibble::tibble(asset = names(weights), weight = unname(weights))
  returns_df |>
    dplyr::inner_join(weight_df, by = "asset") |>
    dplyr::filter(!is.na(.data$return)) |>
    dplyr::group_by(.data$date) |>
    dplyr::summarise(
      return = sum(.data$weight * .data$return) / sum(.data$weight),
      coverage = sum(.data$weight) / sum(weights),
      .groups = "drop"
    ) |>
    dplyr::arrange(.data$date)
}

#' Portfolio returns with periodic rebalancing
#'
#' Lets weights drift with asset performance between rebalancing dates and
#' resets them to the target weights on each rebalancing date.
#'
#' @param returns_df A data frame of returns with `asset`, `date` and
#'   `return` columns.
#' @param weights A named numeric vector of portfolio weights by asset.
#' @param rebalance_dates Dates on which weights are reset to target.
#' @return A data frame with `date` and `return` columns.
#' @family portfolio aggregators
#' @export
rebalanced_returns <- function(returns_df, weights, rebalance_dates) {
  wide <- returns_df |>
    dplyr::filter(.data$asset %in% names(weights)) |>
    dplyr::select("date", "asset", "return") |>
    tidyr::pivot_wider(names_from = "asset", values_from = "return", values_fill = 0) |>
    dplyr::arrange(.data$date)
  ret_mat <- as.matrix(wide[names(weights)])
  current <- weights / sum(weights)
  out <- numeric(nrow(ret_mat))
  for (i in seq_len(nrow(ret_mat))) {
    if (wide$date[i] %in% rebalance_dates) {
      current <- weights / sum(weights)
    }
    out[i] <- sum(current * ret_mat[i, ])
    grown <- current * (1 + ret_mat[i, ])
    current <- grown / sum(grown)
  }
  tibble::tibble(date = wide$date, return = out)
}
