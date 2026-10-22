#' Annualised volatility of a return series
#'
#' @param returns Numeric vector of periodic simple returns.
#' @param periods_per_year Number of return periods in a year.
#' @return A single annualised standard deviation.
#' @export
volatility <- function(returns, periods_per_year = 252) {
  stats::sd(returns, na.rm = TRUE) * sqrt(periods_per_year)
}

#' Annualised downside deviation
#'
#' @description Root mean squared shortfall below `target`; returns above
#'   the target count as zero.
#'
#' @param returns Numeric vector of periodic simple returns.
#' @param target Minimum acceptable periodic return.
#' @param periods_per_year Number of return periods in a year.
#' @return A single annualised downside deviation.
#' @export
downside_deviation <- function(returns, target = 0, periods_per_year = 252) {
  returns <- returns[!is.na(returns)]
  shortfall <- pmin(returns - target, 0)
  sqrt(mean(shortfall^2)) * sqrt(periods_per_year)
}

#' Sharpe and Sortino ratios
#'
#' @description Annualised excess return per unit of total volatility
#'   (Sharpe) or per unit of downside deviation (Sortino).
#'
#' @param returns Numeric vector of periodic simple returns.
#' @param rf Periodic risk-free rate.
#' @param periods_per_year Number of return periods in a year.
#' @return A single annualised ratio, or `NA` when the risk is zero.
#' @rdname risk_adjusted
#' @export
sharpe_ratio <- function(returns, rf = 0, periods_per_year = 252) {
  excess <- returns - rf
  vol <- volatility(excess, periods_per_year)
  if (is.na(vol) || vol == 0) {
    return(NA_real_)
  }
  mean(excess, na.rm = TRUE) * periods_per_year / vol
}

#' @rdname risk_adjusted
#' @export
sortino_ratio <- function(returns, rf = 0, periods_per_year = 252) {
  excess <- returns - rf
  dd <- downside_deviation(excess, 0, periods_per_year)
  if (is.na(dd) || dd == 0) {
    return(NA_real_)
  }
  mean(excess, na.rm = TRUE) * periods_per_year / dd
}

#' Drawdown path of a return series
#'
#' @description Builds a wealth index from `returns` and measures the
#'   percentage fall from its running peak at every point.
#'
#' @param returns Numeric vector of periodic simple returns.
#' @return A data frame with `wealth`, `peak` and `drawdown` columns.
#' @export
drawdowns <- function(returns) {
  returns[is.na(returns)] <- 0
  wealth <- cumprod(1 + returns)
  # the starting wealth of 1 counts as the first peak
  peak <- cummax(c(1, wealth))[-1]
  data.frame(wealth = wealth, peak = peak, drawdown = wealth / peak - 1)
}

#' Maximum drawdown and its timing
#'
#' @param returns Numeric vector of periodic simple returns.
#' @return A list with `depth`, `peak_at`, `trough_at` and `recovered_at`
#'   (row positions; `recovered_at` is `NA` if the loss was never recovered).
#' @export
max_drawdown <- function(returns) {
  dd <- drawdowns(returns)
  no_loss <- list(depth = 0, peak_at = NA_integer_, trough_at = NA_integer_,
    recovered_at = NA_integer_)
  if (nrow(dd) == 0) {
    return(no_loss)
  }
  trough <- which.min(dd$drawdown)
  if (dd$drawdown[trough] == 0) {
    return(no_loss)
  }

  # peak_at is 0 when the fall starts from the initial wealth
  peak_at <- max(which(dd$wealth[seq_len(trough)] >= dd$peak[trough]), 0L)
  after <- which(dd$wealth[-seq_len(trough)] >= dd$peak[trough])
  recovered_at <- if (length(after) > 0) trough + after[1] else NA_integer_

  list(
    depth = dd$drawdown[trough],
    peak_at = peak_at,
    trough_at = trough,
    recovered_at = recovered_at
  )
}

#' Covariance matrix of asset returns
#'
#' @description Pivots a long returns table to wide form and computes the
#'   sample or exponentially weighted covariance matrix.
#'
#' @param returns_df A data frame of returns with `asset`, `date` and
#'   `return` columns.
#' @param method Either `"sample"` or `"ewma"`.
#' @param lambda Decay factor for the `"ewma"` method.
#' @param periods_per_year Number of return periods in a year.
#' @return An annualised covariance matrix with assets as dimnames.
#' @export
return_covariance <- function(returns_df, method = c("sample", "ewma"), lambda = 0.94,
                              periods_per_year = 252) {
  method <- match.arg(method)
  # only dates where every asset has a return
  wide <- returns_df |>
    dplyr::select("date", "asset", "return") |>
    tidyr::pivot_wider(names_from = "asset", values_from = "return") |>
    dplyr::arrange(.data$date) |>
    tidyr::drop_na()
  x <- as.matrix(dplyr::select(wide, -"date"))

  cov_mat <- switch(method,
    sample = stats::cov(x),
    ewma = {
      # most recent row gets weight 1 before normalising
      w <- lambda^rev(seq_len(nrow(x)) - 1)
      w <- w / sum(w)
      centred <- sweep(x, 2, colSums(x * w))
      crossprod(centred * sqrt(w))
    }
  )
  cov_mat * periods_per_year
}

#' Portfolio volatility from weights and a covariance matrix
#'
#' @param weights A named numeric vector of portfolio weights by asset.
#' @param cov_mat A covariance matrix with assets as dimnames.
#' @return A single portfolio standard deviation.
#' @export
portfolio_volatility <- function(weights, cov_mat) {
  missing_assets <- setdiff(names(weights), colnames(cov_mat))
  if (length(missing_assets) > 0) {
    stop("No covariance for: ", paste(missing_assets, collapse = ", "), call. = FALSE)
  }

  sub <- cov_mat[names(weights), names(weights), drop = FALSE]
  sqrt(drop(t(weights) %*% sub %*% weights))
}

#' Risk report for several assets
#'
#' @description Computes return, volatility, risk-adjusted ratios and
#'   drawdown statistics for every asset in a long returns table, plus an
#'   equal-weighted portfolio row when `include_portfolio = TRUE`.
#'
#' @param returns_df A data frame of returns with `asset`, `date` and
#'   `return` columns.
#' @param rf Periodic risk-free rate.
#' @param periods_per_year Number of return periods in a year.
#' @param include_portfolio Logical; append an equal-weighted portfolio row?
#' @return A tibble with one row per asset.
#' @export
risk_report <- function(returns_df, rf = 0, periods_per_year = 252,
                        include_portfolio = TRUE) {
  summarise_one <- function(r) {
    dd <- max_drawdown(r)
    tibble::tibble(
      n_obs = sum(!is.na(r)),
      ann_return = annualised_return(r, periods_per_year),
      ann_vol = volatility(r, periods_per_year),
      sharpe = sharpe_ratio(r, rf, periods_per_year),
      sortino = sortino_ratio(r, rf, periods_per_year),
      max_dd = dd$depth,
      dd_length = dd$trough_at - dd$peak_at,
      recovered = !is.na(dd$recovered_at)
    )
  }

  per_asset <- returns_df |>
    dplyr::arrange(.data$asset, .data$date) |>
    dplyr::group_by(.data$asset) |>
    dplyr::reframe(summarise_one(.data$return))
  if (!include_portfolio) {
    return(per_asset)
  }

  # benchmark row: every asset held in equal proportion
  assets <- unique(returns_df$asset)
  weights <- stats::setNames(rep(1 / length(assets), length(assets)), assets)
  port <- portfolio_returns(returns_df, weights)
  port_row <- summarise_one(port$return) |>
    dplyr::mutate(asset = "(equal weight)", .before = 1)

  dplyr::bind_rows(per_asset, port_row)
}
