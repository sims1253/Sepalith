# skipped: needs dplyr/tidyr/tibble: add_asset_returns, portfolio_returns,
# rebalanced_returns

# simple_returns / log_returns
stopifnot(all.equal(simple_returns(c(100, 110, 99)), c(NA, 0.1, -0.1)))
stopifnot(identical(simple_returns(5), NA_real_))
stopifnot(all.equal(log_returns(c(1, exp(1))), c(NA, 1)))
stopifnot(length(log_returns(numeric(0))) == 0)

# compound_return
stopifnot(all.equal(compound_return(c(0.1, NA, 0.1)), 0.21))
stopifnot(is.na(compound_return(c(0.1, NA), na_rm = FALSE)))

# annualised_return
stopifnot(all.equal(annualised_return(c(0.1, 0.1), periods_per_year = 2), 0.21))
stopifnot(is.na(annualised_return(c(NA_real_, NA_real_))))
