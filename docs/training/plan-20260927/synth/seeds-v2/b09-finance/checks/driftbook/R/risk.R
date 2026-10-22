# skipped: needs dplyr/tidyr/tibble: return_covariance, risk_report

r <- c(0.01, -0.02, 0.03, -0.01, 0.02)

# volatility / downside_deviation
stopifnot(all.equal(volatility(r, 1), sd(r)))
stopifnot(all.equal(downside_deviation(c(0.1, -0.1), 0, 1), sqrt(0.005)))
stopifnot(downside_deviation(c(0.1, 0.2)) == 0)

# sharpe_ratio / sortino_ratio
stopifnot(all.equal(sharpe_ratio(r, 0, 1), mean(r) / sd(r)))
stopifnot(is.na(sharpe_ratio(c(0.01, 0.01))))
stopifnot(is.na(sortino_ratio(c(0.01, 0.02))), sortino_ratio(r) > 0)

# drawdowns / max_drawdown
dd <- drawdowns(c(0.1, -0.5, 1))
stopifnot(all.equal(dd$drawdown, c(0, -0.5, 0)), all.equal(dd$peak, c(1.1, 1.1, 1.1)))
md <- max_drawdown(c(0.1, -0.5, 1))
stopifnot(all.equal(md$depth, -0.5), md$peak_at == 1, md$trough_at == 2, md$recovered_at == 3)
stopifnot(max_drawdown(numeric(0))$depth == 0, max_drawdown(c(0.1, 0.2))$depth == 0)

# portfolio_volatility
cm <- matrix(c(0.04, 0, 0, 0.09), 2, dimnames = list(c("a", "b"), c("a", "b")))
stopifnot(all.equal(portfolio_volatility(c(a = 1, b = 0), cm), 0.2))
stopifnot(inherits(try(portfolio_volatility(c(z = 1), cm), silent = TRUE), "try-error"))
