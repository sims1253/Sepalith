df <- data.frame(
  sex = c("f", "f", "f", "m", "m", "m", "m", "m"),
  age = c("y", "o", "o", "y", "y", "o", "o", "y"),
  stringsAsFactors = FALSE
)
margins <- list(sex = c(f = 52, m = 48), age = c(y = 0.4, o = 0.6))

# check_margins
stopifnot(identical(check_margins(margins, df), margins))
stopifnot(inherits(try(check_margins(list(x = c(a = 1)), df), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_margins(list(sex = c(f = 1)), df), silent = TRUE), "try-error"))
df_na <- df
df_na$sex[2] <- NA
stopifnot(inherits(try(check_margins(margins, df_na), silent = TRUE), "try-error"))

# margins_as_shares
sh <- margins_as_shares(margins)
stopifnot(all.equal(unname(sh$sex), c(0.52, 0.48)), all.equal(sum(sh$age), 1))

# rake_one_margin
w1 <- rake_one_margin(rep(1, 8), df$sex, sh$sex)
stopifnot(all.equal(sum(w1[df$sex == "f"]) / sum(w1), 0.52))

# margin_gap
stopifnot(margin_gap(rep(1, 8), df, list(sex = c(f = 3 / 8, m = 5 / 8))) < 1e-12)
stopifnot(all.equal(margin_gap(rep(1, 8), df, list(sex = c(f = 0.5, m = 0.5))), 0.125))

# rake_weights
w <- rake_weights(df, margins, total = 100)
stopifnot(isTRUE(attr(w, "converged")), all.equal(sum(w), 100))
stopifnot(abs(sum(w[df$age == "o"]) / 100 - 0.6) < 1e-5)

# compare_margins
cmp <- compare_margins(df, w, margins)
stopifnot(nrow(cmp) == 4, max(abs(cmp$difference)) < 1e-5)
stopifnot(all.equal(cmp$unweighted[cmp$category == "f"], 3 / 8))
