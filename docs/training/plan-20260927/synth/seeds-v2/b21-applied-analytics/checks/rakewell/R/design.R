# skipped: needs survey (design_from_weights after validation, replicate_design,
# tidy_svystat, weighted_mean_by, weighted_total_by, weighted_prop_by)

# as_rhs_formula
stopifnot(is.null(as_rhs_formula(NULL)))
stopifnot(identical(deparse(as_rhs_formula(c("a", "b"))), "~a + b"))
stopifnot(identical(deparse(as_rhs_formula("1")), "~1"))

# design_from_weights: input validation happens before survey is touched
df <- data.frame(x = 1:3)
stopifnot(inherits(try(design_from_weights(df, c(1, 1)), silent = TRUE), "try-error"))
stopifnot(inherits(try(design_from_weights(df, c(1, NA, 1)), silent = TRUE), "try-error"))
