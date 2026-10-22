df <- data.frame(
  sex = c("f", "f", "f", "m", "m", "m", "m", "m"),
  region = c("n", "s", "n", "s", "n", "s", "n", "s"),
  stringsAsFactors = FALSE
)
pop <- data.frame(
  sex = c("f", "f", "m", "m"),
  region = c("n", "s", "n", "s"),
  n = c(260, 240, 250, 250)
)

# cell_key
stopifnot(identical(cell_key(df, c("sex", "region"))[1:2], c("f:n", "f:s")))
stopifnot(inherits(try(cell_key(df, "age"), silent = TRUE), "try-error"))

# sample_cell_totals / population_cell_totals
st <- sample_cell_totals(df, c("sex", "region"), c(NA, rep(1, 7)))
stopifnot(st[["f:n"]] == 1, st[["m:s"]] == 3)
pt <- population_cell_totals(pop, c("sex", "region"))
stopifnot(pt[["f:n"]] == 260, length(pt) == 4)

# collapse_sparse_cells
keys <- collapse_sparse_cells(df, c("sex", "region"), min_n = 2L)
stopifnot(all(keys[df$sex == "f"] == "f:n"), length(unique(keys)) == 3)
stopifnot(identical(collapse_sparse_cells(df, "sex"), df$sex))

# poststrat_factors
f <- poststrat_factors(c(a = 2, b = 4), c(a = 10, b = 20))
stopifnot(all.equal(unname(f), c(5, 5)))
bad <- try(suppressWarnings(poststrat_factors(c(z = 1), c(a = 1))), silent = TRUE)
stopifnot(inherits(bad, "try-error"))

# poststratify
w <- poststratify(df, pop, strata = c("sex", "region"))
stopifnot(all.equal(sum(w), 1000), all.equal(sum(w[df$sex == "f" & df$region == "n"]), 260))
