counts <- matrix(
  c(10, 20, 30, 40,
    5, 0, 15, 20,
    100, 200, 300, 400),
  nrow = 3, byrow = TRUE,
  dimnames = list(c("a", "b", "c"), paste0("s", 1:4))
)
lens <- c(1000, 2000, 500)

stopifnot(
  identical(check_counts(counts), counts),
  inherits(try(check_counts(-counts), silent = TRUE), "try-error"),
  inherits(try(check_lengths(c(1, 0, 2), counts), silent = TRUE), "try-error")
)

stopifnot(
  isTRUE(all.equal(unname(colSums(cpm(counts))), rep(1e6, 4))),
  isTRUE(all.equal(cpm(counts, norm_factors = rep(2, 4)), cpm(counts) / 2)),
  all(is.finite(log_cpm(counts))),
  identical(dim(log_cpm(counts)), dim(counts))
)

stopifnot(
  isTRUE(all.equal(unname(colSums(tpm(counts, lens))), rep(1e6, 4))),
  isTRUE(all.equal(rpkm(counts, lens)[1, ], cpm(counts)[1, ])),
  inherits(try(tpm(counts, lens[1:2]), silent = TRUE), "try-error")
)

sf <- size_factors_ratio(counts)
stopifnot(
  identical(names(sf), colnames(counts)),
  isTRUE(all.equal(unname(sf[2] / sf[1]), 2))
)
uq <- size_factors_quantile(counts)
stopifnot(
  isTRUE(all.equal(exp(mean(log(uq))), 1)),
  inherits(try(size_factors_quantile(cbind(counts, 0)), silent = TRUE), "try-error")
)

stopifnot(
  identical(normalise_counts(counts), cpm(counts)),
  inherits(try(normalise_counts(counts, "tpm"), silent = TRUE), "try-error"),
  isTRUE(all.equal(normalise_counts(counts, "ratio"), sweep(counts, 2, sf, "/")))
)
