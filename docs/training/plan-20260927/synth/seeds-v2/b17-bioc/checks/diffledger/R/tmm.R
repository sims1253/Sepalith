# check_counts() lives in normalise.R; a permissive stand-in is enough here.
check_counts <- function(counts) invisible(counts)

set.seed(1)
base <- c(rpois(200, 50), rpois(50, 500))
counts <- cbind(s1 = base, s2 = base * 2, s3 = base)
rownames(counts) <- paste0("g", seq_len(nrow(counts)))

stopifnot(
  identical(.tmm_m_values(4, 2, 10, 10), 1),
  identical(.tmm_a_values(4, 4, 16, 16), -2),
  .tmm_variance(5, 5, 10, 10) == 0.2,
  .tmm_one(base, base, 1000, 1000) == 1
)

f <- tmm_factors(counts)
stopifnot(
  identical(names(f), colnames(counts)),
  isTRUE(all.equal(prod(f), 1)),
  isTRUE(all.equal(unname(f[1]), unname(f[2]))),
  isTRUE(all.equal(tmm_factors(counts, ref_column = "s3"), f)),
  inherits(try(tmm_factors(counts, ref_column = "zz"), silent = TRUE), "try-error")
)

els <- effective_lib_size(counts, norm_factors = c(1, 1, 1))
stopifnot(
  identical(els, colSums(counts)),
  inherits(try(effective_lib_size(counts, c(1, 1)), silent = TRUE), "try-error"),
  inherits(try(tmm_factors(cbind(counts, s4 = 0)), silent = TRUE), "try-error")
)
