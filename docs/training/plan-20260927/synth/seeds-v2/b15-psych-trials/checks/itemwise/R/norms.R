# skipped: needs dplyr (build_norm_table, apply_norms)

# to_z_score / to_t_score / to_stanine
stopifnot(isTRUE(all.equal(to_z_score(c(10, 14), 10, 2), c(0, 2))))
stopifnot(isTRUE(all.equal(to_t_score(c(10, 14), 10, 2), c(50, 70))))
stopifnot(identical(to_stanine(c(-100, 10, 100), 10, 2), c(1L, 5L, 9L)))
stopifnot(inherits(try(to_z_score(1, 0, 0), silent = TRUE), "try-error"))

# percentile_rank
stopifnot(isTRUE(all.equal(percentile_rank(c(2, NA), c(1, 2, 3, 4)), c(37.5, NA))))
stopifnot(percentile_rank(0, 1:4) == 0, percentile_rank(9, 1:4) == 100)

# sem_classical
stopifnot(isTRUE(all.equal(sem_classical(10, 0.75), 5)))
stopifnot(sem_classical(10, 1) == 0)

# reliable_change
rc <- reliable_change(pre = c(20, 20, 20), post = c(5, 20, 35), ref_sd = 5, rel = 0.8)
stopifnot(identical(rc$status, c("improved", "unchanged", "deteriorated")))
rc2 <- reliable_change(c(20, 20), c(5, 35), 5, 0.8, higher_is_better = TRUE)
stopifnot(identical(rc2$status, c("deteriorated", "improved")))
