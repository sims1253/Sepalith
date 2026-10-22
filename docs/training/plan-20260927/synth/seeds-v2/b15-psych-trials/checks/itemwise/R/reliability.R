resp <- data.frame(
  q1 = c(1, 2, 3, 4, 5, 2, 3, NA),
  q2 = c(1, 3, 3, 4, 5, 2, 4, 2),
  q3 = c(2, 2, 4, 4, 5, 1, 3, 3),
  q4 = c(1, 2, 3, 5, 4, 2, 2, 4)
)
items <- c("q1", "q2", "q3", "q4")

# item_matrix
m <- item_matrix(resp, items)
stopifnot(is.matrix(m), nrow(m) == 7, typeof(m) == "double")

# cronbach_alpha
a <- cronbach_alpha(resp, items)
stopifnot(a > 0.8, a < 1)
stopifnot(inherits(try(cronbach_alpha(resp, "q1"), silent = TRUE), "try-error"))

# alpha_if_deleted: same respondents for every reduced scale
aid <- alpha_if_deleted(resp, items)
stopifnot(identical(names(aid), items))
stopifnot(all.equal(aid[["q4"]], cronbach_alpha(resp[1:7, ], c("q1", "q2", "q3"))))

# item_total_cor
cc <- item_total_cor(resp, items, corrected = TRUE)
uc <- item_total_cor(resp, items, corrected = FALSE)
stopifnot(length(cc) == 4, all(cc < uc))

# spearman_brown / spearman_brown_factor
stopifnot(isTRUE(all.equal(spearman_brown(0.5, 2), 2 / 3)))
f <- spearman_brown_factor(0.7, 0.9)
stopifnot(isTRUE(all.equal(spearman_brown(0.7, f), 0.9)))

# split_half
sh <- split_half(resp, items, "odd_even")
stopifnot(sh$split == "odd_even", sh$rel > sh$r_half)
stopifnot(inherits(try(split_half(resp, "q1"), silent = TRUE), "try-error"))

# reliability_report / print
rep <- reliability_report(resp, items, name = "mood")
stopifnot(inherits(rep, "itemwise_reliability"), rep$n == 7, nrow(rep$items) == 4)
out <- capture.output(print(rep))
stopifnot(any(grepl("Reliability of 'mood'", out)))
