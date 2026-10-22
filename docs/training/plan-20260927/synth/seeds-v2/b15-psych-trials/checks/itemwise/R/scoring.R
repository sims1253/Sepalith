resp <- data.frame(a = c(1, 5, NA, NA), b = c(5, 1, 2, NA), c = c(2, 4, 3, NA))
items <- c("a", "b", "c")

# reverse_key / reverse_key_items
stopifnot(isTRUE(all.equal(reverse_key(1:5, 1, 5), 5:1)))
stopifnot(inherits(try(reverse_key("a", 1, 5), silent = TRUE), "try-error"))
rk <- reverse_key_items(resp, "b", 1, 5)
stopifnot(identical(rk$b, c(1, 5, 4, NA)), identical(rk$a, resp$a))

# check_item_range / check_item_missing
stopifnot(inherits(try(check_item_range(resp, items, 1, 4), silent = TRUE), "try-error"))
stopifnot(identical(check_item_missing(resp, items, 1), c(FALSE, FALSE, FALSE, TRUE)))
stopifnot(length(check_item_missing(resp[0, ], items, 0)) == 0)

# score_sum: an all-missing respondent is NA even with a generous max_missing
stopifnot(identical(score_sum(resp, items, reverse = "b"), c(4, 14, NA, NA)))
stopifnot(identical(score_sum(resp, items, max_missing = 3), c(8, 10, 5, NA)))

# score_mean / score_prorated
stopifnot(isTRUE(all.equal(score_mean(resp, items, max_missing = 3), c(8 / 3, 10 / 3, 2.5, NA))))
stopifnot(isTRUE(all.equal(score_prorated(resp, items), c(8, 10, 7.5, NA))))

# score_scales
key <- list(x = list(items = c("a", "b")), y = list(items = "c"))
sc <- score_scales(resp, key, method = "sum")
stopifnot(identical(names(sc), c("x", "y")), identical(sc$y, c(2, 4, 3, NA)))
stopifnot(inherits(try(score_scales(resp, list(1)), silent = TRUE), "try-error"))
