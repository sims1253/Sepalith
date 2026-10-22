# check_split_size
stopifnot(identical(check_split_size(10, 3), 10L))
stopifnot(inherits(try(check_split_size(2, 3), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_split_size(5, 1), silent = TRUE), "try-error"))

# new_splits
ns <- new_splits(list(c(3L, 1L), 2L), 3, "manual")
stopifnot(identical(ns[[1]]$assessment, c(1L, 3L)), identical(ns[[2]]$analysis, c(1L, 3L)))
stopifnot(identical(attr(ns, "method"), "manual"), inherits(ns, "fw_splits"))

# kfold_splits
kf <- kfold_splits(11, k = 3, seed = 5)
held <- unlist(lapply(kf, `[[`, "assessment"))
stopifnot(length(kf) == 3, setequal(held, 1:11), !anyDuplicated(held))

# stratified_splits
st <- stratified_splits(rep(c("a", "b"), c(6, 3)), k = 3, seed = 2)
stopifnot(length(st) == 3, setequal(unlist(lapply(st, `[[`, "assessment")), 1:9))
stopifnot(inherits(try(stratified_splits(c("a", NA, "b"), k = 2), silent = TRUE), "try-error"))

# grouped_splits, including a missing group label
g <- c("a", "a", "b", NA, NA, "c")
gs <- grouped_splits(g, k = 2, seed = 1)
stopifnot(length(gs) == 2, setequal(unlist(lapply(gs, `[[`, "assessment")), 1:6))
for (f in gs) stopifnot(!any(g[f$analysis] %in% g[f$assessment]))

# logo_splits
lg <- logo_splits(c("x", NA, "x", "y"))
stopifnot(length(lg) == 3, identical(lg[[2]]$assessment, 2L))
stopifnot(inherits(try(logo_splits(c(1, 1)), silent = TRUE), "try-error"))

# rolling_splits
rs <- rolling_splits(10, initial = 4, assess = 3)
stopifnot(length(rs) == 2, identical(rs[[2]]$analysis, 1:7), identical(rs[[2]]$assessment, 8:10))
stopifnot(inherits(try(rolling_splits(5, initial = 0), silent = TRUE), "try-error"))
sl <- rolling_splits(6, initial = 2, assess = 2, cumulative = FALSE)
stopifnot(identical(sl[[2]]$analysis, 3:4))

# print.fw_splits
txt <- capture.output(print(rs))
stopifnot(identical(txt[1], "<fw_splits: rolling, 2 folds, n = 10>"), length(txt) == 3)
