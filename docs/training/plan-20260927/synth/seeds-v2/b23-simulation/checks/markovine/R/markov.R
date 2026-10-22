P <- matrix(c(0.9, 0.1, 0.5, 0.5), 2, byrow = TRUE)
chk <- check.transition.matrix(P)
stopifnot(identical(rownames(chk), c("1", "2")), identical(colnames(chk), c("1", "2")))
stopifnot(inherits(try(check.transition.matrix(P * 2), silent = TRUE), "try-error"))
P.na <- P
P.na[1, 1] <- NA
stopifnot(grepl("missing", try(check.transition.matrix(P.na), silent = TRUE)))

set.seed(5)
path <- mc.chain.path(P, 20, start = "2")
stopifnot(length(path) == 21, path[1] == "2", all(path %in% c("1", "2")))
stopifnot(inherits(try(mc.chain.path(P, 5, start = 3), silent = TRUE), "try-error"))

stopifnot(isTRUE(all.equal(mc.n.step(P, 3), unname(P %*% P %*% P), check.attributes = FALSE)))
stopifnot(isTRUE(all.equal(unname(mc.n.step(P, 0)), diag(2))))
stopifnot(inherits(try(mc.n.step(P, -1), silent = TRUE), "try-error"))

st <- mc.stationary(P)
stopifnot(isTRUE(all.equal(unname(st), c(5, 1) / 6)), identical(names(st), c("1", "2")))

fit <- mc.fit.path(c("a", "b", "a", "a", NA, "b", "c"))
stopifnot(sum(fit$counts) == 4, fit$counts["a", "b"] == 1, fit$counts["b", "c"] == 1)
stopifnot(fit$P["c", "c"] == 1, isTRUE(all.equal(unname(rowSums(fit$P)), rep(1, 3))))
fit2 <- mc.fit.path(c("a", "z", "a"), states = "a")
stopifnot(sum(fit2$counts) == 0, fit2$P["a", "a"] == 1)

A <- matrix(c(1, 0, 0, 0.5, 0, 0.5, 0, 0, 1), 3, byrow = TRUE,
  dimnames = list(c("L", "M", "W"), c("L", "M", "W")))
ab <- mc.absorption(A)
stopifnot(identical(ab$transient, "M"), isTRUE(all.equal(c(ab$B), c(0.5, 0.5))))
stopifnot(ab$steps[["M"]] == 1)
stopifnot(inherits(try(mc.absorption(P), silent = TRUE), "try-error"))

set.seed(6)
h <- mc.hitting.times(A, start = "M", target = "W", reps = 50, max.steps = 5)
stopifnot(all(is.na(h) | h == 1), any(!is.na(h)), any(is.na(h)))
stopifnot(inherits(try(mc.hitting.times(A, "M", "Q", reps = 2), silent = TRUE), "try-error"))
