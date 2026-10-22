# skipped: needs lpSolve (assignExact, compareAssignments)
cost <- matrix(c(4, 2, 8, 4, 3, 7, 3, 1, 6), 3, byrow = TRUE)

# checkCostMatrix
cm <- checkCostMatrix(cost)
stopifnot(identical(rownames(cm), c("w1", "w2", "w3")))
stopifnot(identical(colnames(cm), c("t1", "t2", "t3")))
stopifnot(inherits(try(checkCostMatrix(matrix("a")), silent = TRUE), "try-error"))

# assignmentTable skips idle workers
tab <- assignmentTable(cm, c(2L, NA, 1L))
stopifnot(identical(tab$worker, c("w1", "w3")), identical(tab$cost, c(2, 3)))

# assignGreedy / assignRegret give complete assignments
g <- assignGreedy(cost)
stopifnot(nrow(g) == 3, !anyDuplicated(g$task), sum(g$cost) == 12)
r <- assignRegret(cost)
stopifnot(nrow(r) == 3, !anyDuplicated(r$task), sum(r$cost) == 12)

# improveAssignment never increases total cost
bad_start <- data.frame(worker = c("w1", "w2", "w3"), task = c("t3", "t2", "t1"))
better <- improveAssignment(cost, bad_start)
stopifnot(sum(better$cost) <= 8 + 3 + 3, sum(better$cost) == 12)
