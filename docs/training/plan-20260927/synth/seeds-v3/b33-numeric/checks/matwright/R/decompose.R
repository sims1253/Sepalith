L = matrix(c(2, 1, 0, 3), 2, 2)
stopifnot(all.equal(forwardSub(L, c(4, 5)), c(2, 1)))
stopifnot(inherits(try(forwardSub(L, 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(forwardSub(matrix(0, 1, 1), 1), silent = TRUE), "try-error"))
U = t(L)
stopifnot(all.equal(backSub(U, c(5, 3)), c(2, 1)))
stopifnot(inherits(try(backSub(U, 1:3), silent = TRUE), "try-error"))
stopifnot(inherits(try(backSub(diag(c(1, 0)), 1:2), silent = TRUE), "try-error"))

A = matrix(c(1, 4, 2, 3, 1, 5, 2, 6, 1), 3, 3)
lu = luFactor(A)
stopifnot(all.equal(A[lu$perm, ], lu$L %*% lu$U), all(lu$L[upper.tri(lu$L)] == 0))
stopifnot(all(abs(lu$U[lower.tri(lu$U)]) < 1e-12), all(diag(lu$L) == 1))

S = crossprod(A) + diag(3)
b = c(1, -2, 3)
for (m in c("lu", "qr", "svd")) {
  dec = matDecompose(A, m)
  stopifnot(inherits(dec, "mw_decomp"), dec$method == m)
  stopifnot(max(abs(reconstruct(dec) - A)) < 1e-10, max(abs(A %*% solveWith(dec, b) - b)) < 1e-10)
}
for (m in c("cholesky", "eigen")) {
  dec = matDecompose(S, m)
  stopifnot(max(abs(reconstruct(dec) - S)) < 1e-10, max(abs(S %*% solveWith(dec, b) - b)) < 1e-10)
  stopifnot(inherits(try(matDecompose(A, m), silent = TRUE), "try-error"))
}
stopifnot(identical(names(matDecompose(A, "qr")$factors), c("Q", "R", "pivot")))
stopifnot(inherits(try(matDecompose(1:4), silent = TRUE), "try-error"))
stopifnot(inherits(try(matDecompose(matrix(1, 2, 3)), silent = TRUE), "try-error"))

stopifnot(identical(decompCheck(matDecompose(A, "lu")), c(unitLower = TRUE, upper = TRUE)))
stopifnot(identical(decompCheck(matDecompose(A, "qr")), c(orthogonal = TRUE, upper = TRUE)))
chk = decompCheck(matDecompose(S, "cholesky"))
stopifnot(identical(chk, c(lower = TRUE, positiveDiag = TRUE)))
stopifnot(identical(decompCheck(matDecompose(S, "eigen")), c(orthogonal = TRUE, sorted = TRUE)))
stopifnot(all(decompCheck(matDecompose(A, "svd"))))
stopifnot(length(decompCheck(matDecompose(A, "svd"))) == 3L)
bad = matDecompose(A, "lu")
bad$factors$U[3, 1] = 1
stopifnot(identical(decompCheck(bad), c(unitLower = TRUE, upper = FALSE)))
