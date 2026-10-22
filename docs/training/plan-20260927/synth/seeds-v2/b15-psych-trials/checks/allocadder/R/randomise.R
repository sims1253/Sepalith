fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")

# validateArms
stopifnot(identical(validateArms(c("A", "B"), c(2, 1)), c(2L, 1L)))
stopifnot(fails(validateArms(c("A", "A"), c(1, 1))))
stopifnot(fails(validateArms(c("A", "B"), c(1, NA))), fails(validateArms(c("A", "B"), c(1.5, 1))))

# permutedBlock
set.seed(1)
blk <- permutedBlock(c("A", "B", "C"), c(2, 1, 1), 8)
stopifnot(length(blk) == 8, identical(as.vector(table(blk)[c("A", "B", "C")]), c(4L, 2L, 2L)))
stopifnot(fails(permutedBlock(c("A", "B"), c(1, 1), 5)))

# blockRandomList
lst <- blockRandomList(50, seed = 42)
stopifnot(nrow(lst) >= 50, all(tapply(lst$arm == "A", lst$block, mean) == 0.5))
stopifnot(identical(lst, blockRandomList(50, seed = 42)), attr(lst, "seed") == 42)
stopifnot(fails(blockRandomList(0)))

# stratifiedRandomList: factor strata give readable prefixes
strata <- expand.grid(site = c("S1", "S2"), sex = c("F", "M"))
st <- stratifiedRandomList(strata, nPerStratum = 8, seed = 3, blockSizes = 4)
stopifnot(nrow(st) == 32, all(table(paste(st$site, st$sex), st$arm) == 4))
stopifnot(all(startsWith(st$allocId[st$site == "S2" & st$sex == "M"], "S2-M-")))

# simpleRandomList
sl <- simpleRandomList(25, arms = c("drug", "placebo"), seed = 11, prefix = "P")
stopifnot(nrow(sl) == 25, sl$allocId[1] == "P01", all(sl$arm %in% c("drug", "placebo")))

# checkBalance
bal <- checkBalance(data.frame(arm = c("A", "A", "A", "B", "B", "B")))
stopifnot(bal$maxImbalance == 3, all(bal$counts[6, ] == 3))
stopifnot(checkBalance(data.frame(arm = c("A", NA, "B")), c("A", "B"))$maxImbalance == 1)
stopifnot(checkBalance(data.frame(arm = character()), c("A", "B"))$maxImbalance == 0)

# maskArms
mk <- maskArms(blockRandomList(8, blockSizes = 4, seed = 5), c(A = "K1", B = "K2"))
stopifnot(!"arm" %in% names(mk), setequal(mk$kitCode, c("K1", "K2")))
stopifnot(fails(maskArms(lst, c(A = "K1"))))
