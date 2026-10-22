test_that("permutedBlock respects the allocation ratio", {
  set.seed(1)
  block <- permutedBlock(c("A", "B", "C"), ratio = c(2, 1, 1), blockSize = 8)
  expect_length(block, 8)
  expect_equal(as.vector(table(block)[c("A", "B", "C")]), c(4, 2, 2))
  expect_error(permutedBlock(c("A", "B"), c(1, 1), blockSize = 5), "multiple of 2")
})

test_that("validateArms rejects duplicate arms and bad ratios", {
  expect_error(validateArms(c("A", "A"), c(1, 1)), "unique")
  expect_error(validateArms(c("A", "B"), c(1, 0)), "positive whole number")
  expect_error(validateArms(c("A", "B"), c(1.5, 1)), "positive whole number")
  expect_equal(validateArms(c("A", "B"), c(2, 1)), c(2L, 1L))
})

test_that("blockRandomList covers n and keeps every block balanced", {
  lst <- blockRandomList(50, arms = c("A", "B"), blockSizes = c(4, 6), seed = 42)
  expect_gte(nrow(lst), 50)
  expect_named(lst, c("allocId", "block", "blockSize", "position", "arm"))
  per_block <- tapply(lst$arm == "A", lst$block, mean)
  expect_true(all(per_block == 0.5))
  expect_false(anyDuplicated(lst$allocId) > 0)
})

test_that("blockRandomList is reproducible from its seed", {
  first <- blockRandomList(30, seed = 7)
  second <- blockRandomList(30, seed = 7)
  third <- blockRandomList(30, seed = 8)
  expect_identical(first, second)
  expect_false(identical(first$arm, third$arm))
  expect_equal(attr(first, "seed"), 7)
})

test_that("stratifiedRandomList builds one list per stratum", {
  strata <- expand.grid(site = c("S1", "S2"), sex = c("F", "M"), stringsAsFactors = FALSE)
  lst <- stratifiedRandomList(strata, nPerStratum = 12, seed = 3, blockSizes = 4)
  expect_equal(nrow(lst), 4 * 12)
  counts <- table(paste(lst$site, lst$sex), lst$arm)
  expect_true(all(counts == 6))
  expect_true(all(startsWith(lst$allocId[lst$site == "S1" & lst$sex == "F"], "S1-F-")))
})

test_that("checkBalance tracks the largest running imbalance", {
  lst <- data.frame(arm = c("A", "A", "A", "B", "B", "B"))
  bal <- checkBalance(lst)
  expect_equal(bal$maxImbalance, 3)
  expect_equal(unname(bal$counts[6, ]), c(3, 3))
  blocked <- blockRandomList(40, blockSizes = 4, seed = 9)
  expect_lte(checkBalance(blocked)$maxImbalance, 2)
})

test_that("maskArms swaps arm labels for kit codes", {
  lst <- blockRandomList(8, blockSizes = 4, seed = 5)
  masked <- maskArms(lst, c(A = "K-301", B = "K-718"))
  expect_false("arm" %in% names(masked))
  expect_setequal(unique(masked$kitCode), c("K-301", "K-718"))
  expect_error(maskArms(lst, c(A = "K-301")), "No kit code")
})

test_that("simpleRandomList draws the requested number of allocations", {
  lst <- simpleRandomList(25, arms = c("drug", "placebo"), seed = 11, prefix = "P")
  expect_equal(nrow(lst), 25)
  expect_true(all(lst$arm %in% c("drug", "placebo")))
  expect_equal(lst$allocId[1], "P01")
})
