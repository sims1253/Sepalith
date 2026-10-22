trainDf <- data.frame(
  age = c(21, 35, 48, 52, 29, 40),
  income = c(30, 55, 72, 90, 41, 60),
  flat = rep(3, 6),
  region = c("n", "s", "s", "e", "n", "w"),
  stringsAsFactors = FALSE
)

test_that("fitStandardScaler stores mean and sd of numeric columns", {
  sc <- fitStandardScaler(trainDf)
  expect_s3_class(sc, "fp_scaler")
  expect_equal(sc$cols, c("age", "income", "flat"))
  expect_equal(sc$center[["age"]], mean(trainDf$age))
  expect_equal(sc$scale[["income"]], sd(trainDf$income))
})

test_that("constant columns get a scale of one", {
  expect_equal(fitStandardScaler(trainDf)$scale[["flat"]], 1)
  expect_equal(fitMinMaxScaler(trainDf)$scale[["flat"]], 1)
  expect_equal(fitRobustScaler(trainDf)$scale[["flat"]], 1)
})

test_that("applyScaler uses training parameters on new data", {
  sc <- fitMinMaxScaler(trainDf, c("age", "income"))
  newDf <- data.frame(age = c(21, 52, 60), income = c(30, 90, 30))

  # 60 lies outside the training range, so it maps above one
  out <- applyScaler(sc, newDf)
  expect_equal(out$age, c(0, 1, (60 - 21) / 31))
  expect_equal(out$income, c(0, 1, 0))
})

test_that("invertScaler undoes applyScaler", {
  for (fitter in list(fitStandardScaler, fitMinMaxScaler, fitRobustScaler)) {
    sc <- fitter(trainDf, c("age", "income"))
    roundTrip <- invertScaler(sc, applyScaler(sc, trainDf))
    expect_equal(roundTrip$age, trainDf$age)
    expect_equal(roundTrip$income, trainDf$income)
  }
})

test_that("scalers reject unknown and non-numeric columns", {
  expect_error(fitStandardScaler(trainDf, "height"), "Unknown columns: height")
  expect_error(fitStandardScaler(trainDf, "region"), "not numeric: region")

  sc <- fitStandardScaler(trainDf, "age")
  expect_error(applyScaler(sc, trainDf["income"]), "missing columns: age")
})

test_that("writeScaler and readScaler round-trip parameters", {
  sc <- fitRobustScaler(trainDf, c("age", "income"))
  path <- withr::local_tempfile(fileext = ".csv")
  writeScaler(sc, path)

  restored <- readScaler(path)
  expect_equal(restored$center, sc$center)
  expect_equal(restored$scale, sc$scale)
  expect_identical(restored$type, "robust")
})

test_that("readScaler refuses files without scaler columns", {
  path <- withr::local_tempfile(fileext = ".csv")
  utils::write.csv(data.frame(a = 1, b = 2), path, row.names = FALSE)
  expect_error(readScaler(path), "does not look like a saved scaler")
})

test_that("print.fp_scaler shows type and column count", {
  sc <- fitStandardScaler(trainDf, c("age", "income"))
  expect_output(print(sc), "<fp_scaler: standard on 2 columns>")
  expect_output(print(sc), "income")
})
