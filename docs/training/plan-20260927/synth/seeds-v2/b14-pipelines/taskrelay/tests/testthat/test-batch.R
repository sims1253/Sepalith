make_inbox <- function(n = 3, empty = 0) {
  dir <- withr::local_tempdir(.local_envir = parent.frame())
  for (i in seq_len(n)) {
    writeLines(c("id,value", paste0(i, ",", i * 10)), file.path(dir, sprintf("in_%02d.csv", i)))
  }
  for (i in seq_len(empty)) {
    file.create(file.path(dir, sprintf("empty_%02d.csv", i)))
  }
  dir
}

test_that("list_batch_files skips empty files and sorts the result", {
  dir <- make_inbox(n = 3, empty = 2)
  files <- list_batch_files(dir)
  expect_length(files, 3)
  expect_equal(basename(files), c("in_01.csv", "in_02.csv", "in_03.csv"))
})

test_that("list_batch_files errors on a missing directory", {
  expect_error(
    list_batch_files(file.path(tempdir(), "does-not-exist")),
    "does not exist"
  )
})

test_that("batch_manifest returns one row per file with checksums", {
  dir <- make_inbox(n = 2)
  manifest <- batch_manifest(list_batch_files(dir))
  expect_s3_class(manifest, "data.frame")
  expect_equal(nrow(manifest), 2)
  expect_true(all(nchar(manifest$md5) == 32))
})

test_that("batch_manifest handles an empty input", {
  manifest <- batch_manifest(character())
  expect_equal(nrow(manifest), 0)
  expect_named(manifest, c("path", "size", "mtime", "md5"))
})

test_that("run_batch records ok, warning and error outcomes", {
  dir <- make_inbox(n = 3)
  files <- list_batch_files(dir)
  fun <- function(path) {
    if (grepl("02", basename(path))) stop("bad row")
    if (grepl("03", basename(path))) warning("odd row")
    nrow(utils::read.csv(path))
  }

  results <- run_batch(files, fun)
  summary <- summarise_batch(results)
  expect_s3_class(results, "taskrelay_batch")
  expect_equal(summary$status, c("ok", "error", "warning"))
  expect_equal(results[[1]]$value, 1L)
  # A warning is recorded but does not throw away the value
  expect_equal(results[[3]]$value, 1L)
  expect_equal(results[[3]]$message, "odd row")
})

test_that("run_batch stops at the first error when asked", {
  dir <- make_inbox(n = 3)
  files <- list_batch_files(dir)
  results <- run_batch(files, function(path) stop("nope"), stop_on_error = TRUE)
  expect_length(results, 1)
  expect_equal(results[[1]]$status, "error")
})

test_that("archive_processed and quarantine_failed move the right files", {
  dir <- make_inbox(n = 2)
  files <- list_batch_files(dir)
  summary <- data.frame(
    file = files, status = c("ok", "error"), message = NA_character_, elapsed = 0,
    stringsAsFactors = FALSE
  )

  archived <- archive_processed(summary, file.path(dir, "archive"))
  quarantined <- quarantine_failed(summary, file.path(dir, "quarantine"))
  expect_equal(basename(archived), "in_01.csv")
  expect_equal(basename(quarantined), "in_02.csv")
  expect_false(any(file.exists(files)))
})

test_that("write_batch_report writes a csv and a text overview", {
  dir <- withr::local_tempdir()
  summary <- data.frame(
    file = c("a.csv", "b.csv"), status = c("ok", "error"),
    message = c(NA, "bad row"), elapsed = c(0.5, 1.25), stringsAsFactors = FALSE
  )
  paths <- write_batch_report(summary, file.path(dir, "report", "batch.csv"))

  expect_true(all(file.exists(paths)))
  overview <- readLines(paths[["txt"]])
  expect_match(overview[2], "^ok\\s+1$")
  expect_match(overview[length(overview)], "1.8 s")
})
