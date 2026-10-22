local_pkg <- function(version = "0.1.0", env = parent.frame()) {
  dir <- withr::local_tempdir(.local_envir = env)
  writeLines(
    c(
      "Package: toypkg",
      "Title: A Toy Package for Tests",
      paste0("Version: ", version),
      "Description: Exists only for unit tests.",
      "License: MIT + file LICENSE"
    ),
    file.path(dir, "DESCRIPTION")
  )
  dir
}

test_that("semver_parse() splits three and four component versions", {
  v <- semver_parse("1.2.3")
  expect_equal(c(v$major, v$minor, v$patch, v$dev), c(1L, 2L, 3L, 0L))
  expect_true(is.na(v$pre))
  v <- semver_parse("0.4.1.9002")
  expect_equal(c(v$major, v$minor, v$patch, v$dev), c(0L, 4L, 1L, 9002L))
  v <- semver_parse("2.0.0-rc.1")
  expect_equal(v$pre, "rc.1")
})

test_that("semver_parse() rejects malformed input", {
  expect_error(semver_parse("1"), "not a valid version")
  expect_error(semver_parse("1.2.x"), "not a valid version")
  expect_error(semver_parse("1.2.3.4.5"), "not a valid version")
})

test_that("semver_format() round-trips through semver_parse()", {
  for (x in c("1.0.0", "0.3.2.9000", "3.1.4-beta")) {
    expect_identical(semver_format(semver_parse(x)), x)
  }
})

test_that("semver_compare() orders releases, pre-releases and dev versions", {
  expect_identical(semver_compare("1.0.0", "1.0.0"), 0L)
  expect_identical(semver_compare("1.0.0", "1.1.0"), -1L)
  expect_identical(semver_compare("2.0.0", "1.9.9"), 1L)
  expect_identical(semver_compare("1.0.0-rc.1", "1.0.0"), -1L)
  expect_identical(semver_compare("1.0.0", "1.0.0.9000"), -1L)
  expect_identical(semver_compare("1.0.0-alpha", "1.0.0-beta"), -1L)
})

test_that("bump_major() resets minor, patch and dev", {
  expect_identical(bump_major("1.4.2"), "2.0.0")
  expect_identical(bump_major("0.9.1.9000"), "1.0.0")
})

test_that("bump_minor() resets patch and dev", {
  expect_identical(bump_minor("1.4.2"), "1.5.0")
  expect_identical(bump_minor("0.9.1.9000"), "0.10.0")
})

test_that("bump_patch() resets dev", {
  expect_identical(bump_patch("1.4.2"), "1.4.3")
  expect_identical(bump_patch("0.9.1.9000"), "0.9.2")
})

test_that("bump_dev() starts at 9000 and then increments", {
  expect_identical(bump_dev("1.4.2"), "1.4.2.9000")
  expect_identical(bump_dev("1.4.2.9000"), "1.4.2.9001")
})

test_that("use_version() rewrites the DESCRIPTION Version field", {
  dir <- local_pkg("0.3.1")
  expect_message(new <- use_version("minor", path = dir), "0.3.1 -> 0.4.0")
  expect_identical(new, "0.4.0")
  expect_identical(desc_get_field("Version", path = dir), "0.4.0")
  expect_error(use_version("huge", path = dir))
})
