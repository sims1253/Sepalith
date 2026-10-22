test_that("path_ext_set() replaces, adds and removes extensions", {
  expect_identical(path_ext_set("data/raw.csv", "txt"), "data/raw.txt")
  expect_identical(path_ext_set("data/raw.csv", ".parquet"), "data/raw.parquet")
  expect_identical(path_ext_set("notes", "md"), "notes.md")
  expect_identical(path_ext_set("data/raw.csv", ""), "data/raw")
  expect_error(path_ext_set("a.csv", c("x", "y")))
})

test_that("path_ext_strip() only touches the final component", {
  expect_identical(path_ext_strip("archive.tar.gz"), "archive.tar")
  expect_identical(path_ext_strip("v1.2/readme"), "v1.2/readme")
  expect_identical(path_ext_strip(".Rprofile"), ".Rprofile")
  expect_identical(path_ext_strip(character()), character())
})

test_that("path_ext() mirrors path_ext_strip()", {
  paths <- c("raw.csv", "archive.tar.gz", "README", ".Rprofile", NA)
  expect_identical(path_ext(paths), c("csv", "gz", "", "", NA))
  expect_identical(paste0(path_ext_strip(paths[1:2]), ".", path_ext(paths[1:2])), paths[1:2])
})

test_that("path_is_absolute() recognises POSIX and Windows roots", {
  absolute <- c("/srv/app", "~/notes", "C:/Users/me", "D:\\data", "\\\\server\\share")
  relative <- c("data/raw.csv", "./script.R", "../up", "C:relative")
  expect_true(all(path_is_absolute(absolute)))
  expect_false(any(path_is_absolute(relative)))
})

test_that("path_rel_to() walks up and down from the base", {
  skip_on_os("windows")
  expect_identical(path_rel_to("/srv/app/data/raw.csv", "/srv/app"), "data/raw.csv")
  expect_identical(path_rel_to("/srv/other/x.R", "/srv/app"), "../other/x.R")
  expect_identical(path_rel_to("/srv/app", "/srv/app"), ".")
  expect_identical(
    path_rel_to(c("/a/b", "/a/c/d"), "/a"),
    c("b", "c/d")
  )
})

test_that("path_sanitize() produces portable file names", {
  expect_identical(path_sanitize("Q3 report: draft/final?"), "Q3_report_draft_final_")
  expect_identical(path_sanitize("a  b", replacement = "-"), "a-b")
  expect_identical(path_sanitize("...hidden"), "hidden")
  expect_identical(path_sanitize(""), "untitled")
  expect_identical(nchar(path_sanitize(strrep("x", 300))), 255L)
  expect_error(path_sanitize("x", replacement = "/"), class = "pathwren_error_replacement")
})

test_that("path_unique() appends numeric suffixes before the extension", {
  dir <- withr::local_tempdir()
  target <- file.path(dir, "export.csv")
  expect_identical(path_unique(target), target)
  file.create(target)
  expect_identical(path_unique(target), file.path(dir, "export_1.csv"))
  file.create(file.path(dir, "export_1.csv"))
  expect_identical(path_unique(target), file.path(dir, "export_2.csv"))
  err <- expect_error(path_unique(target, max_tries = 1L), class = "pathwren_error_exhausted")
  expect_identical(err$path, target)
})

test_that("path_ensure_dir() creates nested directories once", {
  dir <- withr::local_tempdir()
  nested <- file.path(dir, "a", "b", "c")
  expect_invisible(path_ensure_dir(nested))
  expect_true(dir.exists(nested))
  expect_identical(path_ensure_dir(nested), nested)
  blocker <- file.path(dir, "plain-file")
  file.create(blocker)
  expect_error(path_ensure_dir(blocker), class = "pathwren_error_not_dir")
})
