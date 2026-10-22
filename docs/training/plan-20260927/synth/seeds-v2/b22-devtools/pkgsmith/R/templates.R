#' @title Locate a bundled template
#' @param name File name of the template inside `inst/templates`.
#' @keywords internal
template_path <- function(name) {
  path <- system.file("templates", name, package = "pkgsmith")
  if (!nzchar(path)) {
    stop("Unknown template '", name, "'.", call. = FALSE)
  }
  path
}

#' @title Fill `{{ placeholder }}` slots in template text
#' @description Each placeholder is replaced with the matching element of
#'   `data`. Unmatched placeholders raise an error so that half-rendered files
#'   are never written to disk.
#' @param lines A character vector of template lines.
#' @param data A named list of values to substitute.
#' @return A character vector of rendered lines.
#' @export
render_template <- function(lines, data) {
  pattern <- "\\{\\{\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\}\\}"
  keys <- unique(unlist(regmatches(lines, gregexpr(pattern, lines))))
  keys <- gsub(pattern, "\\1", keys)

  unknown <- setdiff(keys, names(data))
  if (length(unknown) > 0L) {
    stop(
      "Template uses undefined placeholder(s): ",
      paste(unknown, collapse = ", "),
      call. = FALSE
    )
  }

  for (key in keys) {
    value <- paste(format(data[[key]]), collapse = ", ")
    # Backslashes in the value would otherwise be read as backreferences by
    # gsub(), mangling Windows paths and the like.
    value <- gsub("\\", "\\\\", value, fixed = TRUE)
    slot <- paste0("\\{\\{\\s*", key, "\\s*\\}\\}")
    lines <- gsub(slot, value, lines, perl = TRUE)
  }
  lines
}

#' @title Write a rendered template into a package
#' @param template File name of the template inside `inst/templates`.
#' @param save_as Destination path relative to the package root.
#' @param data A named list of values to substitute.
#' @param overwrite Whether an existing file may be replaced.
#' @param path Path to the package root.
#' @return The destination path, invisibly.
#' @export
use_template <- function(template, save_as, data = list(), overwrite = FALSE, path = ".") {
  dest <- file.path(path, save_as)
  if (file.exists(dest) && !overwrite) {
    message("'", save_as, "' already exists; leaving it untouched.")
    return(invisible(dest))
  }

  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  # Every template can use {{ package }} and {{ year }}; `data` wins on clashes.
  pkg <- desc_get_field("Package", default = basename(normalizePath(path)), path = path)
  data <- utils::modifyList(list(package = pkg, year = format(Sys.Date(), "%Y")), data)

  lines <- render_template(readLines(template_path(template), warn = FALSE), data)
  writeLines(lines, dest)
  message("Wrote '", save_as, "'.")
  invisible(dest)
}

#' @title Validate a file stem used to name new source files
#' @param name The proposed file stem, without extension.
#' @return `name`, lower-cased, with spaces replaced by hyphens.
#' @noRd
check_file_stem <- function(name) {
  if (!is.character(name) || length(name) != 1L || !nzchar(name)) {
    stop("`name` must be a single non-empty string.", call. = FALSE)
  }

  name <- tolower(gsub("\\s+", "-", trimws(name)))
  if (grepl("[^a-z0-9._-]", name)) {
    stop("`name` may only contain letters, digits, '.', '_' and '-'.", call. = FALSE)
  }
  sub("\\.[Rr]$", "", name)
}

#' @title Create a new R source file or its test file
#' @description `use_test_file()` writes to `tests/testthat/test-<name>.R`.
#' @param name The file stem, without extension.
#' @param path Path to the package root.
#' @return The new file path, invisibly.
#' @export
use_r_file <- function(name, path = ".") {
  stem <- check_file_stem(name)
  use_template(
    "r-file.R",
    save_as = file.path("R", paste0(stem, ".R")),
    data = list(name = stem),
    path = path
  )
}

#' @rdname use_r_file
#' @export
use_test_file <- function(name, path = ".") {
  stem <- check_file_stem(name)
  use_template(
    "test-file.R",
    save_as = file.path("tests", "testthat", paste0("test-", stem, ".R")),
    data = list(name = stem),
    path = path
  )
}

#' @title Create a new vignette
#' @param name The file stem, without extension.
#' @param title Title shown in the vignette index.
#' @param path Path to the package root.
#' @return The new file path, invisibly.
#' @export
use_vignette_file <- function(name, title = name, path = ".") {
  stem <- check_file_stem(name)

  # Vignettes need knitr and rmarkdown at build time only.
  use_suggest("knitr", path = path)
  use_suggest("rmarkdown", path = path)
  desc_set_field("VignetteBuilder", "knitr", path = path)

  use_template(
    "vignette.Rmd",
    save_as = file.path("vignettes", paste0(stem, ".Rmd")),
    data = list(name = stem, title = title),
    path = path
  )
}
