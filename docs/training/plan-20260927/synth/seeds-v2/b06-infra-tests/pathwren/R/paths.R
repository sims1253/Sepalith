#' @title Replace or remove a file extension
#' @description `path_ext_set()` replaces the final extension of each path,
#'   adding one if none is present. `path_ext_strip()` removes it.
#' @param path A character vector of file paths.
#' @param ext The new extension, with or without a leading dot.
#' @return A character vector the same length as `path`.
#' @export
#' @examples
#' path_ext_set(c("data/raw.csv", "notes"), "txt")
path_ext_set <- function(path, ext) {
  stopifnot(is.character(path), is.character(ext), length(ext) == 1L)
  ext <- sub("^\\.", "", ext)
  stem <- path_ext_strip(path)
  if (!nzchar(ext)) {
    return(stem)
  }

  out <- paste0(stem, ".", ext)
  # paste0() would turn a missing path into "NA.ext"
  out[is.na(path)] <- NA_character_
  out
}

#' @rdname path_ext_set
#' @export
path_ext_strip <- function(path) {
  stopifnot(is.character(path))
  # Only strip after the last separator so "a.b/c" is left alone
  sub("(?<=[^/\\\\])\\.[^./\\\\]+$", "", path, perl = TRUE)
}

#' @title Get the file extension of a path
#' @description Returns the extension without the leading dot, or `""` when
#'   a path has none. Hidden files such as `.Rprofile` have no extension.
#' @param path A character vector of file paths.
#' @return A character vector the same length as `path`.
#' @export
path_ext <- function(path) {
  stopifnot(is.character(path))
  stem <- path_ext_strip(path)
  ext <- substring(path, nchar(stem) + 2L)
  ext[is.na(path)] <- NA_character_
  ext
}

#' @title Test whether paths are absolute
#' @description Recognises POSIX roots, home-relative paths, Windows drive
#'   letters and UNC shares.
#' @param path A character vector of file paths.
#' @return A logical vector the same length as `path`.
#' @export
path_is_absolute <- function(path) {
  stopifnot(is.character(path))
  out <- grepl("^(/|~|[A-Za-z]:[/\\\\]|\\\\\\\\)", path)
  out[is.na(path)] <- NA
  out
}

#' @title Express a path relative to a base directory
#' @description Both paths are made absolute and normalised first, so
#'   relative inputs are taken relative to the working directory. Paths
#'   outside `base` gain the necessary `..` components.
#' @param path A character vector of file paths.
#' @param base The directory that results should be relative to.
#' @return A character vector of relative paths.
#' @export
#' @examples
#' path_rel_to("/srv/app/data/raw.csv", "/srv/app")
path_rel_to <- function(path, base = getwd()) {
  stopifnot(is.character(path), is.character(base), length(base) == 1L)

  # normalizePath() leaves paths that do not exist untouched, so anchor
  # relative paths and resolve "." and ".." by hand
  split_path <- function(p) {
    relative <- !path_is_absolute(p)
    p[relative] <- file.path(getwd(), p[relative])
    p |>
      normalizePath(winslash = "/", mustWork = FALSE) |>
      strsplit("/", fixed = TRUE) |>
      lapply(\(parts) {
        resolved <- character()
        for (part in parts[nzchar(parts) & parts != "."]) {
          if (part == "..") {
            resolved <- utils::head(resolved, -1L)
          } else {
            resolved <- c(resolved, part)
          }
        }
        resolved
      })
  }
  base_parts <- split_path(base)[[1L]]

  vapply(
    split_path(path),
    function(parts) {
      # Length of the shared prefix
      n <- min(length(parts), length(base_parts))
      common <- 0L
      while (common < n && parts[common + 1L] == base_parts[common + 1L]) {
        common <- common + 1L
      }

      ups <- rep("..", length(base_parts) - common)
      rest <- parts[seq_along(parts) > common]
      rel <- paste(c(ups, rest), collapse = "/")
      if (nzchar(rel)) rel else "."
    },
    character(1)
  )
}

#' @title Make a string safe to use as a file name
#' @description Replaces characters that are invalid on common file systems,
#'   collapses runs of the replacement and trims the result to `max_chars`.
#' @param x A character vector of candidate file names.
#' @param replacement The string substituted for invalid characters.
#' @param max_chars The maximum number of characters to keep.
#' @return A character vector the same length as `x`.
#' @export
#' @examples
#' path_sanitize("Q3 report: draft/final?")
path_sanitize <- function(x, replacement = "_", max_chars = 255L) {
  stopifnot(is.character(x), is.character(replacement), length(replacement) == 1L)
  if (grepl("[/\\\\:*?\"<>|]", replacement)) {
    rlang::abort(
      "`replacement` must not contain characters that are invalid in file names.",
      class = c("pathwren_error_replacement", "pathwren_error")
    )
  }

  out <- gsub("[[:cntrl:]/\\\\:*?\"<>|]", replacement, x)
  out <- gsub("[[:space:]]+", replacement, out)

  # "a: b" becomes "a__b" above; fold repeats down to one replacement
  if (nzchar(replacement)) {
    doubled <- strrep(replacement, 2L)
    while (any(grepl(doubled, out, fixed = TRUE))) {
      out <- gsub(doubled, replacement, out, fixed = TRUE)
    }
  }

  # Leading dots would make the file hidden on Unix
  out <- sub("^[. ]+", "", out)
  out <- substr(out, 1L, max_chars)
  out[!is.na(out) & !nzchar(out)] <- "untitled"
  out
}

#' @title Create a path that does not exist yet
#' @description Appends `_1`, `_2`, ... before the extension until the path
#'   is free. Useful for writing exports without overwriting earlier ones.
#' @param path A single file path.
#' @param max_tries The maximum number of suffixes to try.
#' @return A single file path that does not exist.
#' @export
path_unique <- function(path, max_tries = 1000L) {
  stopifnot(is.character(path), length(path) == 1L)
  if (!file.exists(path)) {
    return(path)
  }

  stem <- path_ext_strip(path)
  ext <- path_ext(path)
  suffix_ext <- if (nzchar(ext)) paste0(".", ext) else ""
  for (i in seq_len(max_tries)) {
    candidate <- paste0(stem, "_", i, suffix_ext)
    if (!file.exists(candidate)) {
      return(candidate)
    }
  }

  rlang::abort(
    sprintf("Could not find a free path for '%s' after %d tries.", path, max_tries),
    class = c("pathwren_error_exhausted", "pathwren_error"),
    path = path
  )
}

#' @title Ensure that a directory exists
#' @description Creates the directory and any missing parents. An error is
#'   signalled if `path` exists but is a regular file.
#' @param path A single directory path.
#' @return `path`, invisibly.
#' @export
path_ensure_dir <- function(path) {
  stopifnot(is.character(path), length(path) == 1L)
  if (dir.exists(path)) {
    return(invisible(path))
  }
  if (file.exists(path)) {
    rlang::abort(
      sprintf("'%s' exists but is not a directory.", path),
      class = c("pathwren_error_not_dir", "pathwren_error"),
      path = path
    )
  }

  ok <- dir.create(path, recursive = TRUE, showWarnings = FALSE)
  if (!ok) {
    rlang::abort(
      sprintf("Could not create directory '%s'.", path),
      class = c("pathwren_error_create", "pathwren_error"),
      path = path
    )
  }
  invisible(path)
}
