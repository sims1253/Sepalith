#' Create a directory if it does not exist
#'
#' @param path Directory path.
#' @noRd
ensure.dir <- function(path) {
  if (!dir.exists(path)) {
    ok <- dir.create(path, recursive = TRUE, showWarnings = FALSE)
    if (!ok) stop("Could not create directory: ", path, call. = FALSE)
  }
  invisible(normalizePath(path, mustWork = TRUE))
}

#' List log files under a directory
#'
#' Matches plain and gzipped files, sorted from oldest to newest by
#' modification time.
#'
#' @param dir Directory to search.
#' @param stem File name stem, e.g. `"app"` matches `app.log` and `app.log.1.gz`.
#' @param recursive Whether to search subdirectories.
#' @return A character vector of file paths.
#' @export
list.log.files <- function(dir, stem = NULL, recursive = FALSE) {
  rx <- if (is.null(stem)) "\\.log(\\.[0-9]+)?(\\.gz)?$" else {
    paste0("^", gsub(".", "\\.", stem, fixed = TRUE), "\\.log(\\.[0-9]+)?(\\.gz)?$")
  }
  # list.files() matches the pattern against base names only
  files <- list.files(dir, pattern = rx, full.names = TRUE, recursive = recursive)
  if (length(files) == 0L) return(character(0))
  files[order(file.mtime(files))]
}

#' Age of files in days
#'
#' @param paths Character vector of file paths.
#' @param now Reference time.
#' @keywords internal
file.age.days <- function(paths, now = Sys.time()) {
  mtime <- file.mtime(paths)
  age <- as.numeric(difftime(now, mtime, units = "days"))
  stats::setNames(age, basename(paths))
}

#' Rotate a log file by renaming it with a numeric suffix
#'
#' Existing rotated copies are shifted up by one (`.1` becomes `.2`, and so
#' on) and anything beyond `keep` is deleted.
#'
#' @param path Path to the active log file.
#' @param keep Number of rotated copies to keep.
#' @param compress Whether to gzip the newly rotated copy.
#' @return The path of the newest rotated copy, invisibly.
#' @export
rotate.log.file <- function(path, keep = 5L, compress = TRUE) {
  if (!file.exists(path)) stop("No such log file: ", path, call. = FALSE)
  if (keep < 1L) stop("`keep` must be at least 1.", call. = FALSE)

  ext <- if (compress) ".gz" else ""
  rotated <- function(i) paste0(path, ".", i, ext)

  # Make room at the top, then shift the rest up from the oldest down
  if (file.exists(rotated(keep))) unlink(rotated(keep))
  for (i in rev(seq_len(keep - 1L))) {
    if (file.exists(rotated(i))) file.rename(rotated(i), rotated(i + 1L))
  }

  target <- rotated(1L)
  if (compress) {
    gz.copy(path, target)
    unlink(path)
  } else {
    file.rename(path, target)
  }
  # Writers expect the active file to exist again straight away
  file.create(path)
  invisible(target)
}

#' Copy a file into a gzip-compressed file
#'
#' @param from Path to the source file.
#' @param to Path to the gzip file to create.
#' @param chunk.size Number of bytes copied per read.
#' @noRd
gz.copy <- function(from, to, chunk.size = 65536L) {
  src <- file(from, "rb")
  on.exit(close(src), add = TRUE)
  dst <- gzfile(to, "wb")
  on.exit(close(dst), add = TRUE)

  repeat {
    bytes <- readBin(src, what = "raw", n = chunk.size)
    if (length(bytes) == 0L) break
    writeBin(bytes, dst)
  }
  invisible(to)
}

#' Delete log files older than a cutoff
#'
#' @param dir Directory to search.
#' @param max.age.days Files older than this many days are removed.
#' @param dry.run If `TRUE`, only report what would be removed.
#' @return A character vector of removed (or would-be removed) paths, invisibly.
#' @export
prune.old.logs <- function(dir, max.age.days = 30, dry.run = FALSE) {
  files <- list.log.files(dir, recursive = TRUE)
  if (length(files) == 0L) return(invisible(character(0)))

  # ages are measured from modification time, not creation
  age <- file.age.days(files)
  old <- files[age > max.age.days]

  if (dry.run) {
    message(sprintf("Would remove %d file(s) from %s", length(old), dir))
  } else if (length(old) > 0L) {
    unlink(old)
    removed <- !file.exists(old)
    message(sprintf("Removed %d of %d file(s) from %s", sum(removed), length(old), dir))
  }
  invisible(old)
}

#' Summarise disk usage of log files
#'
#' @param dir Directory to search.
#' @param recursive Whether to search subdirectories.
#' @return A data frame with one row per stem and total size in megabytes.
#' @export
log.disk.usage <- function(dir, recursive = TRUE) {
  files <- list.log.files(dir, recursive = recursive)
  if (length(files) == 0L) {
    return(data.frame(stem = character(0), files = integer(0), size.mb = numeric(0)))
  }

  # app.log, app.log.1 and app.log.2.gz all belong to the stem "app"
  stem <- sub("\\.log(\\.[0-9]+)?(\\.gz)?$", "", basename(files))

  size <- file.size(files) / 1024^2
  out <- data.frame(
    stem = names(tapply(size, stem, sum)),
    files = as.integer(tapply(size, stem, length)),
    size.mb = round(as.numeric(tapply(size, stem, sum)), 2)
  )
  out[order(-out$size.mb), , drop = FALSE]
}

#' Move old log files into a dated archive tree
#'
#' Files older than `older.than.days` are compressed (unless already gzipped)
#' into `archive.dir/YYYY/MM/`, named after their modification date. The
#' original file is removed only after the archive copy has been verified, and
#' an existing archive file is never overwritten.
#'
#' @param dir Directory to search.
#' @param archive.dir Root directory of the archive.
#' @param older.than.days Minimum age in days for a file to be archived.
#' @param dry.run If `TRUE`, only report what would be archived.
#' @return A data frame with columns `from`, `to` and `ok`, invisibly.
#' @export
archive.logs <- function(dir, archive.dir, older.than.days = 7, dry.run = FALSE) {
  files <- list.log.files(dir, recursive = FALSE)
  files <- files[file.age.days(files) > older.than.days]
  result <- data.frame(from = files, to = character(length(files)), ok = logical(length(files)))
  if (length(files) == 0L) return(invisible(result))

  mtime <- file.mtime(files)
  for (i in seq_along(files)) {
    sub.dir <- file.path(archive.dir, format(mtime[i], "%Y"), format(mtime[i], "%m"))
    stamp <- format(mtime[i], "%Y%m%d-%H%M%S")
    base <- sub("\\.gz$", "", basename(files[i]))
    target <- file.path(sub.dir, paste0(base, ".", stamp, ".gz"))
    result$to[i] <- target
    if (dry.run) next

    ensure.dir(sub.dir)
    if (file.exists(target)) {
      # an earlier run already archived a file with this name and stamp
      copied <- FALSE
    } else if (grepl("\\.gz$", files[i])) {
      copied <- file.copy(files[i], target, overwrite = FALSE)
    } else {
      gz.copy(files[i], target)
      copied <- file.exists(target)
    }

    # never delete the source unless the archive copy is non-empty
    result$ok[i] <- isTRUE(copied) && isTRUE(file.size(target) > 0)
    if (result$ok[i]) unlink(files[i])
  }

  if (dry.run) {
    message(sprintf("Would archive %d file(s) to %s", nrow(result), archive.dir))
  } else if (!all(result$ok)) {
    warning(sprintf("%d file(s) could not be archived", sum(!result$ok)), call. = FALSE)
  }
  invisible(result)
}
