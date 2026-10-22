#' Take a snapshot of the files below a directory
#'
#' @param path Path to the directory to watch.
#' @param pattern Regular expression selecting the files to include.
#' @param use.hash Whether to record an MD5 hash of each file, which detects
#'   changes that keep the size and modification time.
#' @return A data frame with columns `file`, `mtime`, `size` and `hash`.
#' @export
take.snapshot <- function(path = ".", pattern = "\\.(R|Rmd|md)$", use.hash = FALSE) {
  files <- list.files(path, pattern = pattern, recursive = TRUE, full.names = TRUE)
  info <- file.info(files, extra_cols = FALSE)
  hash <- if (use.hash && length(files) > 0L) {
    unname(tools::md5sum(files))
  } else {
    rep(NA_character_, length(files))
  }
  data.frame(
    file = files,
    mtime = info$mtime,
    size = info$size,
    hash = hash,
    stringsAsFactors = FALSE
  )
}

#' Compare two snapshots
#'
#' @param old,new Snapshots as returned by [take.snapshot()].
#' @return A list with character vectors `added`, `removed` and `modified`.
#' @export
changes.between <- function(old, new) {
  added <- setdiff(new$file, old$file)
  removed <- setdiff(old$file, new$file)
  common <- intersect(old$file, new$file)
  a <- old[match(common, old$file), , drop = FALSE]
  b <- new[match(common, new$file), , drop = FALSE]
  touched <- a$mtime != b$mtime | a$size != b$size
  hashed <- !is.na(a$hash) & !is.na(b$hash)
  touched[hashed] <- a$hash[hashed] != b$hash[hashed]
  list(added = added, removed = removed, modified = common[touched])
}

#' Test whether a change set is empty
#'
#' @param changes A change set as returned by [changes.between()].
#' @return A single logical.
#' @keywords internal
has.changes <- function(changes) {
  sum(lengths(changes)) > 0L
}

#' Report added files
#'
#' @param files A character vector of file paths.
#' @return `files`, invisibly.
#' @family change reporters
#' @export
report.added <- function(files) {
  if (length(files) > 0L) {
    cat(sprintf("[%s] + %s\n", format(Sys.time(), "%H:%M:%S"), files), sep = "")
  }
  invisible(files)
}

#' Report removed files
#'
#' @param files A character vector of file paths.
#' @return `files`, invisibly.
#' @family change reporters
#' @export
report.removed <- function(files) {
  if (length(files) > 0L) {
    cat(sprintf("[%s] - %s\n", format(Sys.time(), "%H:%M:%S"), files), sep = "")
  }
  invisible(files)
}

#' Report modified files
#'
#' @param files A character vector of file paths.
#' @return `files`, invisibly.
#' @family change reporters
#' @export
report.modified <- function(files) {
  if (length(files) > 0L) {
    cat(sprintf("[%s] ~ %s\n", format(Sys.time(), "%H:%M:%S"), files), sep = "")
  }
  invisible(files)
}

#' Report every part of a change set
#'
#' @param changes A change set as returned by [changes.between()].
#' @return `changes`, invisibly.
#' @family change reporters
#' @export
report.changes <- function(changes) {
  report.added(changes$added)
  report.removed(changes$removed)
  report.modified(changes$modified)
  invisible(changes)
}

#' Watch a directory and react to changes
#'
#' Polls the directory every `interval` seconds and calls `on.change` with
#' the change set whenever something was added, removed or modified. Errors
#' raised by `on.change` are reported and do not stop the watcher.
#'
#' @param path Path to the directory to watch.
#' @param on.change A function called with a change set as returned by
#'   [changes.between()].
#' @param interval Seconds to wait between polls.
#' @param timeout Seconds after which to stop watching, or `Inf`.
#' @param ... Passed to [take.snapshot()].
#' @return The number of change sets handled, invisibly.
#' @export
watch.files <- function(path = ".", on.change = report.changes, interval = 1,
                        timeout = Inf, ...) {
  stopifnot(is.function(on.change), interval > 0)
  started <- Sys.time()
  previous <- take.snapshot(path, ...)
  handled <- 0L
  message("Watching '", normalizePath(path), "'. Press Esc or Ctrl+C to stop.")
  repeat {
    elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
    if (elapsed >= timeout) {
      break
    }
    Sys.sleep(interval)
    current <- take.snapshot(path, ...)
    changes <- changes.between(previous, current)
    if (has.changes(changes)) {
      handled <- handled + 1L
      tryCatch(
        on.change(changes),
        error = function(e) {
          message("on.change failed: ", conditionMessage(e))
        },
        warning = function(w) {
          message("on.change warned: ", conditionMessage(w))
        }
      )
    }
    previous <- current
  }
  invisible(handled)
}

#' Build a change handler that runs a shell command
#'
#' @param command The command to run, e.g. `"Rscript"`.
#' @param args A character vector of arguments for `command`.
#' @return A function suitable for the `on.change` argument of [watch.files()].
#' @export
run.on.change <- function(command, args = character()) {
  force(command)
  force(args)
  function(changes) {
    report.changes(changes)
    status <- system2(command, args)
    if (status != 0L) {
      warning("`", command, "` exited with status ", status, ".", call. = FALSE)
    }
    invisible(status)
  }
}
