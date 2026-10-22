#' List input files for a batch
#'
#' Finds files under `dir` that match `pattern`, skipping empty files and
#' anything smaller than `min_size` bytes.
#'
#' @export
#' @param dir Directory to search.
#' @param pattern A regular expression matched against file names.
#' @param recursive Whether to descend into subdirectories.
#' @param min_size Minimum file size in bytes.
#' @return A sorted character vector of absolute file paths.
list_batch_files <- function(dir, pattern = "\\.csv$", recursive = FALSE, min_size = 1) {
  if (!dir.exists(dir)) {
    stop("Input directory does not exist: ", dir, call. = FALSE)
  }

  files <- list.files(dir, pattern = pattern, recursive = recursive, full.names = TRUE)
  files <- normalizePath(files, winslash = "/", mustWork = TRUE)
  # Exporters often create the file before writing to it; skip those for now
  sizes <- file.size(files)
  files <- files[!is.na(sizes) & sizes >= min_size]
  return(sort(files))
}

#' Build a manifest of batch inputs
#'
#' Records size, modification time and an MD5 checksum for each file so a
#' later run can tell which inputs changed.
#'
#' @export
#' @param files A character vector of file paths.
#' @return A data frame with one row per file.
batch_manifest <- function(files) {
  if (length(files) == 0L) {
    return(data.frame(
      path = character(), size = numeric(), mtime = as.POSIXct(character()),
      md5 = character(), stringsAsFactors = FALSE
    ))
  }

  info <- file.info(files, extra_cols = FALSE)
  manifest <- data.frame(
    path = files,
    size = info$size,
    mtime = info$mtime,
    md5 = unname(tools::md5sum(files)),
    stringsAsFactors = FALSE
  )
  rownames(manifest) <- NULL
  return(manifest)
}

#' Process a single file and capture the outcome
#'
#' Calls `fun` on `path`, catching errors so that one bad file does not abort
#' the batch. Warnings are recorded and muffled but do not interrupt `fun`,
#' so its value is kept. Timing is recorded for every call.
#'
#' @keywords internal
#' @param path Path to one input file.
#' @param fun A function taking a file path as its first argument.
#' @param logger A logger created by [new_logger()], or `NULL`.
#' @param ... Passed to `fun`.
#' @return A list with elements `status`, `value`, `message`, `path` and
#'   `elapsed`.
process_one_file <- function(path, fun, logger = NULL, ...) {
  started <- Sys.time()
  warnings <- character()

  # withCallingHandlers() lets `fun` run to completion after a warning;
  # tryCatch() alone would unwind and lose the value
  outcome <- tryCatch(
    withCallingHandlers(
      list(status = "ok", value = fun(path, ...), message = NA_character_),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      list(status = "error", value = NULL, message = conditionMessage(e))
    }
  )
  if (identical(outcome$status, "ok") && length(warnings) > 0L) {
    outcome$status <- "warning"
    outcome$message <- paste(warnings, collapse = "; ")
  }

  outcome$path <- path
  outcome$elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  if (!is.null(logger) && outcome$status != "ok") {
    log_warn(logger, "%s on %s: %s", outcome$status, basename(path), outcome$message)
  }
  return(outcome)
}

#' Run a function over a batch of files
#'
#' Applies `fun` to every file, logging progress, and optionally stops at
#' the first error. Results are collected in input order.
#'
#' @export
#' @param files A character vector of file paths.
#' @param fun A function taking a file path as its first argument.
#' @param logger A logger created by [new_logger()], or `NULL`.
#' @param stop_on_error Whether to abort the batch on the first error.
#' @param progress_every Log a progress line after this many files.
#' @param ... Passed to `fun`.
#' @return A list of per-file outcomes with class `taskrelay_batch`.
run_batch <- function(files, fun, logger = NULL, stop_on_error = FALSE,
                      progress_every = 25L, ...) {
  if (!is.function(fun)) {
    stop("`fun` must be a function.", call. = FALSE)
  }

  n <- length(files)
  results <- vector("list", n)
  names(results) <- basename(files)
  if (!is.null(logger)) {
    log_info(logger, "Starting batch of %d file(s)", n)
  }

  for (i in seq_len(n)) {
    results[[i]] <- process_one_file(files[[i]], fun, logger = logger, ...)
    if (stop_on_error && identical(results[[i]]$status, "error")) {
      if (!is.null(logger)) {
        log_error(logger, "Aborting batch after failure on %s", basename(files[[i]]))
      }
      # Drop the unfilled slots so callers only see files that were tried
      results <- results[seq_len(i)]
      break
    }
    if (!is.null(logger) && i %% progress_every == 0L) {
      log_info(logger, "Processed %d/%d file(s)", i, n)
    }
  }

  statuses <- vapply(results, function(r) r$status, character(1))
  if (!is.null(logger)) {
    log_info(
      logger, "Batch finished: %d ok, %d warning, %d error",
      sum(statuses == "ok"), sum(statuses == "warning"), sum(statuses == "error")
    )
  }

  class(results) <- c("taskrelay_batch", "list")
  return(results)
}

#' Summarise a batch run
#'
#' Flattens the per-file outcomes of [run_batch()] into a data frame.
#'
#' @export
#' @param results The value returned by [run_batch()].
#' @return A data frame with columns `file`, `status`, `message` and
#'   `elapsed`.
summarise_batch <- function(results) {
  n <- length(results)
  file <- character(n)
  status <- character(n)
  message <- character(n)
  elapsed <- numeric(n)

  for (i in seq_len(n)) {
    file[i] <- results[[i]]$path
    status[i] <- results[[i]]$status
    message[i] <- results[[i]]$message
    elapsed[i] <- results[[i]]$elapsed
  }

  return(data.frame(
    file = file, status = status, message = message, elapsed = elapsed,
    stringsAsFactors = FALSE
  ))
}

#' Move successfully processed files to an archive
#'
#' Files are moved into a dated subdirectory of `archive_dir` so that the
#' inbox only holds work that is still pending.
#'
#' @export
#' @param summary A data frame returned by [summarise_batch()].
#' @param archive_dir Directory that receives the archived files.
#' @param overwrite Whether to replace files that already exist there.
#' @return The new file paths, invisibly.
#' @rdname move_batch_files
archive_processed <- function(summary, archive_dir, overwrite = FALSE) {
  done <- summary$file[summary$status == "ok"]
  target_dir <- file.path(archive_dir, format(Sys.Date(), "%Y-%m-%d"))
  dir.create(target_dir, recursive = TRUE, showWarnings = FALSE)

  # Copy then remove instead of file.rename(), which fails across devices;
  # originals are only deleted once their copy succeeded
  targets <- file.path(target_dir, basename(done))
  copied <- file.copy(done, targets, overwrite = overwrite)
  file.remove(done[copied])
  return(invisible(targets[copied]))
}

#' Move failed files to a quarantine directory
#'
#' Files are moved into a dated subdirectory of `quarantine_dir` so that
#' they are not retried until someone has looked at them.
#'
#' @export
#' @param summary A data frame returned by [summarise_batch()].
#' @param quarantine_dir Directory that receives the failed files.
#' @param overwrite Whether to replace files that already exist there.
#' @rdname move_batch_files
quarantine_failed <- function(summary, quarantine_dir, overwrite = FALSE) {
  failed <- summary$file[summary$status == "error"]
  target_dir <- file.path(quarantine_dir, format(Sys.Date(), "%Y-%m-%d"))
  dir.create(target_dir, recursive = TRUE, showWarnings = FALSE)

  targets <- file.path(target_dir, basename(failed))
  copied <- file.copy(failed, targets, overwrite = overwrite)
  file.remove(failed[copied])
  return(invisible(targets[copied]))
}

#' Write a batch report to disk
#'
#' Saves the batch summary as CSV next to a small plain-text overview that
#' is easy to attach to a notification. The overview shares the CSV's path
#' with a `.txt` extension.
#'
#' @export
#' @param summary A data frame returned by [summarise_batch()].
#' @param path Path of the CSV file to write.
#' @return The paths of both written files, invisibly.
write_batch_report <- function(summary, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(summary, path, row.names = FALSE)

  # Fixed levels so statuses that did not occur still get a zero line
  counts <- table(factor(summary$status, levels = c("ok", "warning", "error")))
  overview <- c(
    sprintf("Batch report written %s", format(Sys.time(), "%Y-%m-%d %H:%M")),
    sprintf("%-8s %d", names(counts), as.integer(counts)),
    sprintf("Total time: %.1f s", sum(summary$elapsed))
  )

  # Build the name from the stem, so a path without .csv cannot end up
  # overwriting the CSV we just wrote
  txt_path <- paste0(tools::file_path_sans_ext(path), ".txt")
  if (identical(txt_path, path)) {
    txt_path <- paste0(path, ".txt")
  }
  writeLines(overview, txt_path)
  return(invisible(c(csv = path, txt = txt_path)))
}
