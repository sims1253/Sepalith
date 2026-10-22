#!/usr/bin/env Rscript
# Validate new arrival exports in the inbox and stage them for the pipeline.

suppressPackageStartupMessages({
  library(optparse)
  library(taskrelay)
})

parser <- OptionParser(usage = "%prog [options] INBOX")
parser <- add_option(parser, c("-s", "--staging"), default = "data/staging",
                     help = "Directory for validated files [default: %default]")
parser <- add_option(parser, c("-a", "--archive"), default = "data/archive",
                     help = "Directory for processed originals [default: %default]")
parser <- add_option(parser, c("-q", "--quarantine"), default = "data/quarantine",
                     help = "Directory for files that failed validation [default: %default]")
parser <- add_option(parser, c("-l", "--log-file"), default = "logs/process_inbox.log",
                     dest = "log_file", help = "Log file path [default: %default]")
parser <- add_option(parser, c("-j", "--jobs"), type = "integer", default = 1L,
                     help = "Number of files to validate in parallel [default: %default]")
parser <- add_option(parser, "--verbose", action = "store_true", default = FALSE,
                     help = "Log debug messages")

make_run_logger <- function(log_file, verbose = FALSE) {
  new_logger(
    name = "process_inbox",
    threshold = if (verbose) "DEBUG" else "INFO",
    appenders = list(appender_console(), appender_file(log_file, min_level = "INFO"))
  )
}

validate_arrivals_file <- function(path, staging) {
  header <- readLines(path, n = 1L, warn = FALSE)
  fields <- tolower(trimws(strsplit(header, ",", fixed = TRUE)[[1]]))
  required <- c("call_id", "vessel", "berth", "status", "eta", "ata", "atd")
  missing <- setdiff(required, fields)
  if (length(missing) > 0) {
    stop("missing column(s) ", paste(missing, collapse = ", "))
  }
  n_rows <- length(readLines(path, warn = FALSE)) - 1L
  if (n_rows == 0L) {
    warning("header only, no records")
  }
  stamp <- format(file.mtime(path), "%Y%m%d%H%M%S")
  target <- file.path(staging, sprintf("arrivals_%s_%s", stamp, basename(path)))
  file.copy(path, target, overwrite = TRUE)
  n_rows
}

validate_parallel <- function(files, staging, jobs, logger) {
  log_info(logger, "Validating %d file(s) with %d worker(s)", length(files), jobs)
  outcomes <- par_map(
    files,
    function(path) process_one_file(path, validate_arrivals_file, staging = staging),
    backend = if (jobs > 1L) "mclapply" else "sequential",
    workers = jobs
  )
  errors <- collect_errors(outcomes)
  for (key in names(errors)) {
    log_error(logger, "Worker crashed on %s: %s", key, errors[[key]])
  }
  outcomes[!vapply(outcomes, inherits, logical(1), what = "error")]
}

log_outcomes <- function(summary, logger) {
  for (i in seq_len(nrow(summary))) {
    row <- summary[i, ]
    if (row$status == "ok") {
      log_info(logger, "Staged %s (%.2f s)", basename(row$file), row$elapsed)
    } else {
      log_warn(logger, "%s: %s (%s)", basename(row$file), row$status, row$message)
    }
  }
  invisible(summary)
}

main <- function(argv = commandArgs(trailingOnly = TRUE)) {
  args <- parse_args(parser, args = argv, positional_arguments = 1L)
  opts <- args$options
  inbox <- args$args[[1]]
  logger <- make_run_logger(opts$log_file, verbose = opts$verbose)
  files <- list_batch_files(inbox, pattern = "\\.csv$")
  if (length(files) == 0L) {
    log_info(logger, "Inbox %s is empty; nothing to do", inbox)
    return(0L)
  }
  dir.create(opts$staging, recursive = TRUE, showWarnings = FALSE)
  if (opts$jobs > 1L) {
    results <- validate_parallel(files, opts$staging, opts$jobs, logger)
  } else {
    results <- run_batch(files, validate_arrivals_file, logger = logger, staging = opts$staging)
  }
  summary <- summarise_batch(results)
  log_outcomes(summary, logger)
  archive_processed(summary, opts$archive)
  quarantine_failed(summary, opts$quarantine)
  report <- write_batch_report(summary, file.path("logs", "inbox", "last_run.csv"))
  report_txt <- report[["txt"]]
  log_info(logger, "Wrote batch report to %s", report_txt)
  if (any(summary$status == "error")) 1L else 0L
}

if (!interactive()) {
  quit(status = main(), save = "no")
}
