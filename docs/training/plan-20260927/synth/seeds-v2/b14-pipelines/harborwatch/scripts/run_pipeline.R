#!/usr/bin/env Rscript
# Run the harborwatch targets pipeline from cron or by hand.

suppressPackageStartupMessages({
  library(optparse)
})

option_list <- list(
  make_option(c("-p", "--profile"), default = "default",
              help = "Config profile from config.yml [default: %default]"),
  make_option(c("-n", "--names"), default = NULL,
              help = "Comma-separated target names to build (default: all)"),
  make_option(c("-w", "--workers"), type = "integer", default = NULL,
              help = "Override the number of crew workers"),
  make_option("--reporter", default = "summary",
              help = "targets reporter: silent, summary, verbose [default: %default]"),
  make_option("--invalidate", default = NULL,
              help = "Comma-separated targets to invalidate before running"),
  make_option("--skip-renv-check", action = "store_true", default = FALSE,
              dest = "skip_renv_check", help = "Do not verify the renv lockfile"),
  make_option("--dry-run", action = "store_true", default = FALSE, dest = "dry_run",
              help = "Show outdated targets and exit")
)

split_csv <- function(x) {
  if (is.null(x) || !nzchar(x)) {
    return(NULL)
  }
  # "a, b,,c" -> c("a", "b", "c")
  out <- trimws(strsplit(x, ",", fixed = TRUE)[[1]])
  out[nzchar(out)]
}

load_settings <- function(profile, workers = NULL) {
  if (!file.exists("config.yml")) {
    stop("config.yml not found; run this script from the project root.", call. = FALSE)
  }
  # _targets.R reads the profile from the environment, so set it for the
  # pipeline as well as for our own config::get() call
  Sys.setenv(R_CONFIG_ACTIVE = profile)
  settings <- config::get(file = "config.yml", config = profile)
  if (!is.null(workers)) {
    settings$workers <- workers
  }

  required <- c("inbox", "berth_register", "tide_feed", "output_dir", "timezone")
  missing <- setdiff(required, names(settings))
  if (length(missing) > 0) {
    stop("Profile '", profile, "' is missing: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  settings
}

check_renv <- function(strict = TRUE) {
  if (!file.exists("renv.lock")) {
    message("No renv.lock found; skipping library check.")
    return(invisible(TRUE))
  }

  status <- renv::status(project = ".")
  if (isTRUE(status$synchronized)) {
    return(invisible(TRUE))
  }

  msg <- "Project library is out of sync with renv.lock; run renv::restore()."
  if (strict) {
    stop(msg, call. = FALSE)
  }
  warning(msg, call. = FALSE)
  invisible(FALSE)
}

report_outdated <- function(names) {
  outdated <- if (is.null(names)) {
    targets::tar_outdated()
  } else {
    targets::tar_outdated(names = tidyselect::any_of(names))
  }

  if (length(outdated) == 0) {
    cat("Everything is up to date.\n")
  } else {
    cat("Outdated targets:\n", paste0("  - ", outdated, "\n"), sep = "")
  }
  invisible(outdated)
}

summarise_run <- function(started) {
  # Only targets built during this run; older metadata is left alone
  meta <- targets::tar_meta(fields = c("name", "seconds", "error", "time"))
  meta <- meta[!is.na(meta$time) & meta$time >= started, , drop = FALSE]
  errored <- meta$name[!is.na(meta$error)]

  cat(sprintf(
    "Built %d target(s) in %.1f s.\n",
    nrow(meta), as.numeric(difftime(Sys.time(), started, units = "secs"))
  ))
  if (length(errored) > 0) {
    cat("Errored targets: ", paste(errored, collapse = ", "), "\n", sep = "")
  }

  length(errored) == 0
}

main <- function(argv = commandArgs(trailingOnly = TRUE)) {
  opts <- parse_args(OptionParser(option_list = option_list), args = argv)
  settings <- load_settings(opts$profile, workers = opts$workers)

  # A stale library on the server is a hard error; on a dev box just warn
  if (!opts$skip_renv_check) {
    check_renv(strict = !identical(opts$profile, "dev"))
  }

  names <- split_csv(opts$names)
  if (opts$dry_run) {
    report_outdated(names)
    return(invisible(0L))
  }
  invalidate <- split_csv(opts$invalidate)
  if (!is.null(invalidate)) {
    targets::tar_invalidate(tidyselect::any_of(invalidate))
  }

  dir.create(settings$output_dir, recursive = TRUE, showWarnings = FALSE)
  started <- Sys.time()
  ok <- tryCatch(
    {
      targets::tar_make(
        names = if (is.null(names)) NULL else tidyselect::any_of(names),
        reporter = opts$reporter
      )
      TRUE
    },
    error = function(e) {
      message("Pipeline failed: ", conditionMessage(e))
      FALSE
    }
  )

  # tar_make() can return normally with errored targets under
  # error = "abridge", so the metadata has the final say
  ok <- summarise_run(started) && ok
  invisible(if (ok) 0L else 1L)
}

if (!interactive()) {
  quit(status = main(), save = "no")
}
