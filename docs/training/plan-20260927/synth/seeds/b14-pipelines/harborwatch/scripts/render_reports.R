#!/usr/bin/env Rscript
# Render the reports that are due according to the registry and mail them out.

suppressPackageStartupMessages({
  library(argparse)
  library(rendershed)
})

build.parser <- function() {
  parser <- ArgumentParser(description = "Render due harborwatch reports")
  parser$add_argument("ids", nargs = "*", help = "Report ids to render (default: all due)")
  parser$add_argument("-r", "--registry", default = "reports/registry.yml",
                      help = "Path to the report registry [default: %(default)s]")
  parser$add_argument("-o", "--output-dir", default = "output/reports",
                      help = "Where rendered reports are written [default: %(default)s]")
  parser$add_argument("--state", default = "output/reports/.last_runs.csv",
                      help = "CSV that records the last successful run per report")
  parser$add_argument("--force", action = "store_true",
                      help = "Render the selected reports even if they are not due")
  parser$add_argument("--dry-run", action = "store_true",
                      help = "Render, but only print the emails that would be sent")
  parser
}

read.state <- function(path) {
  if (!file.exists(path)) {
    return(data.frame(id = character(), lastRun = as.POSIXct(character(), tz = "UTC")))
  }
  state <- utils::read.csv(path, stringsAsFactors = FALSE)
  state$lastRun <- as.POSIXct(state$lastRun, tz = "UTC")
  state
}

write.state <- function(state, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  out <- state
  out$lastRun <- format(out$lastRun, "%Y-%m-%d %H:%M:%S", tz = "UTC")
  tmp <- paste0(path, ".tmp")
  utils::write.csv(out, tmp, row.names = FALSE)
  file.rename(tmp, path)
  invisible(path)
}

update.state <- function(state, id, when = Sys.time()) {
  hit <- state$id == id
  if (any(hit)) {
    state$lastRun[hit] <- when
  } else {
    state <- rbind(state, data.frame(id = id, lastRun = when))
  }
  state
}

select.reports <- function(registry, state, ids, force) {
  if (length(ids) > 0) {
    unknown <- setdiff(ids, registry$id)
    if (length(unknown) > 0) {
      stop("Unknown report id(s): ", paste(unknown, collapse = ", "), call. = FALSE)
    }
    registry <- registry[registry$id %in% ids, , drop = FALSE]
  }
  if (force) {
    return(registry)
  }
  dueReports(registry, state)
}

render.one <- function(report, output.dir, dry.run) {
  params <- list(runLabel = format(Sys.Date()), reportId = report$id)
  details <- renderDocument(report$input, output.dir, params = params, format = report$format)
  if (details$status == "ok") {
    notifySuccess(report$id, details, report$recipients, dryRun = dry.run)
  } else {
    notifyFailure(report$id, details, report$recipients, dryRun = dry.run)
  }
  details
}

main <- function() {
  args <- build.parser()$parse_args()
  registry <- readRegistry(args$registry)
  state <- read.state(args$state)
  todo <- select.reports(registry, state, args$ids, args$force)
  if (nrow(todo) == 0) {
    message("No reports are due.")
    return(0L)
  }
  failures <- 0L
  for (i in seq_len(nrow(todo))) {
    report <- todo[i, ]
    message(sprintf("[%d/%d] rendering %s", i, nrow(todo), report$id))
    details <- render.one(report, args$output_dir, args$dry_run)
    if (details$status == "ok") {
      state <- update.state(state, report$id)
    } else {
      failures <- failures + 1L
    }
  }
  if (!args$dry_run) {
    write.state(state, args$state)
  }
  cleanIntermediates(dirname(args$registry))
  if (failures > 0) 1L else 0L
}

if (!interactive()) {
  quit(status = main(), save = "no")
}
