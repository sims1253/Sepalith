datasourceFn <- function(task, filename, subdir = ".", dirCreate = TRUE) {
  if (is.null(subdir)) {
    te <- file.path(getTaskPaths(task)$datasrc, filename)
  } else {
    te <- file.path(getTaskPaths(task)$datasrc, subdir, filename)
  }
  if (dirCreate && !file.exists(dirname(te))) {
    dir.create(dirname(te), showWarnings = FALSE, recursive = TRUE)
  }
  te
}