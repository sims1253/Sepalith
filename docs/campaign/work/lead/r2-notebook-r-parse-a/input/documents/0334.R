getTaskFilepath <- function(
  task,
  type,
  ext,
  dirtype,
  subdir = NULL,
  dirCreate = TRUE
) {
  if (missing(dirtype)) {
    stop("Directory type is missing.")
  }
  pz <- getTaskPaths(task)
  if (dirtype %in% names(pz)) {
    wdir <- dirtype
  } else {
    wdir <- list(
      documentation = "doc",
      code = "code",
      data = "data",
      `data source` = "datasrc",
      `binary data` = "bin",
      binary = "bin"
    )
  }
  if (!dirtype %in% names(pz)) {
    stop("unknown directory type")
  }
  path <- pz[[dirtype]]
  if (!is.null(subdir)) {
    path <- file.path(path, subdir)
  }
  if (dirCreate && !file.exists(path)) {
    dir.create(path, showWarnings = FALSE, recursive = TRUE)
  }
  filename <- paste0(task$task, "_", paste(type, collapse = "-"), ".", ext)
  path <- file.path(path, filename)
  return(path)
}