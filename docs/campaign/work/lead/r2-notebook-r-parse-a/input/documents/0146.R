.fa_files_root <- function() {
  files <- file.path(.fa_local_root(), "files")
  if (!dir.exists(files)) {
    dir.create(files, recursive = TRUE)
  }
  files
}