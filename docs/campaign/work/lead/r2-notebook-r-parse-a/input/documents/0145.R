.fa_local_root <- function() {
  env_root <- Sys.getenv("FAASR_DATA_ROOT", unset = "")
  if (nzchar(env_root)) {
    root <- env_root
  } else {
    root <- file.path(getwd(), "faasr_data")
  }
  if (!dir.exists(root)) {
    dir.create(root, recursive = TRUE)
  }
  root
}