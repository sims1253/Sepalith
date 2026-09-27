.fa_logs_root <- function() {
  logs <- file.path(.fa_local_root(), "logs")
  if (!dir.exists(logs)) {
    dir.create(logs, recursive = TRUE)
  }
  logs
}