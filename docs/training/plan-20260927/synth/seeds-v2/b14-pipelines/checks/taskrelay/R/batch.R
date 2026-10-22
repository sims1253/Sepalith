dir <- tempfile("inbox-")
dir.create(dir)
for (i in 1:3) {
  writeLines(c("id,value", paste0(i, ",", i * 10)), file.path(dir, sprintf("in_%02d.csv", i)))
}
file.create(file.path(dir, "empty.csv"))

# list_batch_files()
files <- list_batch_files(dir)
stopifnot(
  identical(basename(files), c("in_01.csv", "in_02.csv", "in_03.csv")),
  inherits(try(list_batch_files(file.path(dir, "nope")), silent = TRUE), "try-error")
)

# batch_manifest()
man <- batch_manifest(files)
stopifnot(
  nrow(man) == 3L,
  all(nchar(man$md5) == 32L),
  nrow(batch_manifest(character())) == 0L
)

# process_one_file()
warn_fun <- function(path) {
  warning("odd")
  "value kept"
}
ok <- process_one_file(files[1], function(p) nrow(utils::read.csv(p)))
wr <- process_one_file(files[1], warn_fun)
er <- process_one_file(files[1], function(p) stop("bad"))
stopifnot(
  identical(ok$status, "ok"), identical(ok$value, 1L),
  identical(wr$status, "warning"), identical(wr$value, "value kept"),
  identical(wr$message, "odd"),
  identical(er$status, "error"), is.null(er$value)
)

# run_batch() and summarise_batch()
fun <- function(path) {
  if (grepl("02", basename(path))) stop("bad row")
  nrow(utils::read.csv(path))
}
res <- run_batch(files, fun)
res_stop <- run_batch(files, fun, stop_on_error = TRUE)
smry <- summarise_batch(res)
stopifnot(
  inherits(res, "taskrelay_batch"),
  identical(smry$status, c("ok", "error", "ok")),
  length(res_stop) == 2L,
  inherits(try(run_batch(files, "nrow"), silent = TRUE), "try-error")
)

# archive_processed() and quarantine_failed()
archived <- archive_processed(smry, file.path(dir, "archive"))
quarantined <- quarantine_failed(smry, file.path(dir, "quarantine"))
stopifnot(
  identical(basename(archived), c("in_01.csv", "in_03.csv")),
  identical(basename(quarantined), "in_02.csv"),
  !any(file.exists(files))
)

# write_batch_report()
paths <- write_batch_report(smry, file.path(dir, "report", "batch.csv"))
paths2 <- write_batch_report(smry, file.path(dir, "report2", "batch.txt"))
stopifnot(
  all(file.exists(paths)),
  grepl("^ok\\s+2$", readLines(paths[["txt"]])[2]),
  paths2[["txt"]] != paths2[["csv"]],
  nrow(utils::read.csv(paths2[["csv"]])) == 3L
)
unlink(dir, recursive = TRUE)
