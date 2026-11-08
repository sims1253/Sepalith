fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(isTRUE(coerceDcfValue("Yes")), identical(coerceDcfValue("12"), 12L))
stopifnot(identical(coerceDcfValue("1.5"), 1.5), identical(coerceDcfValue(" a b "), "a b"))
stopifnot(identical(coerceDcfValue("2024-05-01"), as.Date("2024-05-01")))
stopifnot(identical(coerceDcfValue("a, b,"), c("a", "b")))
stopifnot(identical(coerceDcfValue("no", "logical"), FALSE))
stopifnot(identical(coerceDcfValue("7", "character"), "7"))
stopifnot(identical(coerceDcfValue("7", "numeric"), 7))
stopifnot(identical(coerceDcfValue("7", "integer"), 7L))
stopifnot(identical(coerceDcfValue("x", "list"), "x"))

rec <- c(Name = "a", Retries = "3", Note = NA)
l <- dcfRecordToList(rec, c(Retries = "numeric"))
stopifnot(identical(names(l), c("Name", "Retries")), identical(l$Retries, 3))

path <- tempfile(fileext = ".dcf")
cfg <- list(backup = list(Command = "run backup", Schedule = "daily", Retries = 2L,
                          Enabled = TRUE, After = c("fetch", "clean")),
            report = list(Command = "make", Schedule = "weekly", Enabled = FALSE))
stopifnot(identical(writeDcfConfig(cfg, path), path))
back <- readDcfConfig(path)
stopifnot(identical(names(back), c("backup", "report")), identical(back$backup$Retries, 2L))
stopifnot(isTRUE(back$backup$Enabled), identical(back$report$Enabled, FALSE))
stopifnot(is.null(back$report$Retries), identical(back$backup$After, c("fetch", "clean")))
stopifnot(fails(readDcfConfig(tempfile())), fails(readDcfConfig(path, key = "Owner")))

stopifnot(length(validateJobRecord(back$backup)) == 0L)
p <- validateJobRecord(list(Name = "9x", Command = " ", Schedule = "often", Retries = -1,
                            Timeout = 0, Enabled = "maybe"))
stopifnot(length(p) == 6L, grepl("'Name'", p[[1]]), grepl("'Enabled'", p[[6]]))
p <- validateJobRecord(list())
stopifnot(identical(p, c("Field 'Name' is missing.", "Field 'Command' is missing.",
                         "Field 'Schedule' is missing.")))

s <- summariseJobs(back)
stopifnot(identical(s$name, c("backup", "report")), identical(s$schedule, c("daily", "weekly")))
stopifnot(identical(s$retries, c(2L, NA)), all(is.na(s$timeout)))
stopifnot(identical(s$enabled, c(TRUE, FALSE)), identical(s$after, c("fetch, clean", "")))
