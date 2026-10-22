root <- tempfile("logsift-check-")
logs <- ensure.dir(file.path(root, "logs"))
stopifnot(dir.exists(logs), identical(ensure.dir(logs), logs))

# list.log.files and file.age.days
writeLines("a", file.path(logs, "app.log"))
writeLines("b", file.path(logs, "web.log"))
writeLines("c", file.path(logs, "notes.txt"))
Sys.setFileTime(file.path(logs, "web.log"), Sys.time() - 10 * 86400)
files <- list.log.files(logs)
stopifnot(length(files) == 2L, basename(files[1]) == "web.log")
stopifnot(identical(basename(list.log.files(logs, stem = "app")), "app.log"))
age <- file.age.days(files)
stopifnot(identical(names(age), c("web.log", "app.log")), age[["web.log"]] > 9)

# gz.copy and rotate.log.file
gz <- gz.copy(file.path(logs, "app.log"), file.path(root, "copy.gz"))
stopifnot(identical(readLines(gzfile(gz)), "a"))
target <- rotate.log.file(file.path(logs, "app.log"), keep = 2L)
stopifnot(basename(target) == "app.log.1.gz", file.exists(file.path(logs, "app.log")))
writeLines("a2", file.path(logs, "app.log"))
rotate.log.file(file.path(logs, "app.log"), keep = 2L)
stopifnot(file.exists(file.path(logs, "app.log.2.gz")))
stopifnot(identical(readLines(gzfile(file.path(logs, "app.log.1.gz"))), "a2"))

# log.disk.usage
usage <- log.disk.usage(logs)
stopifnot(identical(sort(usage$stem), c("app", "web")), sum(usage$files) == 4L)
stopifnot(nrow(log.disk.usage(file.path(root, "empty-none"))) == 0L)

# archive.logs (dry run first, then real) and prune.old.logs
arch <- file.path(root, "archive")
dry <- suppressMessages(archive.logs(logs, arch, older.than.days = 5, dry.run = TRUE))
stopifnot(nrow(dry) == 1L, !dry$ok, file.exists(file.path(logs, "web.log")))
done <- archive.logs(logs, arch, older.than.days = 5)
stopifnot(done$ok, file.exists(done$to), !file.exists(file.path(logs, "web.log")))
old <- file.path(logs, "app.log.2.gz")
Sys.setFileTime(old, Sys.time() - 40 * 86400)
would <- suppressMessages(prune.old.logs(logs, dry.run = TRUE))
stopifnot(identical(basename(would), "app.log.2.gz"), file.exists(old))
suppressMessages(prune.old.logs(logs))
stopifnot(!file.exists(old))
unlink(root, recursive = TRUE)
