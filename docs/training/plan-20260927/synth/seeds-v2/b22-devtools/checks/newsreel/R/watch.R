dir <- tempfile("watch")
dir.create(dir)
writeLines("a <- 1", file.path(dir, "a.R"))
writeLines("b", file.path(dir, "notes.txt"))

# take.snapshot
s1 <- take.snapshot(dir, use.hash = TRUE)
stopifnot(nrow(s1) == 1L, basename(s1$file) == "a.R", !is.na(s1$hash))
stopifnot(is.na(take.snapshot(dir)$hash))

# changes.between / has.changes
writeLines("a <- 2", file.path(dir, "a.R"))
writeLines("# new", file.path(dir, "b.md"))
s2 <- take.snapshot(dir, use.hash = TRUE)
ch <- changes.between(s1, s2)
stopifnot(identical(basename(ch$added), "b.md"), length(ch$removed) == 0L)
stopifnot(identical(basename(ch$modified), "a.R"), has.changes(ch))
stopifnot(!has.changes(changes.between(s2, s2)))

# report.added / report.removed / report.modified / report.changes
out <- utils::capture.output(res <- report.added(c("x.R", "y.R")))
stopifnot(length(out) == 2L, grepl("] + x.R", out[1], fixed = TRUE))
stopifnot(identical(res, c("x.R", "y.R")))
stopifnot(length(utils::capture.output(report.removed(character()))) == 0L)
stopifnot(grepl("~ z.R$", utils::capture.output(report.modified("z.R"))))
stopifnot(length(utils::capture.output(report.changes(ch))) == 2L)

# watch.files with a zero timeout returns without polling
n <- suppressMessages(watch.files(dir, timeout = 0))
stopifnot(identical(n, 0L))

# run.on.change builds a handler
handler <- run.on.change("true")
stopifnot(is.function(handler), identical(names(formals(handler)), "changes"))
