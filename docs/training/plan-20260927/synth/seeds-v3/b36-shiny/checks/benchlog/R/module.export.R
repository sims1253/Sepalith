# skipped: needs shiny (export.ui, export.server)
when <- as.POSIXct("2024-03-01 14:05", tz = "UTC")
f <- export.format("csv", time = when)
stopifnot(identical(f, list(filename = "samples-20240301-1405.csv", mime = "text/csv",
                            label = "CSV")))
stopifnot(identical(export.format("tsv", "lab", when)$filename, "lab-20240301-1405.tsv"))
stopifnot(identical(export.format("json", time = when)$mime, "application/json"))
stopifnot(identical(export.format("md", time = when)$label, "Markdown"))

d <- data.frame(sample_id = c("A", "B \"x\""), volume = c(10.5, NA), status = c("ok", NA),
                stringsAsFactors = FALSE)
p <- tempfile(fileext = ".csv")
stopifnot(identical(export.csv(d, p), p))
back <- utils::read.csv(p, stringsAsFactors = FALSE)
stopifnot(identical(back$sample_id, d$sample_id), identical(back$volume, d$volume))
stopifnot(inherits(try(export.csv(list(a = 1), p), silent = TRUE), "try-error"))

p2 <- tempfile(fileext = ".tsv")
export.tsv(d[1, ], p2)
stopifnot(identical(readLines(p2), c("sample_id\tvolume\tstatus", "A\t10.5\tok")))

p3 <- tempfile(fileext = ".json")
export.json(d, p3)
js <- readLines(p3)
stopifnot(identical(js[1], "["), identical(js[length(js)], "]"))
stopifnot(identical(js[2], "  {\"sample_id\": \"A\", \"volume\": 10.5, \"status\": \"ok\"},"))
stopifnot(identical(js[3],
                    "  {\"sample_id\": \"B \\\"x\\\"\", \"volume\": null, \"status\": null}"))
export.json(d[0, ], p3)
stopifnot(identical(readLines(p3), c("[", "", "]")))

p4 <- tempfile(fileext = ".md")
export.md(d, p4, na = "-")
md <- readLines(p4)
stopifnot(identical(md[1], "| sample_id | volume | status |"), identical(md[2], "|---|---|---|"))
stopifnot(identical(md[3], "| A | 10.5 | ok |"), identical(md[4], "| B \"x\" | - | - |"))

out <- file.path(tempdir(), "bundle-check")
paths <- export.bundle(d, out, time = when)
stopifnot(identical(names(paths), c("csv", "tsv", "json", "md")), all(file.exists(paths)))
stopifnot(identical(basename(paths[["json"]]), "samples-20240301-1405.json"))
two <- export.bundle(d, out, formats = c("md", "tsv"), prefix = "lab", time = when)
stopifnot(identical(names(two), c("tsv", "md")),
          identical(basename(two[["md"]]), "lab-20240301-1405.md"))
stopifnot(inherits(try(export.bundle(d, out, formats = "xlsx"), silent = TRUE), "try-error"))
unlink(out, recursive = TRUE)
