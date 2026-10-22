# skipped: needs rmarkdown (renderRmd), quarto (renderQuarto); renderDocument()
#   and renderMany() are only checked on paths that never reach an engine

# detectEngine()
stopifnot(
  identical(detectEngine("reports/weekly.qmd"), "quarto"),
  identical(detectEngine("volume.Rmd"), "rmarkdown"),
  inherits(try(detectEngine("notes.md"), silent = TRUE), "try-error")
)

# outputPathFor()
stopifnot(
  identical(outputPathFor("a/weekly.qmd", "out"), file.path("out", "weekly.html")),
  identical(outputPathFor("weekly.qmd", "out", "pdf", label = "2026-03-02 08:00"),
            file.path("out", "weekly_2026-03-02-08-00.pdf")),
  identical(outputPathFor("weekly.qmd", "out", label = ""), file.path("out", "weekly.html")),
  inherits(try(outputPathFor("weekly.qmd", "out", "pptx"), silent = TRUE), "try-error")
)

# renderDocument() and renderMany()
stopifnot(
  inherits(try(renderDocument(tempfile(fileext = ".qmd"), tempdir()), silent = TRUE),
           "try-error"),
  is.null(renderMany(character(), tempdir()))
)

# cleanIntermediates()
dir <- tempfile("clean-")
dir.create(file.path(dir, "weekly_files"), recursive = TRUE)
file.create(file.path(dir, c("weekly.log", "weekly.qmd", "weekly.tex")))
dry <- suppressMessages(cleanIntermediates(dir, dryRun = TRUE))
stopifnot(length(dry) == 3L, all(file.exists(dry)))
gone <- cleanIntermediates(dir)
stopifnot(
  setequal(basename(gone), c("weekly.log", "weekly.tex", "weekly_files")),
  identical(list.files(dir), "weekly.qmd"),
  identical(cleanIntermediates(dir), character())
)
unlink(dir, recursive = TRUE)
