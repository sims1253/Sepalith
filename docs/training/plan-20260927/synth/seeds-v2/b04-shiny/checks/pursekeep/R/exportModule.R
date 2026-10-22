# skipped: exportUI, exportServer, importUI, importServer (need shiny)

data <- data.frame(
  date = as.Date(c("2026-09-02", "2026-09-03")),
  category = c("groceries", "rent"),
  amount = c(-120.5, -900)
)

# writeBudgetFile / readBudgetFile round trips in every format
for (fmt in c("csv", "tsv", "rds")) {
  f <- tempfile(fileext = paste0(".", fmt))
  stopifnot(identical(writeBudgetFile(data, f, format = fmt), f))
  back <- readBudgetFile(f, format = fmt)
  stopifnot(inherits(back$date, "Date"), isTRUE(all.equal(back, data, check.attributes = FALSE)))
  unlink(f)
}
stopifnot(inherits(try(writeBudgetFile(data, tempfile(), format = "xlsx"), silent = TRUE),
  "try-error"))

# exportFilename
name <- exportFilename("My Budget", "csv")
stopifnot(grepl("^my-budget-[0-9]{8}\\.csv$", name))
stopifnot(endsWith(exportFilename("x", "rds"), ".rds"))
