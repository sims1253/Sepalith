dir <- tempfile("pkg")
dir.create(dir)
writeLines(c(
  "Package: toypkg",
  "Title: A Toy Package.",
  "Version: 0.1.0",
  "Description: exists only for checks."
), file.path(dir, "DESCRIPTION"))

# desc_path
stopifnot(identical(desc_path("a/DESCRIPTION"), "a/DESCRIPTION"))
stopifnot(identical(desc_path("a"), file.path("a", "DESCRIPTION")))

# desc_read
d <- desc_read(dir)
stopifnot(identical(names(d), c("Package", "Title", "Version", "Description")))
stopifnot(identical(d[["Version"]], "0.1.0"))
stopifnot(inherits(try(desc_read(tempfile()), silent = TRUE), "try-error"))

# desc_get_field / desc_set_field / desc_drop_field
stopifnot(identical(desc_get_field("Version", path = dir), "0.1.0"))
stopifnot(identical(desc_get_field("URL", default = "none", path = dir), "none"))
old <- desc_set_field("License", "MIT", path = dir)
stopifnot(is.na(old), identical(desc_get_field("License", path = dir), "MIT"))
stopifnot(identical(desc_drop_field("License", path = file.path(dir, "DESCRIPTION")), "MIT"))
stopifnot(is.na(suppressMessages(desc_drop_field("License", path = dir))))
bad <- try(desc_set_field("Title", NA_character_, path = dir), silent = TRUE)
stopifnot(inherits(bad, "try-error"))

# desc_write keeps the canonical field order
desc_write(c(Zzz = "1", Version = "0.2.0", Package = "toypkg"), path = dir)
stopifnot(identical(names(desc_read(dir)), c("Package", "Version", "Zzz")))
stopifnot(inherits(try(desc_write(c("x"), path = dir), silent = TRUE), "try-error"))

# desc_check
writeLines(c(
  "Package: toypkg",
  "Title: A Toy Package.",
  "Version: 0.1.x",
  "Description: exists only for checks."
), file.path(dir, "DESCRIPTION"))
problems <- desc_check(dir)
stopifnot(length(problems) == 4L, "Missing required field: License" %in% problems)
stopifnot("Title should not end in a period." %in% problems)
