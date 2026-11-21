form <- data.frame(id = c("01", "02", "03", "04", "05"),
  text = c("I feel calm", "I worry", "I sleep well", "I wake up tired", "I feel rested"),
  scale = c("anx", "anx", "slp", "slp", "slp"), keyed = c(-1, 1, -1, 1, -1),
  min = 1, max = 5, stringsAsFactors = FALSE)
v <- validate.form(form)
stopifnot(is.integer(v$keyed), identical(v$id, form$id))
bad <- list(form[-1], within(form, id[2] <- "01"), within(form, text[1] <- ""),
  within(form, scale[1] <- NA), within(form, keyed[1] <- 0), within(form, min <- "1"),
  within(form, max <- NA_real_), within(form, max <- 1))
for (b in bad) {
  stopifnot(inherits(try(validate.form(b), silent = TRUE), "try-error"))
}

for (fmt in c("csv", "tsv", "rds", "dcf")) {
  path <- tempfile(fileext = paste0(".", fmt))
  stopifnot(identical(write.form(form, path, fmt), path), file.exists(path))
  back <- read.form(path, fmt)
  stopifnot(identical(back$id, form$id), identical(back$text, form$text))
  stopifnot(identical(back$keyed, as.integer(form$keyed)), all(back$max == 5))
  unlink(path)
}
stopifnot(inherits(try(read.form(tempfile()), silent = TRUE), "try-error"))

st <- data.frame(id = rev(form$id), item.rest = c(0.5, 0.2, 0.6, 0.4, 0.1),
  difficulty = c(0.9, 0.5, 0.4, 0.2, 0.55), variance = c(1, 2, 3, 4, 5))
stopifnot(identical(select.items(form, st, 1)$id, c("02", "03")))
stopifnot(identical(select.items(form, st, 1, "difficulty")$id, c("01", "04")))
stopifnot(identical(select.items(form, st, 2, "variance")$id, c("01", "02", "03", "04")))
stopifnot(identical(select.items(form, st, 1, "first")$id, c("01", "03")))
stopifnot(inherits(try(select.items(form, st[-1, ]), silent = TRUE), "try-error"))

stopifnot(identical(layout.form(form)$id, form$id), identical(layout.form(form)$position, 1:5))
mixed <- form[c(3, 1, 4, 2, 5), ]
stopifnot(identical(layout.form(mixed, "grouped")$id, c("03", "04", "05", "01", "02")))
stopifnot(identical(layout.form(form, "interleaved")$id, c("01", "03", "02", "04", "05")))
set.seed(10)
before <- runif(1)
set.seed(10)
r1 <- layout.form(form, "random", seed = 3)
stopifnot(identical(runif(1), before))
stopifnot(identical(r1$id, layout.form(form, "random", seed = 3)$id), setequal(r1$id, form$id))
