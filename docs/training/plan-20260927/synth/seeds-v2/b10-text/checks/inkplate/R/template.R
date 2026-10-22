# skipped: needs magrittr: template.vars, check.template.vars, render.braces,
# render.rows
# skipped: needs glue: render.glue
# render.sprintf with character values needs escape.for() from R/escape.R

# render.sprintf
stopifnot(identical(render.sprintf("%d items at %.1f", list(3L, 2.5)), "3 items at 2.5"))
stopifnot(identical(render.sprintf("100%% of %d", list(7L)), "100% of 7"))
stopifnot(inherits(try(render.sprintf("%d and %d", list(1L)), silent = TRUE), "try-error"))

# read.template
tmp <- tempfile(fileext = ".txt")
writeLines(c("## header comment", "Hello {{ name }}", "Bye"), tmp)
stopifnot(identical(read.template(tmp), "Hello {{ name }}\nBye"))
full <- read.template(tmp, comment = NULL)
stopifnot(identical(full, "## header comment\nHello {{ name }}\nBye"))
stopifnot(inherits(try(read.template(tempfile()), silent = TRUE), "try-error"))
unlink(tmp)
