# skipped: use_template, use_r_file, use_test_file and use_vignette_file need
# the installed pkgsmith templates and functions from description.R

# template_path fails for templates that are not installed
stopifnot(inherits(try(template_path("nope.R"), silent = TRUE), "try-error"))

# render_template
tpl <- c("# {{ name }}", "(c) {{year}} {{ name }}")
out <- render_template(tpl, list(name = "io", year = 2026))
stopifnot(identical(out, c("# io", "(c) 2026 io")))
stopifnot(identical(render_template("{{ p }}", list(p = "C:\\tmp")), "C:\\tmp"))
stopifnot(inherits(try(render_template("{{ missing }}", list()), silent = TRUE), "try-error"))

# check_file_stem
stopifnot(identical(check_file_stem(" Data Utils.R "), "data-utils"))
stopifnot(inherits(try(check_file_stem("a/b"), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_file_stem(""), silent = TRUE), "try-error"))
