# skipped: needs rlang (error paths of path_sanitize, path_unique and
#   path_ensure_dir)
stopifnot(identical(path_ext_set(c("data/raw.csv", "notes", NA), ".txt"),
  c("data/raw.txt", "notes.txt", NA)))
stopifnot(identical(path_ext_set("data/raw.csv", ""), "data/raw"))
stopifnot(identical(path_ext_strip(c("archive.tar.gz", "v1.2/readme", ".Rprofile")),
  c("archive.tar", "v1.2/readme", ".Rprofile")))
stopifnot(identical(path_ext(c("raw.csv", "README", ".Rprofile", NA)), c("csv", "", "", NA)))

stopifnot(all(path_is_absolute(c("/srv", "~/x", "C:/Users", "D:\\d", "\\\\srv\\share"))))
stopifnot(!any(path_is_absolute(c("data/raw.csv", "../up", "C:relative"))))
stopifnot(is.na(path_is_absolute(NA_character_)))

stopifnot(identical(path_rel_to("/srv/app/data/raw.csv", "/srv/app"), "data/raw.csv"))
stopifnot(identical(path_rel_to("/srv/other/x.R", "/srv/app"), "../other/x.R"))
stopifnot(identical(path_rel_to(c("/srv/app", "/srv/app/./a/../b"), "/srv/app"), c(".", "b")))
old <- setwd(tempdir())
stopifnot(identical(path_rel_to("no-such-dir/f.R", file.path(getwd(), "no-such-dir")), "f.R"))
setwd(old)

stopifnot(identical(path_sanitize("Q3 report: draft/final?"), "Q3_report_draft_final_"))
stopifnot(identical(path_sanitize(c("...hidden", "", NA)), c("hidden", "untitled", NA)))
stopifnot(nchar(path_sanitize(strrep("x", 300))) == 255L)

dir <- tempfile("pw-")
nested <- file.path(dir, "a", "b")
stopifnot(identical(path_ensure_dir(nested), nested), dir.exists(nested))
target <- file.path(dir, "export.csv")
stopifnot(identical(path_unique(target), target))
file.create(target)
stopifnot(identical(path_unique(target), file.path(dir, "export_1.csv")))
unlink(dir, recursive = TRUE)
