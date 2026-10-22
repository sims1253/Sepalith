# skipped: read_extract_csv, read_extract_tsv, read_extract_jsonl, write_extract,
#   read_extract_dir, normalise_names, sync_extracts (need data.table / jsonlite)

tmp <- tempfile()
writeLines("x", tmp)
stopifnot(identical(check_readable(tmp), tmp))
msg <- tryCatch(check_readable(file.path(tmp, "nope")), error = conditionMessage)
stopifnot(grepl("^File not found", msg))
unlink(tmp)
