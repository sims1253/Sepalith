#!/usr/bin/env Rscript
# Enumerate names bound directly in baseenv. This reads namespace metadata only;
# it does not source or invoke package code.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: generate_r_base_bindings.R OUTPUT_TS OUTPUT_JSON")
names <- sort(ls(envir = baseenv(), all.names = TRUE), method = "radix")
version <- R.version.string
payload <- paste0(paste(names, collapse = "\n"), "\n")
tmp <- tempfile()
writeChar(payload, tmp, eos = NULL, useBytes = TRUE)
sha <- system2("sha256sum", tmp, stdout = TRUE)
sha <- strsplit(sha[[1L]], " ", fixed = TRUE)[[1L]][[1L]]
unlink(tmp)
escape <- function(x) paste0('  ', encodeString(x, quote = '"'), ',')
ts <- c(
  "/** Generated from ls(baseenv(), all.names=TRUE); do not hand edit. */",
  sprintf("export const R_BASE_BINDINGS_RUNTIME = %s as const;", encodeString(version, quote='"')),
  sprintf("export const R_BASE_BINDINGS_SHA256 = %s as const;", encodeString(sha, quote='"')),
  "export const R_BASE_BINDINGS = new Set<string>([",
  vapply(names, escape, character(1L)),
  "] as const);", ""
)
writeLines(ts, args[[1L]], useBytes = TRUE)
meta <- sprintf('{\n  "schema": "sepalith.run06.r_base_bindings.v1",\n  "runtime": %s,\n  "enumeration": "sort(ls(envir=baseenv(), all.names=TRUE), method=radix)",\n  "count": %d,\n  "names_lf_sha256": "%s",\n  "source_or_package_code_executed": false\n}\n', encodeString(version, quote='"'), length(names), sha)
writeLines(meta, args[[2L]], useBytes = TRUE)
