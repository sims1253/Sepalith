#' Build a manifest of ledger files
#'
#' Lists every file under `dir` and records what is needed to detect later
#' changes.
#'
#' @param dir Directory to scan.
#' @param pattern Optional regular expression the file names must match.
#' @param recursive Whether to descend into subdirectories.
#' @return A data frame with one row per file and columns `file` (path
#'   relative to `dir`), `format` (lower-case extension), `bytes`, `md5` and
#'   `modified` (ISO 8601 timestamp), ordered by `file`.
#' @export
build_manifest <- function(dir, pattern = NULL, recursive = TRUE) {
  if (!dir.exists(dir)) {
    stop(sprintf("Directory `%s` does not exist.", dir), call. = FALSE)
  }
  rel <- sort(list.files(dir, pattern = pattern, recursive = recursive))
  full <- file.path(dir, rel)
  full <- full[!dir.exists(full)]
  rel <- rel[file.path(dir, rel) %in% full]
  info <- file.info(full)

  # 1. `file`: path relative to `dir`
  file <- rel
  # 2. `format`: lower-case extension
  format <- tolower(tools::file_ext(rel))
  # 3. `bytes`: size on disk
  bytes <- as.numeric(info$size)
  # 4. `md5`: content checksum
  md5 <- unname(tools::md5sum(full))
  # 5. `modified`: last modification time
  modified <- format(info$mtime, "%Y-%m-%dT%H:%M:%S")

  manifest <- data.frame(file = file, format = format, bytes = bytes, md5 = md5,
                         modified = modified, stringsAsFactors = FALSE)
  rownames(manifest) <- NULL
  return(manifest)
}

#' Extract one field from JSON manifest lines
#'
#' @param lines Character vector of flat JSON objects, one per line.
#' @param key Field name to extract.
#' @return A character vector of raw values with surrounding quotes removed.
#' @noRd
manifest_json_value <- function(lines, key) {
  pattern <- sprintf("\"%s\":(\"[^\"]*\"|[^,}]*)", key)
  hits <- regmatches(lines, regexpr(pattern, lines))
  if (length(hits) != length(lines)) {
    stop(sprintf("Manifest field `%s` is missing on some lines.", key), call. = FALSE)
  }
  values <- sub(sprintf("^\"%s\":", key), "", hits)
  return(gsub("^\"|\"$", "", values))
}

#' Write a manifest
#'
#' @param manifest A manifest data frame from [build_manifest()].
#' @param path Destination file path.
#' @param format One of `"csv"`, `"tsv"`, `"dcf"` or `"json"` (one object
#'   per line).
#' @return `path`, invisibly.
#' @export
write_manifest <- function(manifest, path, format = c("csv", "tsv", "dcf", "json")) {
  format <- match.arg(format)
  switch(format,
    csv = {
      utils::write.csv(manifest, path, row.names = FALSE)
    },
    tsv = {
      utils::write.table(manifest, path, sep = "\t", quote = FALSE, row.names = FALSE)
    },
    dcf = {
      write.dcf(manifest, path)
    },
    json = {
      lines <- sprintf(
        "{\"file\":\"%s\",\"format\":\"%s\",\"bytes\":%.0f,\"md5\":\"%s\",\"modified\":\"%s\"}",
        manifest$file, manifest$format, manifest$bytes, manifest$md5, manifest$modified
      )
      writeLines(lines, path)
    }
  )
  return(invisible(path))
}

#' Read a manifest
#'
#' @param path Path to a manifest written by [write_manifest()].
#' @param format One of `"csv"`, `"tsv"`, `"dcf"` or `"json"` (one object
#'   per line).
#' @return A manifest data frame.
#' @export
read_manifest <- function(path, format = c("csv", "tsv", "dcf", "json")) {
  format <- match.arg(format)
  if (!file.exists(path)) {
    stop(sprintf("Manifest `%s` does not exist.", path), call. = FALSE)
  }
  manifest <- switch(format,
    csv = {
      utils::read.csv(path, colClasses = "character", stringsAsFactors = FALSE)
    },
    tsv = {
      utils::read.delim(path, colClasses = "character", stringsAsFactors = FALSE)
    },
    dcf = {
      as.data.frame(read.dcf(path), stringsAsFactors = FALSE)
    },
    json = {
      lines <- readLines(path)
      data.frame(
        file = manifest_json_value(lines, "file"),
        format = manifest_json_value(lines, "format"),
        bytes = manifest_json_value(lines, "bytes"),
        md5 = manifest_json_value(lines, "md5"),
        modified = manifest_json_value(lines, "modified"),
        stringsAsFactors = FALSE
      )
    }
  )

  manifest$bytes <- as.numeric(manifest$bytes)
  return(manifest)
}

#' Verify files against a manifest
#'
#' @param manifest A manifest data frame from [build_manifest()].
#' @param dir Directory the manifest paths are relative to.
#' @return `manifest` with an added `status` column: `"ok"`, `"missing"`,
#'   `"size"` (size differs), `"checksum"` (content differs) or `"touched"`
#'   (same content, different modification time).
#' @export
verify_manifest <- function(manifest, dir) {
  full <- file.path(dir, manifest$file)
  status <- rep("ok", nrow(manifest))

  # Check existence
  exists <- file.exists(full)
  status[!exists] <- "missing"
  # Check size
  bytes <- rep(NA_real_, nrow(manifest))
  bytes[exists] <- file.size(full[exists])
  status[status == "ok" & bytes != manifest$bytes] <- "size"
  # Check checksum
  md5 <- rep(NA_character_, nrow(manifest))
  md5[exists] <- unname(tools::md5sum(full[exists]))
  status[status == "ok" & md5 != manifest$md5] <- "checksum"
  # Check modification time
  modified <- rep(NA_character_, nrow(manifest))
  modified[exists] <- format(file.mtime(full[exists]), "%Y-%m-%dT%H:%M:%S")
  status[status == "ok" & modified != manifest$modified] <- "touched"

  manifest$status <- status
  return(manifest)
}

#' Compare two manifests
#'
#' @param old,new Manifest data frames from [build_manifest()].
#' @return A list with character vectors `added`, `removed`, `changed`,
#'   `unchanged` and `renamed` naming the files in each category. A file
#'   counts as renamed when it was removed and a file with the same checksum
#'   was added; `renamed` holds the new names and those files are no longer
#'   listed in `added` or `removed`.
#' @export
diff_manifests <- function(old, new) {
  # Files only in `new`
  added <- setdiff(new$file, old$file)
  # Files only in `old`
  removed <- setdiff(old$file, new$file)
  # Files in both with different or equal checksums
  common <- intersect(old$file, new$file)
  old_md5 <- old$md5[match(common, old$file)]
  new_md5 <- new$md5[match(common, new$file)]
  changed <- common[old_md5 != new_md5]
  unchanged <- common[old_md5 == new_md5]

  # Pair removed and added files that share a checksum
  removed_md5 <- old$md5[match(removed, old$file)]
  added_md5 <- new$md5[match(added, new$file)]
  renamed <- added[added_md5 %in% removed_md5]
  added <- added[!added_md5 %in% removed_md5]
  removed <- removed[!removed_md5 %in% added_md5]

  return(list(added = sort(added), removed = sort(removed), changed = sort(changed),
              unchanged = sort(unchanged), renamed = sort(renamed)))
}
