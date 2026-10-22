#' Find migration files on disk
#'
#' Lists `.sql` files in `dir` whose names start with a numeric version,
#' e.g. `0007_add_invoice_index.sql`, sorted by version.
#'
#' @param dir Directory containing migration files.
#' @keywords internal
#' @return A data frame with columns `version`, `name` and `path`.
find_migrations <- function(dir) {
  if (!dir.exists(dir)) {
    stop("Migration directory `", dir, "` does not exist.", call. = FALSE)
  }

  files <- list.files(dir, pattern = "^[0-9]+_.*\\.sql$", full.names = TRUE)
  base <- basename(files)
  version <- as.integer(sub("_.*$", "", base))
  # 0007_a.sql and 7_b.sql would both claim version 7
  if (anyDuplicated(version)) {
    dup <- unique(version[duplicated(version)])
    stop("Duplicate migration versions: ", toString(dup), call. = FALSE)
  }

  out <- data.frame(
    version = version,
    name = sub("^[0-9]+_(.*)\\.sql$", "\\1", base),
    path = files,
    stringsAsFactors = FALSE
  )
  out <- out[order(out$version), , drop = FALSE]
  rownames(out) <- NULL
  out
}

#' Create the migration bookkeeping table
#'
#' Creates `schema_migrations` if it does not exist yet. The table records
#' each applied version, its name, a checksum and when it was applied.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the bookkeeping table.
#' @keywords internal
ensure_migration_table <- function(con, table = "schema_migrations") {
  if (DBI::dbExistsTable(con, table)) {
    return(invisible(con))
  }

  sql <- glue::glue_sql(
    "CREATE TABLE {`table`} (
       version INTEGER PRIMARY KEY,
       name TEXT NOT NULL,
       checksum TEXT NOT NULL,
       applied_at TEXT NOT NULL
     )",
    .con = con
  )
  DBI::dbExecute(con, sql)
  invisible(con)
}

#' Read the applied migrations
#'
#' Returns the rows of the bookkeeping table, sorted by version.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the bookkeeping table.
#' @export
#' @return A data frame with columns `version`, `name`, `checksum` and
#'   `applied_at`.
applied_migrations <- function(con, table = "schema_migrations") {
  ensure_migration_table(con, table)
  sql <- glue::glue_sql("SELECT * FROM {`table`} ORDER BY version", .con = con)
  DBI::dbGetQuery(con, sql)
}

#' Checksum a migration file
#'
#' Hashes the file contents with whitespace at line ends removed, so that
#' editor noise does not count as a change.
#'
#' @param path Path to a migration file.
#' @keywords internal
migration_checksum <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- sub("[[:space:]]+$", "", lines)
  digest::digest(paste(lines, collapse = "\n"), algo = "sha256", serialize = FALSE)
}

#' Report pending and modified migrations
#'
#' Compares migration files on disk with the bookkeeping table and labels
#' each version as `"applied"`, `"pending"`, `"modified"` or `"missing"`.
#'
#' @param con A `DBIConnection` object.
#' @param dir Directory containing migration files.
#' @param table Name of the bookkeeping table.
#' @export
#' @return A data frame with columns `version`, `name` and `status`.
migration_status <- function(con, dir, table = "schema_migrations") {
  on_disk <- find_migrations(dir)
  done <- applied_migrations(con, table)
  versions <- sort(union(on_disk$version, done$version))

  status <- character(length(versions))
  names_out <- character(length(versions))
  for (i in seq_along(versions)) {
    v <- versions[i]
    file_row <- on_disk[on_disk$version == v, , drop = FALSE]
    db_row <- done[done$version == v, , drop = FALSE]
    names_out[i] <- if (nrow(file_row) > 0) file_row$name else db_row$name

    # a recorded version with no file, or a file whose content changed,
    # both mean the database no longer matches the repository
    status[i] <- if (nrow(db_row) == 0) {
      "pending"
    } else if (nrow(file_row) == 0) {
      "missing"
    } else if (!identical(migration_checksum(file_row$path), db_row$checksum)) {
      "modified"
    } else {
      "applied"
    }
  }

  data.frame(version = versions, name = names_out, status = status, stringsAsFactors = FALSE)
}

#' Apply a single migration
#'
#' Runs every statement of a migration file and records it in the
#' bookkeeping table, all inside one transaction.
#'
#' @param con A `DBIConnection` object.
#' @param migration A one-row data frame as returned by `find_migrations()`.
#' @param table Name of the bookkeeping table.
#' @keywords internal
#' @return The version number, invisibly.
apply_migration <- function(con, migration, table = "schema_migrations") {
  text <- paste(readLines(migration$path, warn = FALSE), collapse = "\n")
  # only a semicolon at the end of a line separates statements
  statements <- trimws(strsplit(text, ";[[:space:]]*\n")[[1]])
  statements <- statements[nzchar(statements)]

  record <- data.frame(
    version = migration$version,
    name = migration$name,
    checksum = migration_checksum(migration$path),
    applied_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S", tz = "UTC")
  )

  DBI::dbWithTransaction(con, {
    for (stmt in statements) {
      DBI::dbExecute(con, stmt)
    }
    DBI::dbAppendTable(con, table, record)
  })
  invisible(migration$version)
}

#' Run pending migrations
#'
#' Applies every pending migration in version order. Refuses to run when an
#' already-applied migration has been edited or deleted, unless `force`.
#'
#' @param con A `DBIConnection` object.
#' @param dir Directory containing migration files.
#' @param table Name of the bookkeeping table.
#' @param dry_run Whether to only report what would be applied.
#' @param force Whether to ignore modified or missing migrations.
#' @export
#' @return An integer vector of the versions applied (or, for a dry run,
#'   the versions that would be applied), invisibly.
migrate <- function(con, dir, table = "schema_migrations", dry_run = FALSE, force = FALSE) {
  status <- migration_status(con, dir, table)
  broken <- status[status$status %in% c("modified", "missing"), , drop = FALSE]
  if (nrow(broken) > 0 && !force) {
    stop(
      "Applied migrations differ from disk: ",
      paste0(broken$version, " (", broken$status, ")", collapse = ", "),
      call. = FALSE
    )
  }

  pending <- status$version[status$status == "pending"]
  if (length(pending) == 0) {
    message("Database is up to date.")
    return(invisible(integer()))
  }

  # re-read the files so each migration row carries its path
  on_disk <- find_migrations(dir)
  for (v in pending) {
    migration <- on_disk[on_disk$version == v, , drop = FALSE]
    message(sprintf("Applying %04d_%s", v, migration$name))
    if (!dry_run) {
      apply_migration(con, migration, table)
    }
  }
  invisible(pending)
}
