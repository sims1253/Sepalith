# skipped: needs DBI, glue or digest (ensure_migration_table, applied_migrations,
# migration_checksum, migration_status, apply_migration, migrate)
dir <- file.path(tempdir(), "lk-migrations-check")
dir.create(dir, showWarnings = FALSE)
writeLines("CREATE TABLE a (id INTEGER);", file.path(dir, "0010_second.sql"))
writeLines("CREATE TABLE b (id INTEGER);", file.path(dir, "0002_first.sql"))
writeLines("not a migration", file.path(dir, "notes.txt"))

# find_migrations
m <- find_migrations(dir)
stopifnot(identical(m$version, c(2L, 10L)), identical(m$name, c("first", "second")))
stopifnot(identical(rownames(m), c("1", "2")), all(file.exists(m$path)))
writeLines("SELECT 1;", file.path(dir, "2_again.sql"))
stopifnot(inherits(try(find_migrations(dir), silent = TRUE), "try-error"))
stopifnot(inherits(try(find_migrations(file.path(dir, "nope")), silent = TRUE), "try-error"))
