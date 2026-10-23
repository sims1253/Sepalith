#' Render one ALTER TABLE operation
#'
#' Dispatches on `op` and checks the operation against the dialect.
#'
#' @param table The table name.
#' @param op One of `"add_column"`, `"drop_column"`, `"rename_column"`, `"set_type"`,
#'   `"set_default"` or `"drop_default"`.
#' @param column The column the operation applies to.
#' @param value The column type, new name or default expression, depending on `op`.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A single SQL string.
#' @export
alterTableSql = function(table, op = c("add_column", "drop_column", "rename_column",
                                       "set_type", "set_default", "drop_default"),
                         column, value = NULL, dialect = c("postgres", "mysql", "sqlite")) {
  op = match.arg(op)
  dialect = match.arg(dialect)
  quote = if (dialect == "mysql") "`" else "\""
  qi = function(x) paste0(quote, x, quote)
  head = paste("ALTER TABLE", qi(table))
  needValue = function() {
    if (is.null(value)) stop(sprintf("`value` is required for `%s`.", op), call. = FALSE)
  }
  unsupported = function(dialects) {
    if (dialect %in% dialects) {
      stop(sprintf("`%s` is not supported for dialect \"%s\".", op, dialect), call. = FALSE)
    }
  }
  body = switch(op,
    add_column = {
      needValue()
      paste("ADD COLUMN", qi(column), value)
    },
    drop_column = {
      paste("DROP COLUMN", qi(column))
    },
    rename_column = {
      needValue()
      paste("RENAME COLUMN", qi(column), "TO", qi(value))
    },
    set_type = {
      needValue()
      unsupported("sqlite")
      if (dialect == "mysql") {
        paste("MODIFY COLUMN", qi(column), value)
      } else {
        paste("ALTER COLUMN", qi(column), "TYPE", value)
      }
    },
    set_default = {
      needValue()
      unsupported("sqlite")
      paste("ALTER COLUMN", qi(column), "SET DEFAULT", value)
    },
    drop_default = {
      unsupported("sqlite")
      paste("ALTER COLUMN", qi(column), "DROP DEFAULT")
    }
  )
  return(paste0(head, " ", body, ";"))
}

#' Statements that turn one column set into another
#'
#' Compares two named type vectors and emits, in order, statements for the
#' `added`, `dropped` and `retyped` columns.
#'
#' @param old,new Named character vectors mapping column names to SQL types.
#' @param table The table name.
#' @param dialect One of `"postgres"` or `"mysql"`.
#' @return A list with elements `added`, `dropped`, `retyped` and `statements`.
#' @export
schemaDiffSql = function(old, new, table, dialect = c("postgres", "mysql")) {
  dialect = match.arg(dialect)
  if (is.null(names(old)) || is.null(names(new))) {
    stop("`old` and `new` must be named character vectors.", call. = FALSE)
  }
  quote = if (dialect == "mysql") "`" else "\""
  qi = function(x) paste0(quote, x, quote)
  statements = character()

  # Added columns
  added = setdiff(names(new), names(old))
  for (col in added) {
    stmt = sprintf("ALTER TABLE %s ADD COLUMN %s %s;", qi(table), qi(col), new[[col]])
    statements = c(statements, stmt)
  }
  # Dropped columns
  dropped = setdiff(names(old), names(new))
  for (col in dropped) {
    stmt = sprintf("ALTER TABLE %s DROP COLUMN %s;", qi(table), qi(col))
    statements = c(statements, stmt)
  }
  # Retyped columns
  shared = intersect(names(old), names(new))
  retyped = shared[old[shared] != new[shared]]
  for (col in retyped) {
    stmt = if (dialect == "mysql") {
      sprintf("ALTER TABLE %s MODIFY COLUMN %s %s;", qi(table), qi(col), new[[col]])
    } else {
      sprintf("ALTER TABLE %s ALTER COLUMN %s TYPE %s;", qi(table), qi(col), new[[col]])
    }
    statements = c(statements, stmt)
  }
  return(list(added = added, dropped = dropped, retyped = retyped, statements = statements))
}

#' Render a migration file
#'
#' Writes a header line followed by an `up` section and a `down` section.
#'
#' @param id The migration id, such as `"0004"`.
#' @param name A short snake_case description.
#' @param up,down Character vectors of SQL statements.
#' @return A character vector of file lines.
#' @export
renderMigrationFile = function(id, name, up, down) {
  if (!grepl("^[0-9]{4}$", id)) {
    stop("`id` must be four digits.", call. = FALSE)
  }
  if (!grepl("^[a-z][a-z0-9_]*$", name)) {
    stop("`name` must be snake_case.", call. = FALSE)
  }
  lines = sprintf("-- migrata: %s %s", id, name)
  # Up section
  lines = c(lines, "", "-- +up")
  for (stmt in up) {
    lines = c(lines, strsplit(stmt, "\n", fixed = TRUE)[[1]])
  }
  # Down section
  lines = c(lines, "", "-- +down")
  for (stmt in down) {
    lines = c(lines, strsplit(stmt, "\n", fixed = TRUE)[[1]])
  }
  return(lines)
}

#' Parse a migration file
#'
#' Reads the header and splits the `up` and `down` sections into statements.
#'
#' @param lines A character vector of file lines, as made by [renderMigrationFile()].
#' @return A list with elements `id`, `name`, `up` and `down`.
#' @export
parseMigrationFile = function(lines) {
  header = regmatches(lines[1], regexec("^-- migrata: ([0-9]{4}) ([a-z0-9_]+)$", lines[1]))[[1]]
  if (length(header) != 3L) {
    stop("The first line must be a `-- migrata:` header.", call. = FALSE)
  }
  upAt = match("-- +up", lines)
  downAt = match("-- +down", lines)
  if (is.na(upAt) || is.na(downAt) || downAt < upAt) {
    stop("The file must contain `-- +up` followed by `-- +down`.", call. = FALSE)
  }
  splitStatements = function(body) {
    body = body[nzchar(trimws(body))]
    if (length(body) == 0L) return(character())
    ends = cumsum(c(0L, head(grepl(";\\s*$", body), -1L)))
    unname(vapply(split(body, ends), paste, character(1), collapse = "\n"))
  }
  # Up section
  upLines = if (downAt > upAt + 1L) lines[(upAt + 1L):(downAt - 1L)] else character()
  up = splitStatements(upLines)
  # Down section
  downLines = if (length(lines) > downAt) lines[(downAt + 1L):length(lines)] else character()
  down = splitStatements(downLines)
  return(list(id = header[2], name = header[3], up = up, down = down))
}

#' Plan which migrations to run
#'
#' Going `up` runs pending migrations in ascending order up to `target`; going
#' `down` rolls back applied migrations in descending order down to `target`.
#'
#' @param applied Ids of migrations already applied.
#' @param available Ids of all known migrations.
#' @param target The id to stop at, or `NULL` for all (up) or one step (down).
#' @param direction Either `"up"` or `"down"`.
#' @return A character vector of migration ids, in execution order.
#' @export
planMigrations = function(applied, available, target = NULL, direction = c("up", "down")) {
  direction = match.arg(direction)
  unknown = setdiff(applied, available)
  if (length(unknown) > 0) {
    stop(sprintf("Applied migrations are missing on disk: %s.", paste(unknown, collapse = ", ")),
         call. = FALSE)
  }
  if (direction == "up") {
    plan = sort(setdiff(available, applied))
    if (!is.null(target)) {
      plan = plan[plan <= target]
    }
  } else {
    plan = sort(applied, decreasing = TRUE)
    if (!is.null(target)) {
      plan = plan[plan > target]
    } else {
      plan = head(plan, 1L)
    }
  }
  return(plan)
}

#' Apply planned migrations on a connection
#'
#' Runs each migration in its own transaction and records it in `schema_migrations`.
#'
#' @param con A DBI connection.
#' @param migrations A named list of parsed migrations, keyed by id.
#' @param plan Migration ids in execution order, from [planMigrations()].
#' @param direction Either `"up"` or `"down"`.
#' @return The ids that were applied, invisibly.
#' @export
applyMigrations = function(con, migrations, plan, direction = c("up", "down")) {
  direction = match.arg(direction)
  DBI::dbExecute(con, paste(
    "CREATE TABLE IF NOT EXISTS schema_migrations",
    "(id TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
  ))
  done = character()
  for (id in plan) {
    migration = migrations[[id]]
    if (is.null(migration)) {
      stop(sprintf("Migration %s is not loaded.", id), call. = FALSE)
    }
    DBI::dbWithTransaction(con, {
      for (stmt in migration[[direction]]) {
        DBI::dbExecute(con, stmt)
      }
      if (direction == "up") {
        DBI::dbExecute(con, "INSERT INTO schema_migrations (id, applied_at) VALUES (?, ?)",
                       params = list(id, format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
      } else {
        DBI::dbExecute(con, "DELETE FROM schema_migrations WHERE id = ?", params = list(id))
      }
    })
    done = c(done, id)
  }
  invisible(done)
}
