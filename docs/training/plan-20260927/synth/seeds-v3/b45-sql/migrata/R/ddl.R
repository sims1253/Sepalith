#' Render one column definition
#'
#' Maps a portable `type` to the dialect's type name and appends constraints.
#'
#' @param name The column name.
#' @param type One of `"integer"`, `"bigint"`, `"numeric"`, `"text"`, `"boolean"`,
#'   `"date"` or `"timestamp"`.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @param nullable Whether the column accepts `NULL`.
#' @param default A SQL expression used as the default, or `NULL`.
#' @param primaryKey Whether the column is the primary key.
#' @return A single string such as `"id" BIGINT NOT NULL PRIMARY KEY`.
#' @export
columnDdl = function(name,
                     type = c("integer", "bigint", "numeric", "text", "boolean", "date",
                              "timestamp"),
                     dialect = c("postgres", "mysql", "sqlite"),
                     nullable = TRUE, default = NULL, primaryKey = FALSE) {
  type = match.arg(type)
  dialect = match.arg(dialect)
  quote = if (dialect == "mysql") "`" else "\""
  typeName = switch(type,
    integer = c(postgres = "INTEGER", mysql = "INT", sqlite = "INTEGER"),
    bigint = c(postgres = "BIGINT", mysql = "BIGINT", sqlite = "INTEGER"),
    numeric = c(postgres = "NUMERIC(18, 6)", mysql = "DECIMAL(18, 6)", sqlite = "REAL"),
    text = c(postgres = "TEXT", mysql = "VARCHAR(255)", sqlite = "TEXT"),
    boolean = c(postgres = "BOOLEAN", mysql = "TINYINT(1)", sqlite = "INTEGER"),
    date = c(postgres = "DATE", mysql = "DATE", sqlite = "TEXT"),
    timestamp = c(postgres = "TIMESTAMPTZ", mysql = "DATETIME", sqlite = "TEXT")
  )[[dialect]]
  parts = c(paste0(quote, name, quote), typeName)

  # NOT NULL
  if (!isTRUE(nullable) || isTRUE(primaryKey)) {
    parts = c(parts, "NOT NULL")
  }
  # DEFAULT
  if (!is.null(default)) {
    parts = c(parts, paste("DEFAULT", default))
  }
  # PRIMARY KEY
  if (isTRUE(primaryKey)) {
    parts = c(parts, "PRIMARY KEY")
  }
  return(paste(parts, collapse = " "))
}

#' Render a CREATE TABLE statement
#'
#' Renders one column definition per row of `columns`.
#'
#' @param table The table name.
#' @param columns A data frame with columns `name`, `type`, `nullable`, `default` and
#'   `primaryKey`; only `name` and `type` are required.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @param ifNotExists Whether to add `IF NOT EXISTS`.
#' @return A single SQL string.
#' @export
createTableDdl = function(table, columns, dialect = c("postgres", "mysql", "sqlite"),
                          ifNotExists = TRUE) {
  dialect = match.arg(dialect)
  if (!is.data.frame(columns) || !all(c("name", "type") %in% names(columns))) {
    stop("`columns` must be a data frame with `name` and `type` columns.", call. = FALSE)
  }
  n = nrow(columns)
  # Fill optional fields with their defaults
  if (is.null(columns$nullable)) columns$nullable = rep(TRUE, n)
  if (is.null(columns$default)) columns$default = rep(NA_character_, n)
  if (is.null(columns$primaryKey)) columns$primaryKey = rep(FALSE, n)
  defs = character(n)
  for (i in seq_len(n)) {
    default = if (is.na(columns$default[i])) NULL else columns$default[i]
    defs[i] = columnDdl(
      name = columns$name[i],
      type = columns$type[i],
      dialect = dialect,
      nullable = columns$nullable[i],
      default = default,
      primaryKey = columns$primaryKey[i]
    )
  }
  quote = if (dialect == "mysql") "`" else "\""
  guard = if (ifNotExists) "IF NOT EXISTS" else NULL
  head = paste(c("CREATE TABLE", guard, paste0(quote, table, quote)), collapse = " ")
  return(paste0(head, " (\n  ", paste(defs, collapse = ",\n  "), "\n);"))
}

#' Render DROP TABLE statements
#'
#' Drops `tables` in reverse order so dependants go before their parents.
#'
#' @param tables Table names, in creation order.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @param ifExists Whether to add `IF EXISTS`.
#' @param cascade Whether to add `CASCADE` (postgres only).
#' @return A character vector with one statement per table.
#' @export
dropTableDdl = function(tables, dialect = c("postgres", "mysql", "sqlite"), ifExists = TRUE,
                        cascade = FALSE) {
  dialect = match.arg(dialect)
  quote = if (dialect == "mysql") "`" else "\""
  out = character(length(tables))
  for (i in seq_along(tables)) {
    table = rev(tables)[i]
    guard = if (ifExists) "IF EXISTS" else NULL
    stmt = paste(c("DROP TABLE", guard, paste0(quote, table, quote)), collapse = " ")
    if (cascade && dialect == "postgres") {
      stmt = paste(stmt, "CASCADE")
    }
    out[i] = paste0(stmt, ";")
  }
  return(out)
}

#' Render a CREATE INDEX statement
#'
#' Names the index `<table>_<columns>_idx` (or `_key` when unique) unless `name` is given.
#'
#' @param table The table name.
#' @param columns Column names covered by the index.
#' @param unique Whether to create a unique index.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @param name The index name, or `NULL` for the default.
#' @return A single SQL string.
#' @export
createIndexDdl = function(table, columns, unique = FALSE,
                          dialect = c("postgres", "mysql", "sqlite"), name = NULL) {
  dialect = match.arg(dialect)
  if (length(columns) == 0) {
    stop("`columns` must name at least one column.", call. = FALSE)
  }
  quote = if (dialect == "mysql") "`" else "\""
  if (is.null(name)) {
    suffix = if (unique) "key" else "idx"
    name = paste(c(table, columns, suffix), collapse = "_")
  }
  cols = paste0(quote, columns, quote, collapse = ", ")
  verb = if (unique) "CREATE UNIQUE INDEX" else "CREATE INDEX"
  guard = if (dialect == "mysql") NULL else "IF NOT EXISTS"
  stmt = paste(c(verb, guard, paste0(quote, name, quote), "ON", paste0(quote, table, quote)),
               collapse = " ")
  return(paste0(stmt, " (", cols, ");"))
}

#' Render a foreign key constraint
#'
#' Adds a constraint with the referential action chosen by `onDelete`.
#'
#' @param table The referencing table.
#' @param column The referencing column.
#' @param refTable The referenced table.
#' @param refColumn The referenced column.
#' @param onDelete One of `"restrict"`, `"cascade"`, `"set_null"` or `"no_action"`.
#' @param dialect One of `"postgres"` or `"mysql"`; sqlite cannot add constraints.
#' @return A single SQL string.
#' @export
foreignKeyDdl = function(table, column, refTable, refColumn = "id",
                         onDelete = c("restrict", "cascade", "set_null", "no_action"),
                         dialect = c("postgres", "mysql")) {
  onDelete = match.arg(onDelete)
  dialect = match.arg(dialect)
  quote = if (dialect == "mysql") "`" else "\""
  q = function(x) paste0(quote, x, quote)
  name = paste(table, column, "fkey", sep = "_")
  action = switch(onDelete,
    restrict = {
      "ON DELETE RESTRICT"
    },
    cascade = {
      "ON DELETE CASCADE"
    },
    set_null = {
      "ON DELETE SET NULL"
    },
    no_action = {
      "ON DELETE NO ACTION"
    }
  )
  stmt = sprintf(
    "ALTER TABLE %s ADD CONSTRAINT %s FOREIGN KEY (%s) REFERENCES %s (%s) %s;",
    q(table), q(name), q(column), q(refTable), q(refColumn), action
  )
  return(stmt)
}

#' Render a rename statement
#'
#' Renames a table, column or index using the dialect's syntax.
#'
#' @param kind One of `"table"`, `"column"` or `"index"`.
#' @param from,to The old and new names.
#' @param table The owning table; required for columns and for mysql indexes.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A single SQL string.
#' @export
renameDdl = function(kind = c("table", "column", "index"), from, to, table = NULL,
                     dialect = c("postgres", "mysql", "sqlite")) {
  kind = match.arg(kind)
  dialect = match.arg(dialect)
  quote = if (dialect == "mysql") "`" else "\""
  q = function(x) paste0(quote, x, quote)
  stmt = switch(kind,
    table = {
      sprintf("ALTER TABLE %s RENAME TO %s;", q(from), q(to))
    },
    column = {
      if (is.null(table)) stop("`table` is required to rename a column.", call. = FALSE)
      sprintf("ALTER TABLE %s RENAME COLUMN %s TO %s;", q(table), q(from), q(to))
    },
    index = {
      if (dialect == "sqlite") {
        stop("Renaming an index is not supported for dialect \"sqlite\".", call. = FALSE)
      }
      if (dialect == "mysql") {
        if (is.null(table)) stop("`table` is required to rename an index.", call. = FALSE)
        sprintf("ALTER TABLE %s RENAME INDEX %s TO %s;", q(table), q(from), q(to))
      } else {
        sprintf("ALTER INDEX %s RENAME TO %s;", q(from), q(to))
      }
    }
  )
  return(stmt)
}
