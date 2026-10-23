#' Migration 0001 (up): core account tables
#'
#' Creates `accounts`, `sessions`, `api_keys` and `audit_log`, one block per table.
#'
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A character vector with one CREATE TABLE statement per table.
#' @export
migration0001Up = function(dialect = c("postgres", "mysql", "sqlite")) {
  dialect = match.arg(dialect)
  quote = if (dialect == "mysql") "`" else "\""
  types = switch(dialect,
    postgres = c(id = "BIGSERIAL PRIMARY KEY", ref = "BIGINT NOT NULL", text = "TEXT",
                 time = "TIMESTAMPTZ", flag = "BOOLEAN"),
    mysql = c(id = "BIGINT AUTO_INCREMENT PRIMARY KEY", ref = "BIGINT NOT NULL",
              text = "VARCHAR(255)", time = "DATETIME", flag = "TINYINT(1)"),
    sqlite = c(id = "INTEGER PRIMARY KEY AUTOINCREMENT", ref = "INTEGER NOT NULL",
               text = "TEXT", time = "TEXT", flag = "INTEGER")
  )
  template = "CREATE TABLE %s (\n  %s\n);"
  qi = function(x) paste0(quote, x, quote)
  col = function(name, type) paste(qi(name), types[[type]])
  stmts = character()

  # Table `accounts`
  defs = c(col("id", "id"), col("email", "text"), col("created_at", "time"),
           col("active", "flag"))
  stmts = c(stmts, sprintf(template, qi("accounts"), paste(defs, collapse = ",\n  ")))
  # Table `sessions`
  defs = c(col("id", "id"), col("account_id", "ref"), col("token", "text"),
           col("expires_at", "time"))
  stmts = c(stmts, sprintf(template, qi("sessions"), paste(defs, collapse = ",\n  ")))
  # Table `api_keys`
  defs = c(col("id", "id"), col("account_id", "ref"), col("key_hash", "text"),
           col("revoked", "flag"))
  stmts = c(stmts, sprintf(template, qi("api_keys"), paste(defs, collapse = ",\n  ")))
  # Table `audit_log`
  defs = c(col("id", "id"), col("account_id", "ref"), col("action", "text"),
           col("logged_at", "time"))
  stmts = c(stmts, sprintf(template, qi("audit_log"), paste(defs, collapse = ",\n  ")))
  return(stmts)
}

#' Migration 0001 (down): drop core account tables
#'
#' Drops `audit_log`, `api_keys`, `sessions` and `accounts`, one block per table.
#'
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A character vector with one DROP TABLE statement per table.
#' @export
migration0001Down = function(dialect = c("postgres", "mysql", "sqlite")) {
  dialect = match.arg(dialect)
  quote = if (dialect == "mysql") "`" else "\""
  template = "DROP TABLE IF EXISTS %s;"
  qi = function(x) paste0(quote, x, quote)
  stmts = character()
  # Table `audit_log`
  stmts = c(stmts, sprintf(template, qi("audit_log")))
  # Table `api_keys`
  stmts = c(stmts, sprintf(template, qi("api_keys")))
  # Table `sessions`
  stmts = c(stmts, sprintf(template, qi("sessions")))
  # Table `accounts`
  stmts = c(stmts, sprintf(template, qi("accounts")))
  return(stmts)
}

#' Migration 0002 (up): profile columns
#'
#' Adds `accounts.display_name`, `accounts.verified`, `accounts.last_login_at`,
#' `sessions.user_agent` and `api_keys.label`, then backfills each column.
#'
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A character vector with an ALTER and an UPDATE statement per column.
#' @export
migration0002Up = function(dialect = c("postgres", "mysql", "sqlite")) {
  dialect = match.arg(dialect)
  quote = if (dialect == "mysql") "`" else "\""
  types = switch(dialect,
    postgres = c(text = "TEXT", flag = "BOOLEAN", time = "TIMESTAMPTZ"),
    mysql = c(text = "VARCHAR(255)", flag = "TINYINT(1)", time = "DATETIME"),
    sqlite = c(text = "TEXT", flag = "INTEGER", time = "TEXT")
  )
  fills = c(text = "''", flag = if (dialect == "postgres") "FALSE" else "0",
            time = "CURRENT_TIMESTAMP")
  addTemplate = "ALTER TABLE %s ADD COLUMN %s %s;"
  fillTemplate = "UPDATE %s SET %s = %s WHERE %s IS NULL;"
  qi = function(x) paste0(quote, x, quote)
  stmts = character()

  # Column `accounts.display_name`
  stmts = c(stmts, sprintf(addTemplate, qi("accounts"), qi("display_name"), types[["text"]]))
  stmts = c(stmts, sprintf(fillTemplate, qi("accounts"), qi("display_name"), fills[["text"]],
                           qi("display_name")))
  # Column `accounts.verified`
  stmts = c(stmts, sprintf(addTemplate, qi("accounts"), qi("verified"), types[["flag"]]))
  stmts = c(stmts, sprintf(fillTemplate, qi("accounts"), qi("verified"), fills[["flag"]],
                           qi("verified")))
  # Column `accounts.last_login_at`
  stmts = c(stmts, sprintf(addTemplate, qi("accounts"), qi("last_login_at"), types[["time"]]))
  stmts = c(stmts, sprintf(fillTemplate, qi("accounts"), qi("last_login_at"), fills[["time"]],
                           qi("last_login_at")))
  # Column `sessions.user_agent`
  stmts = c(stmts, sprintf(addTemplate, qi("sessions"), qi("user_agent"), types[["text"]]))
  stmts = c(stmts, sprintf(fillTemplate, qi("sessions"), qi("user_agent"), fills[["text"]],
                           qi("user_agent")))
  # Column `api_keys.label`
  stmts = c(stmts, sprintf(addTemplate, qi("api_keys"), qi("label"), types[["text"]]))
  stmts = c(stmts, sprintf(fillTemplate, qi("api_keys"), qi("label"), fills[["text"]],
                           qi("label")))
  return(stmts)
}

#' Migration 0002 (down): drop profile columns
#'
#' Drops `api_keys.label`, `sessions.user_agent`, `accounts.last_login_at`,
#' `accounts.verified` and `accounts.display_name`, one block per column.
#'
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A character vector with one ALTER statement per column.
#' @export
migration0002Down = function(dialect = c("postgres", "mysql", "sqlite")) {
  dialect = match.arg(dialect)
  quote = if (dialect == "mysql") "`" else "\""
  dropTemplate = "ALTER TABLE %s DROP COLUMN %s;"
  qi = function(x) paste0(quote, x, quote)
  stmts = character()
  # Column `api_keys.label`
  stmts = c(stmts, sprintf(dropTemplate, qi("api_keys"), qi("label")))
  # Column `sessions.user_agent`
  stmts = c(stmts, sprintf(dropTemplate, qi("sessions"), qi("user_agent")))
  # Column `accounts.last_login_at`
  stmts = c(stmts, sprintf(dropTemplate, qi("accounts"), qi("last_login_at")))
  # Column `accounts.verified`
  stmts = c(stmts, sprintf(dropTemplate, qi("accounts"), qi("verified")))
  # Column `accounts.display_name`
  stmts = c(stmts, sprintf(dropTemplate, qi("accounts"), qi("display_name")))
  return(stmts)
}

#' Migration 0003 (up): lookup indexes
#'
#' Adds one index per table: `accounts(email)` (unique), `sessions(token)` (unique),
#' `api_keys(account_id)` and `audit_log(account_id, logged_at)`.
#'
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A character vector with one CREATE INDEX statement per table.
#' @export
migration0003Up = function(dialect = c("postgres", "mysql", "sqlite")) {
  dialect = match.arg(dialect)
  quote = if (dialect == "mysql") "`" else "\""
  qi = function(x) paste0(quote, x, quote)
  index = function(table, columns, unique) {
    name = paste(c(table, columns, if (unique) "key" else "idx"), collapse = "_")
    verb = if (unique) "CREATE UNIQUE INDEX" else "CREATE INDEX"
    sprintf("%s %s ON %s (%s);", verb, qi(name), qi(table),
            paste(qi(columns), collapse = ", "))
  }
  stmts = character()
  # Index on `accounts`
  stmts = c(stmts, index("accounts", "email", unique = TRUE))
  # Index on `sessions`
  stmts = c(stmts, index("sessions", "token", unique = TRUE))
  # Index on `api_keys`
  stmts = c(stmts, index("api_keys", "account_id", unique = FALSE))
  # Index on `audit_log`
  stmts = c(stmts, index("audit_log", c("account_id", "logged_at"), unique = FALSE))
  return(stmts)
}

#' Migration 0003 (down): drop lookup indexes
#'
#' Drops the indexes on `audit_log`, `api_keys`, `sessions` and `accounts`, one
#' block per table; mysql needs the owning table in the statement.
#'
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A character vector with one DROP INDEX statement per table.
#' @export
migration0003Down = function(dialect = c("postgres", "mysql", "sqlite")) {
  dialect = match.arg(dialect)
  quote = if (dialect == "mysql") "`" else "\""
  qi = function(x) paste0(quote, x, quote)
  dropIndex = function(table, name) {
    if (dialect == "mysql") {
      sprintf("DROP INDEX %s ON %s;", qi(name), qi(table))
    } else {
      sprintf("DROP INDEX IF EXISTS %s;", qi(name))
    }
  }
  stmts = character()
  # Index on `audit_log`
  stmts = c(stmts, dropIndex("audit_log", "audit_log_account_id_logged_at_idx"))
  # Index on `api_keys`
  stmts = c(stmts, dropIndex("api_keys", "api_keys_account_id_idx"))
  # Index on `sessions`
  stmts = c(stmts, dropIndex("sessions", "sessions_token_key"))
  # Index on `accounts`
  stmts = c(stmts, dropIndex("accounts", "accounts_email_key"))
  return(stmts)
}
