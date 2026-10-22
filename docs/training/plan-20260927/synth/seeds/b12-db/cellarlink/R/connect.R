#' Read connection settings from the environment
#'
#' Looks up host, port, database name, user and password from environment
#' variables sharing a common prefix, e.g. `CELLAR_HOST`, `CELLAR_PORT`.
#'
#' @param prefix Prefix of the environment variable names.
#' @return A named list of connection settings. Missing values are `NULL`.
#' @keywords internal
env_settings <- function(prefix = "CELLAR") {
  keys <- c("host", "port", "dbname", "user", "password")
  vars <- paste0(prefix, "_", toupper(keys))
  values <- Sys.getenv(vars, unset = NA_character_)
  out <- as.list(values)
  names(out) <- keys
  out <- lapply(out, function(v) if (is.na(v) || !nzchar(v)) NULL else v)
  if (!is.null(out$port)) {
    out$port <- as.integer(out$port)
  }
  return(out)
}

#' Connect to a Postgres database
#'
#' Opens a [DBI::dbConnect()] connection using [RPostgres::Postgres()].
#' Arguments left as `NULL` are filled from the environment.
#'
#' @param dbname Name of the database, or `NULL` to read it from the environment.
#' @param ... Passed to [DBI::dbConnect()].
#' @param prefix Prefix of the environment variable names.
#' @return A `DBIConnection` object.
#' @family connectors
#' @export
connect_postgres <- function(dbname = NULL, ..., prefix = "CELLAR") {
  settings <- env_settings(prefix)
  dbname <- dbname %||% settings$dbname
  if (is.null(dbname)) {
    stop("No database name given and `", prefix, "_DBNAME` is unset.", call. = FALSE)
  }
  con <- DBI::dbConnect(
    RPostgres::Postgres(),
    dbname = dbname,
    host = settings$host %||% "localhost",
    port = settings$port %||% 5432L,
    user = settings$user,
    password = settings$password,
    ...
  )
  return(con)
}

#' Connect to a SQLite database
#'
#' Opens a [DBI::dbConnect()] connection using [RSQLite::SQLite()].
#' Use `":memory:"` for a throwaway in-memory database.
#'
#' @param path Path to the database file.
#' @param ... Passed to [DBI::dbConnect()].
#' @return A `DBIConnection` object.
#' @family connectors
#' @export
connect_sqlite <- function(path = ":memory:", ...) {
  if (!identical(path, ":memory:") && !dir.exists(dirname(path))) {
    stop("Directory `", dirname(path), "` does not exist.", call. = FALSE)
  }
  con <- DBI::dbConnect(RSQLite::SQLite(), dbname = path, ...)
  DBI::dbExecute(con, "PRAGMA foreign_keys = ON")
  return(con)
}

#' Connect to a DuckDB database
#'
#' Opens a [DBI::dbConnect()] connection using [duckdb::duckdb()].
#' Use `":memory:"` for a throwaway in-memory database.
#'
#' @param path Path to the database file.
#' @param ... Passed to [DBI::dbConnect()].
#' @return A `DBIConnection` object.
#' @family connectors
#' @export
connect_duckdb <- function(path = ":memory:", ...) {
  if (!identical(path, ":memory:") && !dir.exists(dirname(path))) {
    stop("Directory `", dirname(path), "` does not exist.", call. = FALSE)
  }
  con <- DBI::dbConnect(duckdb::duckdb(), dbdir = path, ...)
  DBI::dbExecute(con, "SET threads TO 2")
  return(con)
}

#' Check that a connection is usable
#'
#' Runs a trivial query against the connection and reports whether it
#' succeeded, without raising an error.
#'
#' @param con A `DBIConnection` object.
#' @return `TRUE` if the connection answered, `FALSE` otherwise.
#' @export
is_alive <- function(con) {
  if (!DBI::dbIsValid(con)) {
    return(FALSE)
  }
  ok <- tryCatch(
    {
      DBI::dbGetQuery(con, "SELECT 1 AS ok")
      TRUE
    },
    error = function(e) FALSE,
    warning = function(w) FALSE
  )
  return(ok)
}

#' Run code with a temporary connection
#'
#' Opens a connection with `connector`, evaluates `code` with it, and
#' always disconnects afterwards, even on error.
#'
#' @param connector A function returning a `DBIConnection`, such as
#'   [connect_sqlite()].
#' @param code A function taking the connection as its only argument.
#' @param ... Passed to `connector`.
#' @return The value returned by `code`.
#' @export
with_connection <- function(connector, code, ...) {
  connector <- match.fun(connector)
  code <- match.fun(code)
  con <- connector(...)
  on.exit(DBI::dbDisconnect(con), add = TRUE)
  result <- code(con)
  return(result)
}

#' Describe a connection
#'
#' Summarises the backend, server version and database name of a
#' connection in a single line, suitable for logging.
#'
#' @param con A `DBIConnection` object.
#' @return A character string.
#' @export
describe_connection <- function(con) {
  backend <- class(con)[1]
  if (!DBI::dbIsValid(con)) {
    return(sprintf("<%s, closed>", backend))
  }
  info <- DBI::dbGetInfo(con)
  version <- info$db.version %||% info$serverVersion %||% "unknown"
  dbname <- info$dbname %||% "unknown"
  return(sprintf("<%s %s, db=%s>", backend, version, dbname))
}

#' Default value for `NULL`
#'
#' @param x,y Values; `y` is returned when `x` is `NULL`.
#' @return `x` unless it is `NULL`, in which case `y`.
#' @noRd
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
