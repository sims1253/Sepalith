#' Create a connection pool for Postgres
#'
#' Wraps [pool::dbPool()] with sensible defaults for a small web service.
#' Settings not given explicitly are read from the environment.
#'
#' @param dbname Name of the database, or `NULL` to read it from the environment.
#' @param min_size Minimum number of idle connections kept open.
#' @param max_size Maximum number of connections the pool may open.
#' @param prefix Prefix of the environment variable names.
#' @return A `Pool` object.
#' @family pools
#' @export
pool_postgres <- function(dbname = NULL, min_size = 1, max_size = 5, prefix = "CELLAR") {
  settings <- env_settings(prefix)
  dbname <- dbname %||% settings$dbname
  if (is.null(dbname)) {
    stop("No database name given and `", prefix, "_DBNAME` is unset.", call. = FALSE)
  }
  pool <- pool::dbPool(
    drv = RPostgres::Postgres(),
    dbname = dbname,
    host = settings$host %||% "localhost",
    port = settings$port %||% 5432L,
    user = settings$user,
    password = settings$password,
    minSize = min_size,
    maxSize = max_size,
    idleTimeout = 60
  )
  return(pool)
}

#' Create a connection pool for SQLite
#'
#' Wraps [pool::dbPool()] for a SQLite file. SQLite serialises writes, so
#' the pool is capped at a single connection by default.
#'
#' @param path Path to the database file.
#' @param min_size Minimum number of idle connections kept open.
#' @param max_size Maximum number of connections the pool may open.
#' @return A `Pool` object.
#' @family pools
#' @export
pool_sqlite <- function(path, min_size = 1, max_size = 1) {
  if (!file.exists(path)) {
    stop("SQLite file `", path, "` does not exist.", call. = FALSE)
  }
  pool <- pool::dbPool(
    drv = RSQLite::SQLite(),
    dbname = path,
    minSize = min_size,
    maxSize = max_size,
    idleTimeout = 60
  )
  return(pool)
}

#' Run code with a checked-out connection
#'
#' Checks a connection out of `pool`, passes it to `code`, and returns it
#' to the pool afterwards, even on error.
#'
#' @param pool A `Pool` object.
#' @param code A function taking the connection as its only argument.
#' @return The value returned by `code`.
#' @export
with_pooled <- function(pool, code) {
  code <- match.fun(code)
  con <- pool::poolCheckout(pool)
  on.exit(pool::poolReturn(con), add = TRUE)
  return(code(con))
}

#' Run a query through a pool
#'
#' Convenience wrapper around [DBI::dbGetQuery()] for a pool, with optional
#' bound parameters.
#'
#' @param pool A `Pool` object.
#' @param sql A query string or `SQL` object.
#' @param params A list of bound parameters, or `NULL`.
#' @return A data frame of results.
#' @family pooled queries
#' @export
pooled_query <- function(pool, sql, params = NULL) {
  with_pooled(pool, function(con) {
    if (is.null(params)) {
      DBI::dbGetQuery(con, sql)
    } else {
      DBI::dbGetQuery(con, sql, params = params)
    }
  })
}

#' Execute a statement through a pool
#'
#' Convenience wrapper around [DBI::dbExecute()] for a pool, with optional
#' bound parameters.
#'
#' @param pool A `Pool` object.
#' @param sql A query string or `SQL` object.
#' @param params A list of bound parameters, or `NULL`.
#' @return The number of rows affected.
#' @family pooled queries
#' @export
pooled_execute <- function(pool, sql, params = NULL) {
  with_pooled(pool, function(con) {
    if (is.null(params)) {
      DBI::dbExecute(con, sql)
    } else {
      DBI::dbExecute(con, sql, params = params)
    }
  })
}

#' Close a pool when a Shiny session ends
#'
#' Registers [pool::poolClose()] with [shiny::onStop()] so the pool is
#' shut down cleanly when the app stops.
#'
#' @param pool A `Pool` object.
#' @return `pool`, invisibly.
#' @export
close_on_stop <- function(pool) {
  shiny::onStop(function() {
    if (pool::dbIsValid(pool)) {
      pool::poolClose(pool)
    }
  })
  return(invisible(pool))
}

#' Report the state of a pool
#'
#' Returns counts of free and taken connections, useful for a health
#' endpoint.
#'
#' @param pool A `Pool` object.
#' @return A one-row data frame with columns `valid`, `free`, `taken` and
#'   `alive`.
#' @export
pool_status <- function(pool) {
  valid <- pool::dbIsValid(pool)
  if (!valid) {
    return(data.frame(valid = FALSE, free = 0L, taken = 0L, alive = FALSE))
  }
  counts <- pool$counters
  alive <- with_pooled(pool, is_alive)
  return(data.frame(
    valid = TRUE,
    free = as.integer(counts$free),
    taken = as.integer(counts$taken),
    alive = alive
  ))
}
