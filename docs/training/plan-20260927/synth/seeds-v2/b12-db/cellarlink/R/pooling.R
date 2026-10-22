#' @title Create a connection pool for Postgres
#' @description Wraps [pool::dbPool()] with sensible defaults for a small web
#'   service. Settings not given explicitly are read from the environment.
#' @param dbname Name of the database, or `NULL` to read it from the environment.
#' @param min_size Minimum number of idle connections kept open.
#' @param max_size Maximum number of connections the pool may open.
#' @param prefix Prefix of the environment variable names.
#' @return A `Pool` object.
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

#' @title Create a connection pool for SQLite
#' @description Wraps [pool::dbPool()] for a SQLite file. SQLite serialises
#'   writes, so the pool is capped at a single connection by default.
#' @param path Path to the database file.
#' @param min_size Minimum number of idle connections kept open.
#' @param max_size Maximum number of connections the pool may open.
#' @return A `Pool` object.
#' @export
pool_sqlite <- function(path, min_size = 1, max_size = 1) {
  # dbPool() would silently create an empty database for a typo
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

#' @title Run code with a checked-out connection
#' @description Checks a connection out of `pool`, passes it to `code`, and
#'   returns it to the pool afterwards, even on error.
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

#' @title Run a query or statement through a pool
#' @description `pooled_query()` wraps [DBI::dbGetQuery()] and
#'   `pooled_execute()` wraps [DBI::dbExecute()] for a pool, each with
#'   optional bound parameters.
#' @param pool A `Pool` object.
#' @param sql A query string or `SQL` object.
#' @param params A list of bound parameters, or `NULL`.
#' @return A data frame of results for `pooled_query()`, the number of rows
#'   affected for `pooled_execute()`.
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

#' @rdname pooled_query
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

#' @title Close a pool when a Shiny session ends
#' @description Registers [pool::poolClose()] with [shiny::onStop()] so the
#'   pool is shut down cleanly when the app stops.
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

#' @title Report the state of a pool
#' @description Returns counts of free and taken connections, useful for a
#'   health endpoint.
#' @param pool A `Pool` object.
#' @return A one-row data frame with columns `valid`, `free`, `taken` and
#'   `alive`.
#' @export
pool_status <- function(pool) {
  valid <- pool::dbIsValid(pool)
  if (!valid) {
    return(data.frame(valid = FALSE, free = 0L, taken = 0L, alive = FALSE))
  }

  # read the counters before checking out, or the probe shows up as taken
  counts <- pool$counters
  free <- as.integer(counts$free)
  taken <- as.integer(counts$taken)
  alive <- with_pooled(pool, is_alive)

  return(data.frame(valid = TRUE, free = free, taken = taken, alive = alive))
}
