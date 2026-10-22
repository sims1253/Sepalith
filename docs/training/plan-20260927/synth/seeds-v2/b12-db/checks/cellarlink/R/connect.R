# skipped: needs DBI and a driver (connect_postgres, connect_sqlite, connect_duckdb,
# is_alive, with_connection, describe_connection)

# `%||%`
stopifnot(identical(NULL %||% 2, 2), identical(1 %||% 2, 1), identical(NA %||% 2, NA))

# env_settings
Sys.setenv(CHKX_HOST = "db.local", CHKX_PORT = "6543", CHKX_USER = "")
s <- env_settings("CHKX")
stopifnot(identical(s$host, "db.local"), identical(s$port, 6543L))
stopifnot(is.null(s$user), is.null(s$dbname))
stopifnot(identical(names(s), c("host", "port", "dbname", "user", "password")))
Sys.setenv(CHKX_PORT = "abc")
stopifnot(inherits(try(env_settings("CHKX"), silent = TRUE), "try-error"))
