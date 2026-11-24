# skipped: needs DBI (fetch.typed)
stopifnot(identical(cast.column(c("1", " 2", ""), "integer"), c(1L, 2L, NA)))
stopifnot(identical(cast.column(c("1.5", NA), "numeric"), c(1.5, NA)))
stopifnot(identical(suppressWarnings(cast.column(c("t", "False", "1", "x"), "logical")),
                    c(TRUE, FALSE, TRUE, NA)))
stopifnot(identical(cast.column("2024-03-01 00:00:00", "date"), as.Date("2024-03-01")))
ts <- cast.column("2024-03-01 10:20:30.5", "timestamp")
stopifnot(inherits(ts, "POSIXct"), format(ts, "%H:%M:%S", tz = "UTC") == "10:20:30")
stopifnot(identical(cast.column(c(" a ", ""), "character"), c("a", NA)))
w <- tryCatch(cast.column(c("1", "x"), "integer", name = "id"), warning = conditionMessage)
stopifnot(w == "Column `id`: 1 value(s) could not be cast to integer.")

df <- data.frame(id = c("1", "2"), ok = c("t", "f"), note = c("a", "b"))
out <- cast.result.set(df, c(id = "integer", ok = "logical"))
stopifnot(is.integer(out$id), is.logical(out$ok), identical(out$note, c("a", "b")))
stopifnot(identical(cast.result.set(df, character()), df))
stopifnot(inherits(try(cast.result.set(df, c(zz = "integer")), silent = TRUE), "try-error"))
stopifnot(inherits(try(cast.result.set(list(), c(a = "integer")), silent = TRUE), "try-error"))

stopifnot(identical(normalize.booleans(c("t", "f", NA)), c(TRUE, FALSE, NA)))
stopifnot(identical(normalize.booleans(c(1, 0), "mysql"), c(TRUE, FALSE)))
stopifnot(identical(normalize.booleans(c("true", "0"), "sqlite"), c(TRUE, FALSE)))
stopifnot(identical(normalize.booleans(c("True", "False"), "mssql"), c(TRUE, FALSE)))
stopifnot(identical(normalize.booleans(c(TRUE, NA)), c(TRUE, NA)))
stopifnot(inherits(try(normalize.booleans("true", "mysql"), silent = TRUE), "try-error"))

ref <- as.POSIXct("2024-01-02 03:04:05", tz = "UTC")
stopifnot(parse.db.timestamp("2024-01-02 03:04:05.678+00") == ref)
stopifnot(parse.db.timestamp("2024-01-02 03:04:05+05:30") == ref)
stopifnot(parse.db.timestamp("2024-01-02 03:04:05.678", "mysql") == ref)
stopifnot(parse.db.timestamp("2024-01-02T03:04:05.678Z", "sqlite") == ref)
stopifnot(parse.db.timestamp("2024-01-02 03:04:05.6780000", "mssql") == ref)

m <- cast.types.from.meta(c(a = "bigint", b = "DECIMAL(10,2)", c = "boolean", d = "date",
                            e = "timestamptz", f = "varchar(20)", g = "TINYINT(1)"))
stopifnot(identical(unname(m), c("integer", "numeric", "logical", "date", "timestamp",
                                 "character", "logical")))
stopifnot(identical(names(m), letters[1:7]))
stopifnot(cast.types.from.meta(c(x = "DATETIME2"))[["x"]] == "timestamp")
