# skipped: needs data.table (readTable, writeTable, appendTable, castColumns, readTableSet,
# exportTables)
stopifnot(is.function(readTable), is.function(writeTable), is.function(appendTable))
stopifnot(is.function(castColumns), is.function(readTableSet), is.function(exportTables))
stopifnot(identical(eval(formals(readTable)$format), c("csv", "tsv", "psv")))
stopifnot(identical(eval(formals(writeTable)$format), eval(formals(exportTables)$format)))
