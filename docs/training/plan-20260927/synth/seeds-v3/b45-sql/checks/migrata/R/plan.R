# skipped: needs DBI (applyMigrations)
stopifnot(alterTableSql("t", "add_column", "c", "TEXT") ==
            "ALTER TABLE \"t\" ADD COLUMN \"c\" TEXT;")
stopifnot(alterTableSql("t", "drop_column", "c", dialect = "mysql") ==
            "ALTER TABLE `t` DROP COLUMN `c`;")
stopifnot(alterTableSql("t", "rename_column", "a", "b", "sqlite") ==
            "ALTER TABLE \"t\" RENAME COLUMN \"a\" TO \"b\";")
stopifnot(alterTableSql("t", "set_type", "c", "BIGINT") ==
            "ALTER TABLE \"t\" ALTER COLUMN \"c\" TYPE BIGINT;")
stopifnot(alterTableSql("t", "set_type", "c", "BIGINT", "mysql") ==
            "ALTER TABLE `t` MODIFY COLUMN `c` BIGINT;")
stopifnot(alterTableSql("t", "set_default", "c", "0") ==
            "ALTER TABLE \"t\" ALTER COLUMN \"c\" SET DEFAULT 0;")
stopifnot(alterTableSql("t", "drop_default", "c") ==
            "ALTER TABLE \"t\" ALTER COLUMN \"c\" DROP DEFAULT;")
stopifnot(inherits(try(alterTableSql("t", "add_column", "c"), silent = TRUE), "try-error"))
stopifnot(inherits(try(alterTableSql("t", "drop_default", "c", dialect = "sqlite"),
                       silent = TRUE), "try-error"))
msg = tryCatch(alterTableSql("t", "set_default", "c", "1", "sqlite"), error = conditionMessage)
stopifnot(msg == "`set_default` is not supported for dialect \"sqlite\".")

d = schemaDiffSql(c(id = "BIGINT", a = "TEXT", b = "INT"),
                  c(id = "BIGINT", b = "BIGINT", c = "DATE"), "t")
stopifnot(d$added == "c", d$dropped == "a", d$retyped == "b", length(d$statements) == 3L)
stopifnot(d$statements[1] == "ALTER TABLE \"t\" ADD COLUMN \"c\" DATE;")
stopifnot(d$statements[2] == "ALTER TABLE \"t\" DROP COLUMN \"a\";")
stopifnot(d$statements[3] == "ALTER TABLE \"t\" ALTER COLUMN \"b\" TYPE BIGINT;")
dm = schemaDiffSql(c(b = "INT"), c(b = "BIGINT"), "t", "mysql")
stopifnot(dm$statements == "ALTER TABLE `t` MODIFY COLUMN `b` BIGINT;", length(dm$added) == 0L)
stopifnot(inherits(try(schemaDiffSql("INT", c(a = "INT"), "t"), silent = TRUE), "try-error"))

up = c("CREATE TABLE \"x\" (\n  \"id\" INTEGER\n);",
       "CREATE INDEX \"x_id_idx\" ON \"x\" (\"id\");")
down = "DROP TABLE \"x\";"
f = renderMigrationFile("0004", "add_x", up, down)
stopifnot(f[1] == "-- migrata: 0004 add_x", f[3] == "-- +up", length(f) == 10L)
stopifnot(f[9] == "-- +down", f[10] == down)
stopifnot(inherits(try(renderMigrationFile("4", "x", up, down), silent = TRUE), "try-error"))
stopifnot(inherits(try(renderMigrationFile("0004", "Add X", up, down), silent = TRUE),
                   "try-error"))

p = parseMigrationFile(f)
stopifnot(p$id == "0004", p$name == "add_x", identical(p$up, up), identical(p$down, down))
p2 = parseMigrationFile(renderMigrationFile("0005", "noop", character(), character()))
stopifnot(length(p2$up) == 0L, length(p2$down) == 0L)
stopifnot(inherits(try(parseMigrationFile(c("x", "-- +up")), silent = TRUE), "try-error"))
stopifnot(inherits(try(parseMigrationFile(c("-- migrata: 0001 a", "-- +down", "-- +up")),
                       silent = TRUE), "try-error"))

av = c("0003", "0001", "0002", "0004")
stopifnot(identical(planMigrations(c("0001"), av), c("0002", "0003", "0004")))
stopifnot(identical(planMigrations(c("0001"), av, target = "0003"), c("0002", "0003")))
stopifnot(identical(planMigrations(c("0001", "0002", "0003"), av, direction = "down"), "0003"))
stopifnot(identical(planMigrations(c("0001", "0002", "0003"), av, "0001", "down"),
                    c("0003", "0002")))
stopifnot(inherits(try(planMigrations("0009", av), silent = TRUE), "try-error"))
