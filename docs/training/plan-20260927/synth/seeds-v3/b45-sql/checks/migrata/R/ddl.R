stopifnot(columnDdl("id", "bigint", primaryKey = TRUE) == "\"id\" BIGINT NOT NULL PRIMARY KEY")
stopifnot(columnDdl("n", "integer", "mysql") == "`n` INT")
stopifnot(columnDdl("x", "numeric", "sqlite", nullable = FALSE) == "\"x\" REAL NOT NULL")
stopifnot(columnDdl("s", "text", "mysql") == "`s` VARCHAR(255)")
stopifnot(columnDdl("b", "boolean", default = "FALSE") == "\"b\" BOOLEAN DEFAULT FALSE")
stopifnot(columnDdl("d", "date", "sqlite") == "\"d\" TEXT")
stopifnot(columnDdl("t", "timestamp") == "\"t\" TIMESTAMPTZ")

cols = data.frame(name = c("id", "email"), type = c("bigint", "text"),
                  primaryKey = c(TRUE, FALSE), nullable = c(FALSE, FALSE))
s = createTableDdl("users", cols)
stopifnot(s == paste0("CREATE TABLE IF NOT EXISTS \"users\" (\n",
                      "  \"id\" BIGINT NOT NULL PRIMARY KEY,\n  \"email\" TEXT NOT NULL\n);"))
s = createTableDdl("t", data.frame(name = "a", type = "integer", default = "0"), "mysql", FALSE)
stopifnot(s == "CREATE TABLE `t` (\n  `a` INT DEFAULT 0\n);")
stopifnot(inherits(try(createTableDdl("t", data.frame(x = 1)), silent = TRUE), "try-error"))

stopifnot(identical(dropTableDdl(c("a", "b")),
                    c("DROP TABLE IF EXISTS \"b\";", "DROP TABLE IF EXISTS \"a\";")))
stopifnot(dropTableDdl("a", cascade = TRUE) == "DROP TABLE IF EXISTS \"a\" CASCADE;")
stopifnot(dropTableDdl("a", "mysql", FALSE, TRUE) == "DROP TABLE `a`;")

stopifnot(createIndexDdl("users", "email") ==
            "CREATE INDEX IF NOT EXISTS \"users_email_idx\" ON \"users\" (\"email\");")
stopifnot(createIndexDdl("u", c("a", "b"), TRUE, "mysql") ==
            "CREATE UNIQUE INDEX `u_a_b_key` ON `u` (`a`, `b`);")
stopifnot(grepl("\"ix\"", createIndexDdl("u", "a", dialect = "sqlite", name = "ix")))
stopifnot(inherits(try(createIndexDdl("u", character()), silent = TRUE), "try-error"))

fk = foreignKeyDdl("orders", "user_id", "users")
stopifnot(fk == paste0("ALTER TABLE \"orders\" ADD CONSTRAINT \"orders_user_id_fkey\" ",
                       "FOREIGN KEY (\"user_id\") REFERENCES \"users\" (\"id\") ",
                       "ON DELETE RESTRICT;"))
stopifnot(endsWith(foreignKeyDdl("o", "u", "users", onDelete = "cascade"), "ON DELETE CASCADE;"))
stopifnot(endsWith(foreignKeyDdl("o", "u", "users", onDelete = "set_null"), "SET NULL;"))
stopifnot(grepl("^ALTER TABLE `o`.*NO ACTION;$",
                foreignKeyDdl("o", "u", "users", onDelete = "no_action", dialect = "mysql")))

stopifnot(renameDdl("table", "a", "b") == "ALTER TABLE \"a\" RENAME TO \"b\";")
stopifnot(renameDdl("column", "a", "b", "t", "mysql") ==
            "ALTER TABLE `t` RENAME COLUMN `a` TO `b`;")
stopifnot(renameDdl("index", "i", "j") == "ALTER INDEX \"i\" RENAME TO \"j\";")
stopifnot(renameDdl("index", "i", "j", "t", "mysql") == "ALTER TABLE `t` RENAME INDEX `i` TO `j`;")
stopifnot(inherits(try(renameDdl("index", "i", "j", dialect = "sqlite"), silent = TRUE),
                   "try-error"))
stopifnot(inherits(try(renameDdl("column", "a", "b"), silent = TRUE), "try-error"))
stopifnot(inherits(try(renameDdl("index", "i", "j", dialect = "mysql"), silent = TRUE),
                   "try-error"))
