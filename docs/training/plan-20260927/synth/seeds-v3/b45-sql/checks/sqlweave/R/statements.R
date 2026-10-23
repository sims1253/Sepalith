stopifnot(identical(sql_quote(c("a", "*")), c("\"a\"", "*")))
stopifnot(sql_quote("a`b", "identifier", "mysql") == "`a``b`")
stopifnot(identical(sql_quote(c(TRUE, NA), "literal", "sqlite"), c("1", "NULL")))
stopifnot(identical(sql_quote(c(1.5, 2), "literal"), c("1.5", "2")))
stopifnot(sql_quote("o'k", "literal") == "'o''k'")

s <- build_select("users")
stopifnot(s == "SELECT *\nFROM \"users\"")
s <- build_select("users", c("id", "name"), where = c("a = 1", "b = 2"), group_by = "name",
                  having = "COUNT(*) > 1", order_by = c("-id", "name"), limit = 5)
lines <- strsplit(s, "\n")[[1]]
stopifnot(lines[1] == "SELECT \"id\", \"name\"", lines[3] == "WHERE a = 1 AND b = 2")
stopifnot(lines[4] == "GROUP BY \"name\"", lines[5] == "HAVING COUNT(*) > 1")
stopifnot(lines[6] == "ORDER BY \"id\" DESC, \"name\" ASC", lines[7] == "LIMIT 5")
stopifnot(build_select("t", "x", dialect = "mysql") == "SELECT `x`\nFROM `t`")

df <- data.frame(id = 1:2, name = c("a", NA))
s <- build_insert("users", df, returning = "id")
stopifnot(s == paste0("INSERT INTO \"users\"\n(\"id\", \"name\")\n",
                      "VALUES (1, 'a'),\n  (2, NULL)\nRETURNING \"id\""))
stopifnot(startsWith(build_insert("u", df, "ignore", dialect = "mysql"), "INSERT IGNORE INTO `u`"))
stopifnot(startsWith(build_insert("u", df, "ignore", dialect = "sqlite"), "INSERT OR IGNORE"))
stopifnot(endsWith(build_insert("u", df, "ignore"), "ON CONFLICT DO NOTHING"))
stopifnot(startsWith(build_insert("u", df, "replace", dialect = "mysql"), "REPLACE INTO"))
stopifnot(startsWith(build_insert("u", df, "replace", dialect = "sqlite"), "INSERT OR REPLACE"))
stopifnot(inherits(try(build_insert("u", df[0, ]), silent = TRUE), "try-error"))
stopifnot(inherits(try(build_insert("u", df, returning = "id", dialect = "mysql"),
                       silent = TRUE), "try-error"))

s <- build_update("users", list(name = "b", active = TRUE), where = "id = 1", returning = "id")
stopifnot(s == paste0("UPDATE \"users\"\nSET \"name\" = 'b', \"active\" = TRUE\n",
                      "WHERE id = 1\nRETURNING \"id\""))
stopifnot(inherits(try(build_update("u", list(1)), silent = TRUE), "try-error"))
stopifnot(inherits(try(build_update("u", list(a = 1), returning = "a", dialect = "mysql"),
                       silent = TRUE), "try-error"))

stopifnot(build_delete("u", "id = 1") == "DELETE FROM \"u\"\nWHERE id = 1")
stopifnot(build_delete("u", allow_all = TRUE, returning = "id", dialect = "sqlite") ==
            "DELETE FROM \"u\"\nRETURNING \"id\"")
stopifnot(inherits(try(build_delete("u"), silent = TRUE), "try-error"))
stopifnot(inherits(try(build_delete("u", "x", returning = "id", dialect = "mysql"),
                       silent = TRUE), "try-error"))

stopifnot(build_join("a", "b", "id") ==
            "FROM \"a\" INNER JOIN \"b\" ON \"a\".\"id\" = \"b\".\"id\"")
stopifnot(build_join("a", "b", c(uid = "id"), "left") ==
            "FROM \"a\" LEFT JOIN \"b\" ON \"a\".\"uid\" = \"b\".\"id\"")
stopifnot(grepl("RIGHT JOIN", build_join("a", "b", "id", "right", "mysql")))
stopifnot(grepl("FULL JOIN", build_join("a", "b", "id", "full")))
stopifnot(build_join("a", "b", type = "cross") == "FROM \"a\" CROSS JOIN \"b\"")
stopifnot(inherits(try(build_join("a", "b", "id", "right", "sqlite"), silent = TRUE), "try-error"))
stopifnot(inherits(try(build_join("a", "b", "id", "full", "mysql"), silent = TRUE), "try-error"))
stopifnot(inherits(try(build_join("a", "b"), silent = TRUE), "try-error"))
stopifnot(inherits(try(build_join("a", "b", "id", "cross"), silent = TRUE), "try-error"))
