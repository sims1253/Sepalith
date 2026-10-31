stopifnot(strip.sql.noise("select 'a;b' -- note\nfrom t") == "SELECT '' FROM T")
stopifnot(strip.sql.noise("select /* x */ \"Odd \"\"Name\" from t") == "SELECT \"\" FROM T")
stopifnot(strip.sql.noise("  a   b ") == "A B")

v <- validate.query("SELECT * FROM t")
stopifnot(identical(v$rule[1], "no_select_star"), !v$passed[1], v$passed[2], v$passed[3])
stopifnot(!v$passed[4], v$passed[5], v$passed[6])
stopifnot(v$message[1] == "Query violates rule `no_select_star`.", v$message[2] == "")
stopifnot(validate.query("SELECT a FROM t LIMIT 5", "select_needs_limit")$passed)
stopifnot(validate.query("SELECT '*' AS x FROM t", "no_select_star")$passed)
stopifnot(!validate.query("delete from t", "delete_needs_where")$passed)
stopifnot(validate.query("delete from t where id = 1", "delete_needs_where")$passed)
stopifnot(!validate.query("update t set a = 1", "update_needs_where")$passed)
stopifnot(validate.query("update t set a = 1 where b = 2", "update_needs_where")$passed)
stopifnot(!validate.query("select 1; drop table t", "single_statement")$passed)
stopifnot(validate.query("select ';' from t;", "single_statement")$passed)
stopifnot(!validate.query("select (1 from t", "balanced_parens")$passed)
stopifnot(!validate.query("select )1( from t", "balanced_parens")$passed)
stopifnot(validate.query("select (1) from t", "balanced_parens")$passed)
stopifnot(inherits(try(validate.query(c("a", "b")), silent = TRUE), "try-error"))

f <- check.identifiers(c("ok_name", strrep("a", 64), "2col", "has-dash", "Mixed", "user"))
stopifnot(nrow(f) == 5L, f$name[f$rule == "max_length"] == strrep("a", 64))
stopifnot(f$name[f$rule == "leading_letter"] == "2col")
stopifnot(identical(f$name[f$rule == "plain_characters"], "has-dash"))
stopifnot(f$name[f$rule == "lower_case"] == "Mixed", f$name[f$rule == "not_reserved"] == "user")
stopifnot(nrow(check.identifiers(strrep("a", 64), "mysql")) == 0L)
stopifnot(nrow(check.identifiers("ok")) == 0L)

stopifnot(identical(check.params("select ? , '?' from t where a = ?", list(1, 2)), list(1, 2)))
stopifnot(inherits(try(check.params("select ?", list()), silent = TRUE), "try-error"))
stopifnot(length(check.params("select $1, $2", list(1, 2), "dollar")) == 2L)
msg <- tryCatch(check.params("select $3", list(1), "dollar"), error = conditionMessage)
stopifnot(msg == "Query has 3 placeholder(s) but 1 parameter(s).")
p <- check.params("select :b, :a, x::text, :b", list(a = 1, b = 2, c = 3), "colon")
stopifnot(identical(p, list(b = 2, a = 1)))
stopifnot(inherits(try(check.params("select :z", list(a = 1), "colon"), silent = TRUE),
                   "try-error"))
stopifnot(identical(check.params("select @id", list(id = 5), "at"), list(id = 5)))
msg <- tryCatch(check.params("select @x, @y", list(x = 1), "at"), error = conditionMessage)
stopifnot(msg == "Missing parameter(s): y.")

spec <- data.frame(name = c("id", "email"), type = c("integer", "text"),
                   nullable = c(FALSE, TRUE), default = c(NA, ""))
stopifnot(identical(validate.column.spec(spec), spec))
bad <- spec
bad$name[2] <- "Email"
msg <- tryCatch(validate.column.spec(bad), error = conditionMessage)
stopifnot(msg == "`spec$name` is invalid in row(s) 2.")
bad <- spec
bad$type[1] <- "uuid"
stopifnot(grepl("spec$type", tryCatch(validate.column.spec(bad), error = conditionMessage),
                fixed = TRUE))
bad <- spec
bad$nullable[2] <- NA
stopifnot(inherits(try(validate.column.spec(bad), silent = TRUE), "try-error"))
bad <- spec
bad$default[1] <- ""
stopifnot(grepl("spec$default", tryCatch(validate.column.spec(bad), error = conditionMessage),
                fixed = TRUE))
stopifnot(inherits(try(validate.column.spec(spec[, 1:2]), silent = TRUE), "try-error"))
# Placeholders inside string literals or comments are not parameters.
ok <- check.params("SELECT ':ghost' AS x WHERE id = :id", list(id = 1), "colon")
stopifnot(identical(ok, list(id = 1)))
ok <- check.params("SELECT 1 -- :ghost\nWHERE id = :id", list(id = 1), "colon")
stopifnot(identical(ok, list(id = 1)))
ok <- check.params("SELECT '@ghost' AS x WHERE id = @id", list(id = 1), "at")
stopifnot(identical(ok, list(id = 1)))
