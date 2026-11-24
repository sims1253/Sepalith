fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")

col <- schemaColumn("numeric", nullable = FALSE)
stopifnot(identical(col, list(type = "numeric", required = TRUE, nullable = FALSE)))
stopifnot(identical(schemaColumn()$type, "character"))
stopifnot(fails(schemaColumn("complex")))
stopifnot(fails(schemaColumn(required = NA)), fails(schemaColumn(nullable = "yes")))

s <- newSchema(id = schemaColumn("character", nullable = FALSE),
               n = schemaColumn("integer"),
               when = schemaColumn("Date", required = FALSE))
stopifnot(inherits(s, "sieve_schema"), identical(names(s), c("id", "n", "when")))
stopifnot(fails(newSchema(schemaColumn())))
stopifnot(fails(newSchema(a = schemaColumn(), a = schemaColumn())))
stopifnot(fails(newSchema(a = schemaColumn(), b = list(type = "numeric"))))

stopifnot(columnHasType("a", "character"), columnHasType(1.5, "numeric"))
stopifnot(columnHasType(c(1, 2), "integer"), !columnHasType(1.5, "integer"))
stopifnot(columnHasType(NA, "logical"), columnHasType(Sys.Date(), "Date"))
stopifnot(!columnHasType("2024-01-01", "Date"), !columnHasType(1, "blob"))

df <- data.frame(id = c("a", "b"), n = c(1L, NA), x = c(0.5, 1), flag = c(TRUE, FALSE),
                 stringsAsFactors = FALSE)
df$when <- as.Date(c("2024-01-01", "2024-01-02"))
inf <- inferSchema(df)
stopifnot(identical(vapply(inf, `[[`, "", "type"),
                    c(id = "character", n = "integer", x = "numeric", flag = "logical",
                      when = "Date")))
stopifnot(isTRUE(inf$n$nullable), isFALSE(inf$id$nullable))

res <- checkSchema(df, s)
stopifnot(identical(names(res), c("missing", "unexpected", "wrongType", "nullViolations",
                                  "empty", "ok")))
stopifnot(identical(res$unexpected, c("x", "flag")), isFALSE(res$ok))
stopifnot(isTRUE(checkSchema(df, s, allowExtra = TRUE)$ok))
bad <- data.frame(id = c(NA, "b"), n = c("1", "2"), stringsAsFactors = FALSE)
res2 <- checkSchema(bad, newSchema(id = schemaColumn(nullable = FALSE),
                                   n = schemaColumn("integer"),
                                   z = schemaColumn("logical")))
stopifnot(identical(res2$missing, "z"), identical(res2$wrongType, "n"))
stopifnot(identical(res2$nullViolations, "id"))
stopifnot(fails(checkSchema(df, list())))

d <- schemaDiff(s, newSchema(id = schemaColumn("character"), n = schemaColumn("integer"),
                             extra = schemaColumn()))
stopifnot(identical(d$added, "extra"), identical(d$removed, "when"), identical(d$changed, "id"))
stopifnot(identical(d$details, c(id = "nullable FALSE -> TRUE")))
stopifnot(identical(length(schemaDiff(s, s)$changed), 0L))
d2 <- schemaDiff(s, newSchema(id = schemaColumn("numeric"), n = schemaColumn("integer", FALSE),
                              when = schemaColumn("Date", required = FALSE)))
stopifnot(identical(d2$details, c(id = "type character -> numeric", n = "required TRUE -> FALSE")))

out <- capture.output(print(s))
stopifnot(length(out) == 4L, grepl("3 column", out[1]), grepl("required, not null", out[2]))
res3 <- checkSchema(data.frame(id = c("a", "b"), n = c(NA_integer_, NA)), s, allowExtra = TRUE)
stopifnot(identical(res3$empty, "n"), isFALSE(res3$ok), identical(res3$wrongType, character()))

raw <- data.frame(n = c("1", "x"), id = factor(c("a", "b")), flag = c("TRUE", "no"),
                  stringsAsFactors = FALSE)
sc <- newSchema(id = schemaColumn("character"), n = schemaColumn("integer"),
                x = schemaColumn("numeric", required = FALSE),
                flag = schemaColumn("logical", required = FALSE),
                when = schemaColumn("Date", required = FALSE))
co <- coerceToSchema(raw, sc)
stopifnot(identical(names(co), c("id", "n", "x", "flag", "when")))
stopifnot(identical(co$id, c("a", "b")), identical(co$n, c(1L, NA)))
stopifnot(identical(co$x, c(NA_real_, NA_real_)), identical(co$flag, c(TRUE, NA)))
stopifnot(inherits(co$when, "Date"), all(is.na(co$when)))
stopifnot(fails(coerceToSchema(raw[, "n", drop = FALSE], sc)))
co2 <- coerceToSchema(data.frame(id = "a", n = 2, when = "2024-01-01"), sc)
stopifnot(identical(co2$when, as.Date("2024-01-01")))
