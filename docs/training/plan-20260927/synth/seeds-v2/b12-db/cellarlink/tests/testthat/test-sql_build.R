con <- DBI::ANSI()

test_that("quote_cols quotes and joins column names", {
  sql <- quote_cols(con, c("id", "order date"))
  expect_s4_class(sql, "SQL")
  expect_equal(as.character(sql), '"id", "order date"')
  expect_error(quote_cols(con, character()), "at least one column")
})

test_that("quote_table splits schema-qualified names", {
  expect_equal(as.character(quote_table(con, "sales.orders")), '"sales"."orders"')
  expect_equal(as.character(quote_table(con, "orders")), '"orders"')
})

test_that("build_select handles columns and filters", {
  sql <- build_select(con, "orders", cols = c("id", "total"))
  expect_equal(as.character(sql), 'SELECT "id", "total" FROM "orders"')

  sql <- build_select(con, "orders", where = list(status = "open"))
  expect_equal(as.character(sql), "SELECT * FROM \"orders\" WHERE \"status\" = 'open'")
})

test_that("build_delete refuses to run without a filter", {
  expect_error(build_delete(con, "orders", list()), "without a WHERE")
  sql <- build_delete(con, "orders", list(id = 3L))
  expect_equal(as.character(sql), 'DELETE FROM "orders" WHERE "id" = 3')
})

test_that("build_update refuses to run without a filter", {
  expect_error(build_update(con, "orders", list(total = 10), list()), "without a WHERE")
  sql <- build_update(con, "orders", list(total = 10), list(id = 3L))
  expect_equal(as.character(sql), 'UPDATE "orders" SET "total" = 10 WHERE "id" = 3')
})

test_that("where_clause covers scalars, vectors and missing values", {
  sql <- where_clause(con, list(a = 1L, b = c("x", "y"), c = NA))
  expect_equal(
    as.character(sql),
    "\"a\" = 1 AND \"b\" IN ('x', 'y') AND \"c\" IS NULL"
  )
  expect_error(where_clause(con, list(1, 2)), "must be named")
  expect_error(where_clause(con, list(a = character())), "has no values")
})

test_that("date_range_sql leaves open bounds out", {
  expect_equal(as.character(date_range_sql(con, "day")), "1 = 1")
  expect_equal(
    as.character(date_range_sql(con, "day", from = "2024-01-01")),
    "\"day\" >= '2024-01-01'"
  )
  expect_equal(
    as.character(date_range_sql(con, "day", from = "2024-01-01", to = "2024-02-01")),
    "\"day\" >= '2024-01-01' AND \"day\" < '2024-02-01'"
  )
})

test_that("upsert_on_conflict updates existing rows in SQLite", {
  skip_if_not_installed("RSQLite")
  db <- connect_sqlite()
  on.exit(DBI::dbDisconnect(db), add = TRUE)

  DBI::dbExecute(db, "CREATE TABLE stock (sku TEXT PRIMARY KEY, qty INTEGER)")
  DBI::dbWriteTable(db, "stock", data.frame(sku = c("a", "b"), qty = c(1L, 2L)), append = TRUE)
  # "b" exists and is updated, "c" is new
  upsert_on_conflict(db, "stock", data.frame(sku = c("b", "c"), qty = c(20L, 30L)), "sku")

  out <- DBI::dbGetQuery(db, "SELECT sku, qty FROM stock ORDER BY sku")
  expect_equal(out$qty, c(1L, 20L, 30L))
})
