listing_html <- function() {
  rvest::minimal_html('
    <ul class="listing">
      <li class="item"><a class="title" href="/p/1">First   post</a>
        <span class="price">$1,200.50</span></li>
      <li class="item"><a class="title" href="/p/2#top">Second post</a>
        <span class="price">n/a</span></li>
      <li class="item"><span class="price">$15</span></li>
    </ul>
    <a href="mailto:someone@example.org">mail</a>
    <a href="https://other.example.net/x">elsewhere</a>
  ')
}

test_that("extract_text collapses whitespace and handles no matches", {
  html <- listing_html()
  expect_equal(extract_text(html, "a.title"), c("First post", "Second post"))
  expect_identical(extract_text(html, "h1"), NA_character_)
  expect_identical(extract_text(html, "h1", missing = ""), "")
})

test_that("extract_attr returns attribute values per element", {
  html <- listing_html()
  expect_equal(extract_attr(html, "a.title", "href"), c("/p/1", "/p/2#top"))
  expect_identical(extract_attr(html, "a.title", "data-id"), c(NA_character_, NA_character_))
  expect_identical(extract_attr(html, "h1", "href"), NA_character_)
})

test_that("extract_number strips currency and separators", {
  html <- listing_html()
  expect_equal(extract_number(html, ".price"), c(1200.5, NA, 15))
  expect_identical(extract_number(html, "h1"), NA_real_)
  euro <- rvest::minimal_html('<p class="v">1.234,5 EUR</p>')
  expect_equal(extract_number(euro, ".v", decimal_mark = ","), 1234.5)
})

test_that("extract_links resolves, deduplicates and filters links", {
  html <- listing_html()
  links <- extract_links(html, "https://blog.example.org/index.html")
  expect_true("https://blog.example.org/p/1" %in% links)
  expect_true("https://blog.example.org/p/2" %in% links)
  expect_false(any(grepl("^mailto:", links)))
  local <- extract_links(html, "https://blog.example.org/", same_host = TRUE)
  expect_false("https://other.example.net/x" %in% local)
  expect_length(local, 2)
})

test_that("extract_meta collects title and repeated meta tags", {
  html <- rvest::minimal_html('
    <meta name="Description" content="A test page">
    <meta property="og:image" content="a.png">
    <meta property="og:image" content="b.png">
    <meta charset="utf-8">
  ', title = "Demo")
  meta <- extract_meta(html)
  expect_equal(meta$title, "Demo")
  expect_equal(meta$description, "A test page")
  expect_equal(meta[["og:image"]], c("a.png", "b.png"))
  expect_null(meta$charset)
})

test_that("parse_field_spec splits selector and attribute", {
  expect_equal(parse_field_spec("a.title"), list(css = "a.title", attr = NULL))
  expect_equal(parse_field_spec("a.title@href"), list(css = "a.title", attr = "href"))
  expect_equal(parse_field_spec("@data-id"), list(css = NULL, attr = "data-id"))
  expect_error(parse_field_spec("a@b@c"), "Invalid field spec")
})

test_that("extract_records builds one row per item", {
  html <- listing_html()
  records <- extract_records(
    html,
    "li.item",
    c(title = "a.title", link = "a.title@href", price = ".price")
  )
  expect_s3_class(records, "data.frame")
  expect_equal(nrow(records), 3)
  expect_equal(names(records), c("title", "link", "price"))
  expect_equal(records$title, c("First post", "Second post", NA))
  expect_equal(records$price[3], "$15")
  expect_error(extract_records(html, "li.item", c("a.title")), "fully named")
})
