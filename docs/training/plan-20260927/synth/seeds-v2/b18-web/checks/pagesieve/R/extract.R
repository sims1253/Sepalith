# skipped: needs rvest/xml2 (extract_text, extract_attr, extract_number,
# extract_links, extract_meta, extract_records)
stopifnot(
  identical(parse_field_spec(" a.title "), list(css = "a.title", attr = NULL)),
  identical(parse_field_spec("a.title@href"), list(css = "a.title", attr = "href")),
  identical(parse_field_spec("@data-id"), list(css = NULL, attr = "data-id")),
  inherits(try(parse_field_spec("a@b@c"), silent = TRUE), "try-error"),
  inherits(try(parse_field_spec("a@"), silent = TRUE), "try-error")
)
