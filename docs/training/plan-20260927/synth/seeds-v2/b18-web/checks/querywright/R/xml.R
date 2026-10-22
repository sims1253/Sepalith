# skipped: needs xml2 and magrittr (child_text, child_attr, read_sitemap,
# read_sitemap_index, parse_rss_items, parse_atom_entries, read_feed)
stopifnot(
  identical(tz_offset_seconds("Tue, 03 Jun 2025 09:39:21 +0200"), 7200),
  identical(tz_offset_seconds("2025-06-03T09:39:21-05:30"), -19800),
  identical(tz_offset_seconds(c("2025-06-03T09:39:21Z", "x GMT", NA)), c(0, 0, 0))
)
