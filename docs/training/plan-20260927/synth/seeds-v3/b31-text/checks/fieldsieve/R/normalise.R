stopifnot(identical(squish("  a \t b  "), "a b"))
stopifnot(identical(squish(" a  \n  b ", keep_newlines = TRUE), "a\nb"))
stopifnot(identical(squish(c("x", " ")), c("x", "")))

stopifnot(identical(count_changed(c("a", NA, "b"), c("a", NA, "c")), 1L))
stopifnot(identical(count_changed(c("a", "b"), c(NA, "b")), 1L))
stopifnot(identical(count_changed(character(), character()), 0L))

stopifnot(identical(normalise_field("  jANE   o'neil-SMITH ", "name"), "Jane O'Neil-Smith"))
stopifnot(identical(normalise_field(c(a = "bo"), "name"), c(a = "Bo")))
stopifnot(identical(normalise_field(" MAILTO:Ann@Example.ORG ", "email"), "ann@example.org"))
stopifnot(identical(normalise_field("0049 (30) 123-456", "phone"), "+4930123456"))
stopifnot(is.na(normalise_field("n/a", "phone")))
stopifnot(identical(normalise_field(" sw1a 1aa ", "postcode"), "SW1A1AA"))
stopifnot(identical(normalise_field("d.e.", "country"), "DE"))
stopifnot(identical(normalise_field(" ", "country", na_empty = FALSE), ""))

d <- data.frame(
  name = c("ann  lee", "Bob"),
  email = c("A@B.CO", "c@d.io"),
  phone = c("030 12", "+4912"),
  postcode = c("ab1 2cd", "XY9"),
  country = c("de", "FR"),
  other = 1:2,
  stringsAsFactors = FALSE
)
nd <- normalise_contacts(d)
stopifnot(identical(nd$name, c("Ann Lee", "Bob")), identical(nd$country, c("DE", "FR")))
stopifnot(identical(nd$phone, c("03012", "+4912")), identical(nd$postcode, c("AB12CD", "XY9")))
stopifnot(identical(attr(nd, "changed"),
  c(name = 1L, email = 1L, phone = 1L, postcode = 1L, country = 1L)))
stopifnot(identical(names(attr(normalise_contacts(d[c("email", "name")]), "changed")),
  c("name", "email")))

sn <- split_names(c("Lee, Ann", "Ann  Marie Lee", "Prince", NA, " , X"))
stopifnot(identical(sn$family, c("Lee", "Lee", "Prince", NA, NA)))
stopifnot(identical(sn$given, c("Ann", "Ann Marie", NA, NA, "X")))

nd2 <- normalise_dates(c("2024-03-01", "01.03.2024", "1/3/2024", "20240301", "x", NA))
stopifnot(identical(as.vector(nd2), c("2024-03-01", "2024-03-01", NA, "2024-03-01", NA, NA)))
stopifnot(identical(attr(nd2, "format_used")[1:2], c("%Y-%m-%d", "%d.%m.%Y")))
stopifnot(is.na(normalise_dates("2024-03-01", formats = "%d.%m.%Y")))
