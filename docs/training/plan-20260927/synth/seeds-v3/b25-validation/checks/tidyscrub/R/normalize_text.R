stopifnot(identical(squish.text(c("  a  b ", "c\t\nd")), c("a b", "c d")))
stopifnot(identical(squish.text(NA_character_), NA_character_))

stopifnot(identical(change.case(" Ab  C ", "lower"), "ab c"))
stopifnot(identical(change.case("ab c", "upper"), "AB C"))
stopifnot(identical(change.case("hELLO wide-world", "title"), "Hello Wide-World"))
stopifnot(identical(change.case("hELLO World", "sentence"), "Hello world"))
stopifnot(identical(change.case("First Name", "snake"), "first_name"))
stopifnot(identical(change.case("birthDate", "snake"), "birth_date"))

stopifnot(identical(strip.affix(c("id_a", "b", NA), prefix = "id_"), c("a", "b", NA)))
stopifnot(identical(strip.affix("data.csv", suffix = ".csv"), "data"))
stopifnot(identical(strip.affix("ID_x.CSV", "id_", ".csv", ignore.case = TRUE), "x"))
stopifnot(identical(strip.affix("ID_x", prefix = "id_"), "ID_x"))

stopifnot(identical(pad.text(c("7", "123"), 3, pad = "0"), c("007", "123")))
stopifnot(identical(pad.text("ab", 4, "right"), "ab  "))
stopifnot(identical(pad.text("ab", 5, "both", "*"), "*ab**"))
stopifnot(identical(pad.text("abcdef", 3), "abcdef"))

stopifnot(identical(normalize.names(c("First Name", "first name", "2nd", " ", "Zip-Code")),
                    c("first_name", "first_name_1", "x2nd", "column", "zip_code")))
stopifnot(identical(normalize.names("Total Amount", case = "upper"), "TOTAL AMOUNT"))

df <- data.frame(a = c(" x  y", "z"), b = 1:2, c = c("Q", " r"), stringsAsFactors = FALSE)
out <- normalize.text.columns(df)
stopifnot(identical(out$a, c("x y", "z")), identical(out$c, c("Q", "r")), identical(out$b, 1:2))
out2 <- normalize.text.columns(df, columns = "c", case = "lower", width = 2)
stopifnot(identical(out2$c, c(" q", " r")), identical(out2$a, df$a))

stopifnot(identical(change.case("First Name", "kebab"), "first-name"))
stopifnot(identical(change.case("birthDate value", "kebab"), "birth-date-value"))
stopifnot(identical(change.case("first name", "camel"), "firstName"))
stopifnot(identical(change.case("Zip-Code number", "camel"), "zipCodeNumber"))
stopifnot(inherits(try(pad.text("a", 3, pad = "ab"), silent = TRUE), "try-error"))

addr <- data.frame(street = c("  main st. 5", "BERG STR. 2"), city = c("new  york", "BERLIN"),
                   postcode = c(" sw1a 1aa", "10115"), country = c("u.s.", "de"),
                   other = c(" x ", "y"), stringsAsFactors = FALSE)
na <- normalize.address(addr)
stopifnot(identical(na$street, c("Main Street 5", "Berg Strasse 2")))
stopifnot(identical(na$city, c("New York", "Berlin")))
stopifnot(identical(na$postcode, c("SW1A 1AA", "10115")))
stopifnot(identical(na$country, c("US", "DE")), identical(na$other, addr$other))
stopifnot(identical(normalize.address(addr["city"])$city, c("New York", "Berlin")))
stopifnot(identical(normalize.address(addr, abbreviations = character())$street[1],
                    "Main St. 5"))
stopifnot(identical(normalize.names(c("abcdef", "abcxyz"), max.len = 3), c("abc", "abc_1")))
stopifnot(identical(normalize.address(data.frame(houseNumber = c("12 A", " 3")))$houseNumber,
                    c("12a", "3")))
