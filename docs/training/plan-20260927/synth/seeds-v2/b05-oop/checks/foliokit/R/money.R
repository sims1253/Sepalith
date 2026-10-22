# skipped: needs vctrs (new_money, validate_money, money, format.foliokit_money)
stopifnot(identical(minor_units("JPY"), 0L))
stopifnot(identical(minor_units("KWD"), 3L))
stopifnot(identical(minor_units("EUR"), 2L))

fake <- structure(1.5, currency = "EUR", class = "foliokit_money")
stopifnot(is_money(fake), !is_money(1.5))
stopifnot(identical(currency(fake), "EUR"))
stopifnot(inherits(try(currency(1.5), silent = TRUE), "try-error"))

usd <- structure(2, currency = "USD", class = "foliokit_money")
stopifnot(identical(check_same_currency(fake, fake), "EUR"))
stopifnot(inherits(try(check_same_currency(fake, usd), silent = TRUE), "try-error"))
stopifnot(identical(vec_ptype_abbr.foliokit_money(fake), "eur"))
