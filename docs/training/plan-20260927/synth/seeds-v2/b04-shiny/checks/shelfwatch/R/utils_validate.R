# need_positive_int
stopifnot(is.null(need_positive_int(3, "Qty")))
stopifnot(identical(need_positive_int(NA, "Qty"), "Qty is required."))
stopifnot(identical(need_positive_int(2.5, "Qty"), "Qty must be a whole number."))
stopifnot(identical(need_positive_int(0, "Qty"), "Qty must be greater than zero."))

# need_price
stopifnot(is.null(need_price(0.1, "Cost")), is.null(need_price(12.35, "Cost")))
stopifnot(identical(need_price(-1, "Cost"), "Cost cannot be negative."))
stopifnot(grepl("two decimal", need_price(1.234, "Cost")))
stopifnot(identical(need_price("5", "Cost"), "Cost must be a number."))

# need_sku and clean_sku
stopifnot(is.null(need_sku(" bol_0042 ", "SKU")), identical(clean_sku(" bol 0042"), "BOL-0042"))
stopifnot(identical(need_sku("  ", "SKU"), "SKU is required."))
stopifnot(identical(need_sku("BOLT-42", "SKU"), "SKU must look like ABC-1234."))

# need_text and clean_supplier
stopifnot(is.null(need_text(" Acme ", "Supplier")))
stopifnot(identical(need_text("   ", "Supplier"), "Supplier cannot be blank."))
stopifnot(grepl("at most 3", need_text("abcd", "Code", max_chars = 3)))
stopifnot(identical(clean_supplier("  ACME   fasteners "), "Acme Fasteners"))

# collect_errors
stopifnot(identical(collect_errors(NULL, "a", NULL, "b"), c("a", "b")))
stopifnot(identical(collect_errors(NULL, NULL), character()))
