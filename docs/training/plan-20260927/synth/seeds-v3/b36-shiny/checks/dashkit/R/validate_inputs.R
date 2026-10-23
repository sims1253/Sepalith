lo <- as.Date("2024-01-01")
hi <- as.Date("2024-12-31")
stopifnot(is.null(check_date_range("2024-02-01", "2024-03-01", lo, hi)))
stopifnot(identical(check_date_range(NA, "2024-03-01", lo, hi), "`start` must be a single date."))
stopifnot(identical(check_date_range("2023-12-01", "2024-03-01", lo, hi),
                    "`start` must not be before 2024-01-01."))
stopifnot(identical(check_date_range("2024-02-01", character(0), lo, hi),
                    "`end` must be a single date."))
stopifnot(identical(check_date_range("2024-02-01", "2025-01-05", lo, hi),
                    "`end` must not be after 2024-12-31."))
stopifnot(identical(check_date_range("2024-05-01", "2024-03-01", lo, hi),
                    "`start` must not be after `end`."))

stopifnot(is.null(check_slider_range(c(2, 8), "price", 0, 10)))
stopifnot(identical(check_slider_range(3, "price", 0, 10), "`price` must be two numbers."))
stopifnot(identical(check_slider_range(c(-1, 5), "price", 0, 10),
                    "The lower end of `price` must be at least 0."))
stopifnot(identical(check_slider_range(c(11, 12), "price", 0, 10),
                    "The lower end of `price` must be at most 10."))
stopifnot(identical(check_slider_range(c(1, 12), "price", 0, 10),
                    "The upper end of `price` must be at most 10."))
stopifnot(identical(check_slider_range(c(5, 2), "price", 0, 10),
                    "The ends of `price` are in the wrong order."))

ch <- list(region = c("North", "South"), channel = c("Web", "Store"),
           segment = c("B2B", "B2C"), product = c("Core", "Plus"))
stopifnot(length(check_filter_inputs("North", "Web", "B2B", "Core", ch)) == 0L)
p <- check_filter_inputs(character(0), "Fax", character(0), c("Core", "Max"), ch)
stopifnot(identical(names(p), c("region", "channel", "segment", "product")))
stopifnot(identical(p[["region"]], "Select at least one `region`."))
stopifnot(identical(p[["channel"]], "Unknown `channel`: Fax."))
stopifnot(identical(p[["segment"]], "Select at least one `segment`."))
stopifnot(identical(p[["product"]], "Unknown `product`: Max."))

stopifnot(length(check_numeric_inputs(1000, 500, 0.5, 12)) == 0L)
q <- check_numeric_inputs("a", -1, 2, NA_real_)
stopifnot(identical(names(q), c("target", "budget", "threshold", "horizon")))
stopifnot(identical(q[["target"]], "`target` must be a single number."))
stopifnot(identical(q[["budget"]], "`budget` must lie between 0 and 1e+07."))
stopifnot(identical(q[["threshold"]], "`threshold` must lie between 0 and 1."))
stopifnot(identical(q[["horizon"]], "`horizon` must be a single number."))
stopifnot(identical(check_numeric_inputs(1, 1, 0.1, 60)[["horizon"]],
                    "`horizon` must lie between 1 and 52."))

tmp <- tempfile(fileext = ".csv")
writeLines("a,b", tmp)
ok <- data.frame(name = "sales.csv", size = 2048, type = "text/csv", datapath = tmp)
stopifnot(length(check_upload(ok)) == 0L)
stopifnot(identical(check_upload(NULL), "No file was uploaded."))
bad <- data.frame(name = "sales.xlsx", size = 10 * 1024^2, type = "application/zip",
                  datapath = file.path(tempdir(), "missing-file"))
res <- check_upload(bad)
stopifnot(length(res) == 4L, grepl("^`name`", res[1]), grepl("^`size` must be at most 5 MB",
                                                               res[2]))
stopifnot(grepl("^`type`", res[3]), grepl("^`datapath`", res[4]))

stopifnot(is.null(collect_problems(NULL, character(0))))
stopifnot(identical(collect_problems("a", NULL, c("b", "a")), "a\nb"))
stopifnot(identical(collect_problems("a", "b", sep = "; "), "a; b"))

stopifnot(length(check_text_inputs("Q1 report", "", "ana@example.org", "sales, q1")) == 0L)
tx <- check_text_inputs("  ", NA, "not-an-address", "a,,b")
stopifnot(identical(names(tx), c("title", "note", "email", "tags")))
stopifnot(identical(tx[["title"]], "`title` must not be empty."))
stopifnot(identical(tx[["note"]], "`note` must be a single string."))
stopifnot(identical(tx[["email"]], "`email` must be an e-mail address."))
stopifnot(identical(tx[["tags"]], "`tags` must not contain empty entries."))
long <- check_text_inputs("abcdef", "abcdef", "", "abc, def", max_chars = 5L)
stopifnot(identical(long[["title"]], "`title` must have at most 5 characters."))
stopifnot(identical(long[["note"]], "`note` must have at most 5 characters."))
stopifnot(identical(long[["email"]], "`email` must not be empty."))
stopifnot(identical(long[["tags"]], "`tags` must have at most 5 characters."))
