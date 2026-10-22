# skipped: deps_list, deps_add, use_import, use_suggest, use_linking_to and
# deps_duplicates need desc_get_field() and desc_set_field() from
# description.R, which is not sourced here

# deps_parse
d <- deps_parse("R (>= 4.1), dplyr (>=1.1.0),\n    rlang, ", "Imports")
stopifnot(identical(d$package, c("R", "dplyr", "rlang")), all(d$type == "Imports"))
stopifnot(identical(d$op, c(">=", ">=", NA)), identical(d$version, c("4.1", "1.1.0", NA)))
stopifnot(nrow(deps_parse(NA_character_, "Suggests")) == 0L)
odd <- deps_parse("pkg (1.0)", "Imports")
stopifnot(is.na(odd$op), identical(odd$version, "1.0"))

# deps_format
txt <- deps_format(d[c(3, 2, 1), ])
stopifnot(identical(txt, "\n    R (>= 4.1),\n    dplyr (>= 1.1.0),\n    rlang"))
stopifnot(is.na(deps_format(d[0, ])))
