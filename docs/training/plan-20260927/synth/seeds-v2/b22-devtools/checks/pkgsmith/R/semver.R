# skipped: use_version needs desc_get_field() and desc_set_field() from
# description.R, which is not sourced here

# semver_parse
v <- semver_parse(" 1.2 ")
stopifnot(identical(c(v$major, v$minor, v$patch, v$dev), c(1L, 2L, 0L, 0L)), is.na(v$pre))
stopifnot(identical(semver_parse("2.0.0-rc.1")$pre, "rc.1"))
stopifnot(inherits(try(semver_parse("1.x"), silent = TRUE), "try-error"))

# semver_format
stopifnot(identical(semver_format(semver_parse("0.3.2.9000")), "0.3.2.9000"))
stopifnot(identical(semver_format(semver_parse("3.1.4-beta")), "3.1.4-beta"))

# semver_compare_pre / semver_compare
stopifnot(semver_compare_pre("rc.2", "rc.10") == -1L, semver_compare_pre("1", "alpha") == -1L)
stopifnot(identical(semver_compare_pre("rc", "rc.1"), -1L))
stopifnot(identical(semver_compare("1.0.0-rc.10", "1.0.0-rc.2"), 1L))
stopifnot(identical(semver_compare("1.0.0-rc.1", "1.0.0"), -1L))
stopifnot(identical(semver_compare("1.0.0.9000", "1.0.0"), 1L))
stopifnot(identical(semver_compare("1.2.0", "1.2"), 0L))

# bump_major / bump_minor / bump_patch / bump_dev
stopifnot(identical(bump_major("0.9.1.9000"), "1.0.0"), identical(bump_minor("1.4.2"), "1.5.0"))
stopifnot(identical(bump_patch("1.0.0-rc.1"), "1.0.1"), identical(bump_dev("1.4.2"), "1.4.2.9000"))
stopifnot(identical(bump_dev("1.4.2.9000"), "1.4.2.9001"))
