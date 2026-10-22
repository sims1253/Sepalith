fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")

# new_registry / assert_registry
reg <- new_registry(a = 1, b = "x")
stopifnot(inherits(reg, "nw_registry"), is.environment(reg))
stopifnot(fails(new_registry(1)), fails(assert_registry(list())))

# registry_set / registry_get / registry_has
registry_set(reg, "c", 3)
stopifnot(registry_get(reg, "c") == 3, is.null(registry_get(reg, "zz")))
stopifnot(registry_get(reg, "zz", default = 0) == 0)
stopifnot(registry_has(reg, "a"))
stopifnot(!registry_has(reg, "zz"))
stopifnot(fails(registry_set(reg, c("p", "q"), 1)), fails(registry_set(reg, "", 1)))

# registry_remove
registry_remove(reg, "a")
registry_remove(reg, "never")
stopifnot(!registry_has(reg, "a"))

# registry_update
registry_update(reg, "n", function(v) v + 1, default = 0)
registry_update(reg, "n", function(v) v + 1, default = 0)
stopifnot(registry_get(reg, "n") == 2)

# registry_snapshot
snap <- registry_snapshot(reg)
stopifnot(identical(names(snap), c("b", "c", "n")), is.list(snap))

# print.nw_registry
out <- capture.output(print(reg))
stopifnot(out[1] == "<nw_registry> 3 entries", any(out == "  b: character"))
stopifnot(capture.output(print(new_registry(k = 1)))[1] == "<nw_registry> 1 entry")
