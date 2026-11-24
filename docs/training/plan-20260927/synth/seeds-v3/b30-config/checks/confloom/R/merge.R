fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
d <- list(a = 1, b = list(c = 2, d = 3))
m <- mergeDefaults(list(b = list(c = 9), e = 5), d)
stopifnot(identical(m, list(a = 1, b = list(c = 9, d = 3), e = 5)))
stopifnot(identical(mergeDefaults(NULL, d), d))
stopifnot(identical(mergeDefaults(list(a = NULL), d)$a, NULL))
w <- tryCatch(mergeDefaults(list(b = list(c = list(x = 1))), d), warning = conditionMessage)
stopifnot(identical(w, "Setting 'b.c' changes shape from its default."))

x <- list(a = 1, b = list(c = 1))
y <- list(a = 2, b = list(d = 2), z = 3)
stopifnot(identical(mergeConfigs(x, y), list(a = 2, b = list(d = 2), z = 3)))
stopifnot(identical(mergeConfigs(x, y, policy = "first"), list(a = 1, b = list(c = 1), z = 3)))
stopifnot(fails(mergeConfigs(x, y, policy = "error")))
stopifnot(identical(mergeConfigs(x, list(z = 3), policy = "error"), c(x, list(z = 3))))
stopifnot(identical(mergeConfigs(x, y, policy = "deep"),
                    list(a = 2, b = list(c = 1, d = 2), z = 3)))

cd <- configDefaults()
stopifnot(identical(names(cd), c("server", "database", "logging", "cache")))
stopifnot(identical(cd$server$port, 8080L))

cfg <- applySectionDefaults(list(server = list(port = "9000"), database = list(pool = "3"),
                                 logging = list(rotate = "TRUE"), cache = list(ttl = "60"),
                                 extra = list(x = 1)))
stopifnot(identical(cfg$server$port, 9000L), identical(cfg$server$host, "127.0.0.1"))
stopifnot(identical(cfg$database$pool, 3L), identical(cfg$database$timeout, 30))
stopifnot(isTRUE(cfg$logging$rotate), identical(cfg$extra, list(x = 1)))
stopifnot(identical(cfg$cache$ttl, 60), isTRUE(cfg$cache$enabled))
w <- character()
withCallingHandlers(applySectionDefaults(list(server = list(bad = 1), database = list(nope = 2),
                                              logging = list(huh = 3), cache = list(eh = 4))),
                    warning = function(cnd) {
                      w <<- c(w, conditionMessage(cnd))
                      invokeRestart("muffleWarning")
                    })
stopifnot(identical(w, c("Unknown key(s) in [server]: bad", "Unknown key(s) in [database]: nope",
                         "Unknown key(s) in [logging]: huh", "Unknown key(s) in [cache]: eh")))

nested <- list(a = 1, b = list(c = 2, d = list(e = 3)), f = list())
flat <- flattenConfig(nested)
stopifnot(identical(names(flat), c("a", "b.c", "b.d.e", "f")))
stopifnot(identical(unflattenConfig(flat), nested))
stopifnot(identical(unflattenConfig(list(x_y = 1), sep = "_"), list(x = list(y = 1))))

dd <- diffConfig(list(a = 1, b = list(c = 2)), list(b = list(c = 3), z = 0))
stopifnot(identical(dd$key, c("a", "b.c", "z")))
stopifnot(identical(dd$change, c("removed", "changed", "added")))
stopifnot(identical(dd$old, c("1", "2", NA)), identical(dd$new, c(NA, "3", "0")))
stopifnot(nrow(diffConfig(x, x)) == 0L)
