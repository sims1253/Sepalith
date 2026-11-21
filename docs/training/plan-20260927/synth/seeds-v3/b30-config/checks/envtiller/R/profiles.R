# skipped: needs envtiller (env.override() from env.R): profile.resolve()
for (p in c("dev", "test", "staging", "prod")) {
  s <- profile.settings(p)
  stopifnot(identical(names(s), c("server", "log", "cache")), length(profile.check(s)) == 0L)
}
stopifnot(identical(profile.settings("prod")$server$port, 80L))
stopifnot(isTRUE(profile.settings()$server$debug))

old <- Sys.getenv(c("TP_PROFILE", "CI"), unset = NA)
Sys.setenv(TP_PROFILE = "Prod", CI = "false")
stopifnot(identical(profile.detect("TP_PROFILE"), "prod"))
Sys.unsetenv("TP_PROFILE")
stopifnot(identical(profile.detect("TP_PROFILE"), "dev"))
Sys.setenv(CI = "true")
stopifnot(identical(profile.detect("TP_PROFILE"), "test"))
Sys.unsetenv("CI")

bad <- list(server = list(host = "", port = 70000, debug = NA),
            log = list(level = "loud", file = 1), cache = list(dir = "", max.mb = 0))
p <- profile.check(bad)
stopifnot(length(p) == 7L, identical(p[[1]], "Setting `server.host` must be a non-empty string."))
stopifnot(identical(p[[7]], "Setting `cache.max.mb` must be a positive number."))

s <- profile.settings("prod")
attr(s, "sources") <- c(server.host = "env")
d <- profile.describe(s)
stopifnot(identical(d[[1]], "server.host = 0.0.0.0 (env)"))
stopifnot(identical(d[[2]], "server.port = 80 (unknown)"))
stopifnot(length(d) == 7L)
