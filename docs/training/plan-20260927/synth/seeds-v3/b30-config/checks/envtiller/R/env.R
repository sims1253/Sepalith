fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(identical(env.var.name("tiller", "cache", "max.mb"), "TILLER_CACHE_MAX_MB"))
stopifnot(identical(env.var.name("App"), "APP"))

Sys.setenv(ET_F = "yes", ET_I = "42", ET_N = "2.5", ET_S = "  hi ", ET_BAD = "x1", ET_L = "a, b,")
Sys.unsetenv("ET_NONE")
stopifnot(isTRUE(env.get("ET_F", "flag")), identical(env.get("ET_NONE", "flag", TRUE), TRUE))
stopifnot(fails(env.get("ET_BAD", "flag")))
stopifnot(identical(env.get("ET_I", "int"), 42L), identical(env.get("ET_NONE", "int", 1L), 1L))
stopifnot(fails(env.get("ET_N", "int")))
stopifnot(identical(env.get("ET_N", "num"), 2.5), is.null(env.get("ET_NONE", "num")))
stopifnot(fails(env.get("ET_BAD", "num")))
stopifnot(identical(env.get("ET_S"), "hi"), identical(env.get("ET_NONE", "str", "d"), "d"))
stopifnot(identical(env.get("ET_L", "list"), c("a", "b")))

cfg <- list(server = list(host = "h", port = 1L, workers = 2L, debug = FALSE),
            log = list(level = "info", file = "", tags = "x"),
            cache = list(dir = "/c", max.mb = 10, ttl = 5))
vars <- c(T1_SERVER_HOST = "example", T1_SERVER_PORT = "99", T1_SERVER_WORKERS = "4",
          T1_SERVER_DEBUG = "on", T1_LOG_LEVEL = "warn", T1_LOG_FILE = "x.log",
          T1_LOG_TAGS = "a,b", T1_CACHE_DIR = "/d", T1_CACHE_MAX_MB = "5.5",
          T1_CACHE_TTL = "60")
res <- with.env.vars(vars, env.override(cfg, "T1"))
stopifnot(identical(res$server, list(host = "example", port = 99L, workers = 4L, debug = TRUE)))
stopifnot(identical(res$log, list(level = "warn", file = "x.log", tags = c("a", "b"))))
stopifnot(identical(res$cache, list(dir = "/d", max.mb = 5.5, ttl = 60)))
stopifnot(identical(env.override(cfg, "T1"), cfg))

Sys.setenv(ET_KEEP = "a")
Sys.unsetenv("ET_GONE")
inner <- with.env.vars(c(ET_KEEP = NA, ET_GONE = "z"),
                       c(Sys.getenv("ET_KEEP", NA), Sys.getenv("ET_GONE")))
stopifnot(is.na(inner[[1]]), identical(inner[[2]], "z"))
stopifnot(identical(Sys.getenv("ET_KEEP"), "a"), is.na(Sys.getenv("ET_GONE", unset = NA)))
try(with.env.vars(c(ET_KEEP = "b"), stop("boom")), silent = TRUE)
stopifnot(identical(Sys.getenv("ET_KEEP"), "a"))

rep <- with.env.vars(c(T2_SERVER_PORT = "80", T2_DB_PASSWORD = "pw", T2_LOG_LEVEL = NA),
                     env.report(c("server.port", "db.password", "log.level"), "T2"))
stopifnot(identical(rep$variable, c("T2_SERVER_PORT", "T2_DB_PASSWORD", "T2_LOG_LEVEL")))
stopifnot(identical(rep$set, c(TRUE, TRUE, FALSE)), identical(rep$value, c("80", "<hidden>", NA)))

un <- with.env.vars(c(QXCHK_SERVER_PROT = "1", QXCHK_SERVER_PORT = "2", QXCHK_ZZZZZZZZZZZZ = "3"),
                    env.unused(c("server.port", "log.level"), "QXCHK"))
stopifnot(identical(un[["QXCHK_SERVER_PROT"]], "QXCHK_SERVER_PORT"))
stopifnot(is.na(un[["QXCHK_ZZZZZZZZZZZZ"]]))
stopifnot(length(un) == 2L)
