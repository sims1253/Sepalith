fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(identical(classifyIniLine("  "), "blank"), identical(classifyIniLine("; x"), "comment"))
stopifnot(identical(classifyIniLine("[db]"), "section"))
stopifnot(identical(classifyIniLine("a = 1"), "pair"))
stopifnot(identical(classifyIniLine("oops"), "invalid"))
stopifnot(identical(classifyIniLine("   more"), "continuation"))

path <- tempfile(fileext = ".ini")
writeLines(c("# top", "name = demo", "", "[server]", "port = 80", "port = 8080",
             "motd = hello", "  world", "[db]", "url=sqlite", "[empty]"), path)
cfg <- readIniConfig(path)
stopifnot(identical(names(cfg), c("global", "server", "db", "empty")))
stopifnot(identical(cfg$server$port, "8080"), identical(cfg$db$url, "sqlite"))
stopifnot(identical(cfg$global$name, "demo"), identical(cfg$server$motd, "hello\nworld"))
writeLines(c("[a]", "junk", "k = v"), path)
stopifnot(fails(readIniConfig(path)))
w <- tryCatch(readIniConfig(path, strict = FALSE), warning = conditionMessage)
stopifnot(grepl("Line 2", w))
lax <- suppressWarnings(readIniConfig(path, strict = FALSE))
stopifnot(identical(lax$a$k, "v"))
writeLines(c("[a]", "  orphan"), path)
stopifnot(fails(readIniConfig(path)))

out <- tempfile(fileext = ".ini")
writeIniConfig(cfg, out, header = "generated")
txt <- readLines(out)
stopifnot(identical(txt[1:3], c("# generated", "name = demo", "")), "[server]" %in% txt)
stopifnot("    world" %in% txt, identical(readIniConfig(out), cfg))

stopifnot(identical(iniGet(cfg, "server", "port"), "8080"), identical(iniGet(cfg, "x", "y", 1), 1))
stopifnot(identical(iniGet(cfg, "server", "port", type = "integer"), 8080L))
stopifnot(identical(iniGet(cfg, "server", "port", type = "number"), 8080))
stopifnot(identical(iniGet(list(a = list(b = "On")), "a", "b", type = "boolean"), TRUE))
stopifnot(identical(iniGet(cfg, "server", "motd", type = "list"), c("hello", "world")))
stopifnot(fails(iniGet(cfg, "db", "url", type = "integer")))
stopifnot(fails(iniGet(cfg, "db", "url", type = "boolean")))

cfg2 <- iniSet(cfg, "cache", "dir", "/tmp")
stopifnot(identical(cfg2$cache$dir, "/tmp"))
stopifnot(identical(iniSet(cfg, "db", "url", "pg")$db$url, "pg"))
stopifnot(is.null(iniSet(cfg, "db", "url", NULL)$db$url))

ic <- list(paths = list(root = "/srv", data = "${root}/data", logs = "${data}/logs"),
           app = list(cache = "${paths:data}/cache"))
r <- interpolateIni(ic)
stopifnot(identical(r$paths$logs, "/srv/data/logs"), identical(r$app$cache, "/srv/data/cache"))
stopifnot(fails(interpolateIni(list(a = list(x = "${nope}")))))
stopifnot(fails(interpolateIni(list(a = list(x = "${y}", y = "${x}")))))

sch <- list(server = c(port = "integer", ratio = "number", debug = "boolean",
                       motd = "list", name = "string", host = "string"))
ok <- list(server = list(port = "80", ratio = "0.5", debug = "yes", motd = "a,b", name = "x"))
stopifnot(identical(checkIniTypes(ok, sch), character(0)))
stopifnot(identical(checkIniTypes(ok, sch, required = TRUE), "[server] host is missing."))
bad <- list(server = list(port = "8o", ratio = "x", debug = "maybe", motd = "a,,b"))
stopifnot(identical(checkIniTypes(bad, sch), c("[server] port is not a valid integer.",
                                               "[server] ratio is not a valid number.",
                                               "[server] debug is not a valid boolean.",
                                               "[server] motd is not a valid list.")))
stopifnot(fails(checkIniTypes(ok, list(server = c(port = "blob")))))
