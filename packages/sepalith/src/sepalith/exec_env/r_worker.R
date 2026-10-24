# Persistent R execution worker for the execution-grounded reward (sepalith.exec_env.harness).
#
# Runs inside sandbox.sh (no network, read-only filesystem except a private tmpfs).
# Protocol on stdin/stdout, one job at a time:
#   request:  one JSON header line {"id":"..","timeout":2.0,"n_src":N,"n_test":M,"expect":"a,b,f"}
#             followed by exactly N bytes of candidate source and M bytes of test script (UTF-8).
#   response: one JSON line {"id":..,"status":..,"parse_ok":..,"structure_ok":..,"sourced_ok":..,
#             "checks":"cat\t1\ncat\t0...","n_checks":..,"error_class":..,"error":..,"elapsed":..}
# "expect" lists the top-level assignment names the candidate file must define, in order
# (empty = no structure check). Each job runs in a forked child (parallel::mcparallel), so a
# candidate cannot leak state into later jobs, and quit(), crashes, hangs and runaway memory only
# end the child: the parent reports candidate_exit or timeout and kills it.
# Test scripts use the assertion helpers of testlib.R (check, check_equal, check_identical,
# check_error, check_warning); the counts come only from those helpers, never from exit status.

local({
  args <- commandArgs(trailingOnly = TRUE)
  testlib_path <- args[[1]]
  testlib_src <- paste(readLines(testlib_path, warn = FALSE), collapse = "\n")
  suppressPackageStartupMessages({
    for (p in c("stats", "utils", "methods", "tools", "parallel")) loadNamespace(p)
  })
  con <- file("stdin", open = "rb")
  out <- stdout()

  read_exact <- function(n) {
    if (n == 0) return("")
    chunks <- list()
    got <- 0L
    while (got < n) {
      part <- readBin(con, "raw", n - got)
      if (length(part) == 0) stop("eof")
      chunks[[length(chunks) + 1L]] <- part
      got <- got + length(part)
    }
    x <- rawToChar(do.call(c, chunks))
    Encoding(x) <- "UTF-8"
    x
  }
  read_header <- function() {
    buf <- raw(0)
    repeat {
      b <- readBin(con, "raw", 1L)
      if (length(b) == 0) return(NULL)
      if (b == as.raw(10L)) break
      buf <- c(buf, b)
    }
    rawToChar(buf)
  }
  field <- function(h, key, numeric = TRUE) {
    pat <- if (numeric) sprintf('"%s":\\s*([-0-9.eE]+)', key) else sprintf('"%s":\\s*"((?:[^"\\\\]|\\\\.)*)"', key)
    m <- regmatches(h, regexec(pat, h, perl = TRUE))[[1]]
    if (length(m) < 2) return(if (numeric) NA_real_ else "")
    if (numeric) as.numeric(m[[2]]) else m[[2]]
  }
  esc <- function(x) {
    x <- gsub("\\\\", "\\\\\\\\", x)
    x <- gsub('"', '\\\\"', x)
    x <- gsub("\n", "\\\\n", x)
    x <- gsub("\t", "\\\\t", x)
    x <- gsub("\r", "\\\\r", x)
    gsub("[[:cntrl:]]", " ", x)
  }
  js <- function(x) {
    if (is.null(x) || (length(x) == 1 && is.na(x))) return("null")
    if (is.logical(x)) return(if (x) "true" else "false")
    if (is.numeric(x)) return(format(x, digits = 6, scientific = FALSE))
    paste0('"', esc(enc2utf8(as.character(x))), '"')
  }
  emit <- function(res) {
    parts <- vapply(names(res), function(k) paste0('"', k, '":', js(res[[k]])), "")
    writeLines(paste0("{", paste(parts, collapse = ","), "}"), out, useBytes = TRUE)
    flush(out)
  }

  child <- function(src, test, expect, timeout) {
    # Discard everything the candidate or the tests print (bounded memory, no reliance on output).
    sink(nullfile())
    sink(file(nullfile(), open = "w"), type = "message")
    setTimeLimit(elapsed = timeout, transient = FALSE)
    res <- list(parse_ok = FALSE, structure_ok = FALSE, sourced_ok = FALSE, checks = "", n_checks = 0L,
                error_class = NA, error = NA)
    fail <- function(cls, e) {
      res$error_class <- cls
      res$error <- substr(conditionMessage(e), 1, 300)
      res
    }
    exprs <- tryCatch(parse(text = src, keep.source = FALSE), error = function(e) e)
    if (inherits(exprs, "error")) return(fail("parse_error", exprs))
    res$parse_ok <- TRUE
    # Structure: the candidate defines exactly the expected top-level assignments, in order.
    if (nzchar(expect)) {
      want <- strsplit(expect, ",", fixed = TRUE)[[1]]
      got <- vapply(as.list(exprs), function(e) {
        if (is.call(e) && as.character(e[[1]]) %in% c("<-", "=") && is.name(e[[2]])) as.character(e[[2]]) else "<expr>"
      }, "")
      fun_ok <- length(exprs) > 0 && is.call(exprs[[length(exprs)]][[3]]) &&
        identical(as.character(exprs[[length(exprs)]][[3]][[1]]), "function")
      if (!identical(got, want) || !fun_ok) {
        res$error_class <- "structure_violation"
        res$error <- paste("top-level:", paste(got, collapse = ","))
        return(res)
      }
    }
    res$structure_ok <- TRUE
    cand <- new.env(parent = globalenv())
    ok <- tryCatch({
      for (e in exprs) eval(e, cand)
      TRUE
    }, error = function(e) e)
    if (!isTRUE(ok)) return(fail(if (grepl("time limit", conditionMessage(ok))) "timeout" else "source_error", ok))
    res$sourced_ok <- TRUE
    # Assertion helpers live in their own environment; tests see candidate bindings by copy, and
    # the global environment is locked so a candidate cannot plant replacements there.
    lib <- new.env(parent = globalenv())
    eval(parse(text = testlib_src, keep.source = FALSE), lib)
    tests <- new.env(parent = lib)
    for (n in ls(cand, all.names = TRUE)) assign(n, get(n, envir = cand), envir = tests)
    lockEnvironment(lib, bindings = TRUE)
    # No new global bindings (a candidate cannot plant replacements for test helpers there); the
    # RNG state `.Random.seed` must exist first and stays writable so tests can call set.seed().
    if (!exists(".Random.seed", envir = globalenv(), inherits = FALSE)) set.seed(0L)
    lockEnvironment(globalenv())
    texprs <- tryCatch(parse(text = test, keep.source = FALSE), error = function(e) e)
    if (inherits(texprs, "error")) return(fail("test_parse_error", texprs))
    first <- NULL
    for (e in texprs) {
      r <- tryCatch({ eval(e, tests); NULL }, error = function(err) err)
      if (!is.null(r)) {
        if (grepl("time limit", conditionMessage(r))) { first <- list("timeout", r); break }
        if (is.null(first)) first <- list("test_statement_error", r)
      }
    }
    log <- get(".exec_log", envir = lib)()
    res$checks <- paste(log$category, as.integer(log$ok), sep = "\t", collapse = "\n")
    res$n_checks <- length(log$ok)
    if (!is.null(first)) {
      res$error_class <- first[[1]]
      res$error <- substr(conditionMessage(first[[2]]), 1, 300)
    }
    res
  }

  repeat {
    h <- tryCatch(read_header(), error = function(e) NULL)
    if (is.null(h)) break
    id <- field(h, "id", numeric = FALSE)
    timeout <- field(h, "timeout")
    n_src <- field(h, "n_src")
    n_test <- field(h, "n_test")
    expect <- field(h, "expect", numeric = FALSE)
    src <- read_exact(n_src)
    test <- read_exact(n_test)
    t0 <- proc.time()[["elapsed"]]
    job <- parallel::mcparallel(child(src, test, expect, timeout), silent = TRUE)
    got <- parallel::mccollect(job, wait = FALSE, timeout = timeout + 0.25)
    status <- "ok"
    if (is.null(got)) {
      tools::pskill(job$pid, tools::SIGKILL)
      suppressWarnings(parallel::mccollect(job, wait = TRUE))
      res <- list(parse_ok = NA, structure_ok = NA, sourced_ok = NA, checks = "", n_checks = 0L,
                  error_class = "timeout", error = "killed after timeout")
      status <- "timeout"
    } else {
      res <- got[[1]]
      if (is.null(res)) {
        res <- list(parse_ok = NA, structure_ok = NA, sourced_ok = NA, checks = "", n_checks = 0L,
                    error_class = "candidate_exit", error = "child ended without a result")
        status <- "candidate_exit"
      } else if (inherits(res, "try-error")) {
        msg <- as.character(res)
        res <- list(parse_ok = NA, structure_ok = NA, sourced_ok = NA, checks = "", n_checks = 0L,
                    error_class = if (grepl("time limit", msg)) "timeout" else "child_error",
                    error = substr(msg, 1, 300))
        status <- res$error_class
      }
    }
    emit(c(list(id = id, status = status), res, list(elapsed = proc.time()[["elapsed"]] - t0)))
  }
})
