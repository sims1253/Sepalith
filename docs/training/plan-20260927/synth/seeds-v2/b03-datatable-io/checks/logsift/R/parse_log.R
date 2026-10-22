# parseAccessLines
acc <- c(
  '10.0.0.1 - bob [01/Mar/2026:10:00:00 +0000] "GET /index.html HTTP/1.1" 200 512 "-" "curl"',
  'garbage line',
  '10.0.0.2 - - [01/Mar/2026:10:05:00 +0000] "POST /api HTTP/1.1" 500 -'
)
a <- parseAccessLines(acc)
stopifnot(nrow(a) == 2L, attr(a, "skipped") == 1L)
stopifnot(identical(a$status, c(200L, 500L)), identical(a$bytes, c(512L, NA)))
stopifnot(identical(a$user, c("bob", "-")), !anyNA(a$time))
stopifnot(nrow(parseAccessLines(character(0))) == 0L)

# parseSyslogLines
sys <- c("Mar  4 09:15:01 web01 sshd[221]: Accepted key", "Mar 14 10:00:00 web01 cron: ok")
s <- parseSyslogLines(sys, year = "2026")
stopifnot(nrow(s) == 2L, identical(s$pid, c(221L, NA)), identical(s$program, c("sshd", "cron")))
stopifnot(format(s$time[1], "%Y-%m-%d") == "2026-03-04")

# parseAppLines, logLevels, filterByLevel
app <- c(
  "2026-03-01T10:22:01Z WARN [worker-3] queue is full",
  "2026-03-01T10:40:00Z INFO [main] started",
  "2026-03-01T11:01:00Z ERROR [worker-1] job 1234 failed",
  "2026-03-01T11:02:00Z ERROR [worker-2] job 98 failed",
  "not a log line"
)
logs <- parseAppLines(app)
stopifnot(nrow(logs) == 4L, attr(logs, "skipped") == 1L, is.factor(logs$level))
stopifnot(identical(levels(logs$level), logLevels()), length(logLevels()) == 6L)
stopifnot(nrow(filterByLevel(logs)) == 3L, nrow(filterByLevel(logs, "ERROR")) == 2L)
stopifnot(inherits(try(filterByLevel(logs, "LOUD"), silent = TRUE), "try-error"))

# countByInterval orders levels by severity within a bucket
cnt <- countByInterval(logs, "hour")
stopifnot(identical(cnt$level, c("INFO", "WARN", "ERROR")), identical(cnt$n, c(1L, 1L, 2L)))
stopifnot(inherits(cnt$bucket, "POSIXct"), nrow(countByInterval(logs, "day")) == 3L)

# summariseMessages
sm <- summariseMessages(logs)
stopifnot(identical(sm$template[1], "job <num> failed"), sm$n[1] == 2L, nrow(sm) == 2L)
stopifnot(nrow(summariseMessages(logs, "FATAL")) == 0L)

# readLogFile, including gzip
tmp <- tempfile(fileext = ".log.gz")
con <- gzfile(tmp, "w")
writeLines(app, con)
close(con)
r <- suppressMessages(readLogFile(tmp, "app"))
stopifnot(nrow(r) == 4L)
stopifnot(inherits(try(readLogFile(tempfile()), silent = TRUE), "try-error"))
unlink(tmp)
