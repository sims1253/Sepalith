# skipped: needs shiny (actionsUi, actionsServer)
stopifnot(identical(actionAllowed(c("pending", "approved", "flagged"), "approve"),
                    c(TRUE, FALSE, TRUE)))
stopifnot(identical(actionAllowed(c("pending", "approved"), "flag"), c(TRUE, TRUE)))
stopifnot(identical(actionAllowed(c("rejected", "archived"), "reassign"), c(TRUE, FALSE)))
stopifnot(identical(actionAllowed(c("approved", "pending"), "archive"), c(TRUE, FALSE)))
stopifnot(inherits(try(actionAllowed("pending", "delete"), silent = TRUE), "try-error"))

rec = data.frame(id = 1:4, status = c("pending", "flagged", "approved", "rejected"),
                 interviewer = c("ann", "ann", "ben", "ben"), flags = c(0L, 2L, 0L, 0L),
                 stringsAsFactors = FALSE)
a = applyAction(rec, 1:3, "approve")
stopifnot(identical(a$status, c("approved", "approved", "approved", "rejected")),
          identical(a$flags, c(0L, 0L, 0L, 0L)))
r = applyAction(rec, c(1, 4), "reject")
stopifnot(identical(r$status, c("rejected", "flagged", "approved", "rejected")))
f = applyAction(rec, c(2, 3), "flag")
stopifnot(identical(f$status[2:3], c("flagged", "flagged")), identical(f$flags[2:3], c(2L, 1L)))
re = applyAction(rec, c(2, 4), "reassign", interviewer = "cy")
stopifnot(identical(re$interviewer, c("ann", "cy", "ben", "cy")),
          identical(re$status[c(2, 4)], c("pending", "pending")))
stopifnot(inherits(try(applyAction(rec, 1, "reassign"), silent = TRUE), "try-error"))
ar = applyAction(rec, 1:4, "archive")
stopifnot(identical(ar$status, c("pending", "flagged", "archived", "archived")))
stopifnot(identical(applyAction(rec, 99, "approve"), rec))

e = actionLogEntry("flag", c(2, 3), "maria", time = as.POSIXct("2024-03-01 10:00", tz = "UTC"))
stopifnot(identical(names(e), c("time", "user", "action", "n", "ids")))
stopifnot(e$n == 2L, identical(e$ids, "2,3"), identical(e$user, "maria"))

log = rbind(e, actionLogEntry("approve", 1:4, "maria"), actionLogEntry("approve", 7, "olu"))
cnt = actionCounts(log)
stopifnot(identical(names(cnt), c("approve", "reject", "flag", "reassign", "archive")))
stopifnot(identical(unname(cnt), c(5L, 0L, 2L, 0L, 0L)))
stopifnot(identical(actionCounts(log, "olu")[["approve"]], 1L))
stopifnot(all(actionCounts(NULL) == 0L))

stopifnot(is.null(actionRequestError(rec, 1:2, "approve")))
stopifnot(identical(actionRequestError(rec, integer(0), "approve"), "Select at least one record."))
stopifnot(identical(actionRequestError(rec, c(1, 9), "approve"), "Unknown record(s): 9."))
stopifnot(identical(actionRequestError(rec, 1, "delete"), "Unknown action: delete."))
stopifnot(identical(actionRequestError(rec, 1, "reassign"),
                    "Choose an interviewer to reassign to."))
stopifnot(is.null(actionRequestError(rec, 1, "reassign", "cy")))
stopifnot(identical(actionRequestError(rec, 1:4, "archive"), "Cannot archive record(s): 1, 2."))
