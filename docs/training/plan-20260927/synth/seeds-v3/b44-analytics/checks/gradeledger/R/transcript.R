fails = function(expr) inherits(try(expr, silent = TRUE), "try-error")
rec = data.frame(
  student = "s1", term = c("T1", "T1", "T2", "T2"), course = c("MATH", "HIST", "CHEM", "ART"),
  credits = c(4, 3, 4, 2), points = c(4, 0, 3, 2)
)

stopifnot(identical(validateRecords(rec), rec), fails(validateRecords(rec[-5])))
bad = rec; bad$student[1] = ""
stopifnot(fails(validateRecords(bad)))
bad = rec; bad$term[2] = NA
stopifnot(fails(validateRecords(bad)))
bad = rec; bad$course[3] = ""
stopifnot(fails(validateRecords(bad)))
bad = rec; bad$credits[4] = 0
stopifnot(fails(validateRecords(bad)))
bad = rec; bad$points[1] = 4.3
stopifnot(fails(validateRecords(bad)))

ts = termSummary(rec)
stopifnot(identical(ts$term, c("T1", "T2")), identical(ts$courses, c(2L, 2L)))
stopifnot(identical(ts$attempted, c(7, 6)), identical(ts$earned, c(4, 6)))
stopifnot(identical(ts$failed, c(1L, 0L)), identical(ts$qualityPoints, c(16, 16)))
stopifnot(identical(ts$gpa, c(2.29, 2.67)))

cg = cumulativeGpa(rec)
stopifnot(identical(cg$termGpa, c(2.29, 2.67)), identical(cg$credits, c(7, 13)))
stopifnot(identical(cg$cumGpa, c(2.29, 2.46)))
stopifnot(identical(cumulativeGpa(rec[3:4, ])$cumGpa, 2.67))

st = academicStanding(c(3.8, 3.0, 1.5), c(3.9, 1.8, 1.5), c(15, 12, 9))
stopifnot(identical(st$deansList, c(TRUE, FALSE, FALSE)))
stopifnot(identical(st$honors, c(TRUE, FALSE, FALSE)))
stopifnot(identical(st$warning, c(FALSE, TRUE, TRUE)))
stopifnot(identical(st$probation, c(FALSE, FALSE, TRUE)))
stopifnot(identical(st$underload, c(FALSE, FALSE, TRUE)), identical(st$count, c(2L, 1L, 3L)))
stopifnot(identical(st$standing, c("good", "warning", "probation")))
st2 = academicStanding(3.8, 3.9, 9)
stopifnot(!st2$deansList, st2$underload, st2$honors)

ec = convertScale(c(95, 55, 49))
stopifnot(identical(ec$grade, c("A", "E", "F")), identical(ec$passed, c(TRUE, TRUE, FALSE)))
stopifnot(identical(convertScale(c(72, 65, 39), "uk")$grade, c("First", "2:1", "Fail")))
stopifnot(identical(convertScale(c(85, 50), "german")$grade, c("1.5", "4.0")))
ib = convertScale(c(90, 44, 10, -5), "ib")
stopifnot(identical(ib$grade, c("7", "3", "1", NA)))
stopifnot(identical(ib$passed, c(TRUE, FALSE, FALSE, FALSE)))
stopifnot(fails(convertScale("A")))

tx = transcriptLines(rec)
stopifnot(length(tx) == 5L, identical(tx[1], "term     course     credits points"))
stopifnot(identical(tx[2], "T1       HIST           3.0   0.00"))
stopifnot(identical(transcriptLines(rec, "csv")[2:3], c("T1,HIST,3,0", "T1,MATH,4,4")))
stopifnot(identical(transcriptLines(rec, "tsv")[5], "T2\tCHEM\t4\t3"))
md = transcriptLines(rec, "markdown")
stopifnot(length(md) == 6L, identical(md[1], "| term | course | credits | points |"))
stopifnot(identical(md[3], "| T1 | HIST | 3 | 0 |"))
