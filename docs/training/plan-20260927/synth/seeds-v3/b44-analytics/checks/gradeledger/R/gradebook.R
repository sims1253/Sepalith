fails = function(expr) inherits(try(expr, silent = TRUE), "try-error")
book = data.frame(
  student = c("s1", "s2", "s3"), homework = c(45, 50, NA), quizzes = c(18, 20, 10),
  midterm = c(80, 100, 60), final = c(150, 200, 100), project = c(30, 25, 40)
)
maxPoints = c(homework = 50, quizzes = 20, midterm = 100, final = 200, project = 40)

stopifnot(identical(validateGradebook(book, maxPoints), book))
stopifnot(fails(validateGradebook(list(), maxPoints)))
bad = book; bad$student[2] = "s1"
stopifnot(fails(validateGradebook(bad, maxPoints)))
bad = book; bad$homework[1] = 51
stopifnot(fails(validateGradebook(bad, maxPoints)))
bad = book; bad$quizzes = as.character(bad$quizzes)
stopifnot(fails(validateGradebook(bad, maxPoints)))
bad = book; bad$midterm[1] = -1
stopifnot(fails(validateGradebook(bad, maxPoints)))
bad = book; bad$final[3] = 201
stopifnot(fails(validateGradebook(bad, maxPoints)))
bad = book; bad$project[3] = 41
stopifnot(fails(validateGradebook(bad, maxPoints)))

pc = componentPercent(book, maxPoints)
stopifnot(identical(pc$homework, c(90, 100, NA)), identical(pc$quizzes, c(90, 100, 50)))
stopifnot(identical(pc$midterm, c(80, 100, 60)), identical(pc$final, c(75, 100, 50)))
stopifnot(identical(pc$project, c(75, 62.5, 100)), !any(pc$overMax))
over = book; over$final[1] = 210
stopifnot(identical(componentPercent(over, maxPoints)$overMax, c(TRUE, FALSE, FALSE)))
stopifnot(componentPercent(over, maxPoints)$final[1] == 100)

fg = finalGrade(c(90, NA), c(90, 100), c(80, 100), c(75, 100), c(75, 100))
stopifnot(identical(fg, c(80.75, 80)))
fd = finalGrade(c(90, NA), c(90, 100), c(80, 100), c(75, 100), c(75, 100), missingAs = "drop")
stopifnot(identical(fd, c(80.75, 100)))
stopifnot(fails(finalGrade(101, 1, 1, 1, 1)), fails(finalGrade(1, -1, 1, 1, 1)))
stopifnot(fails(finalGrade(1, 1, 120, 1, 1)), fails(finalGrade(1, 1, 1, 101, 1)))
stopifnot(fails(finalGrade(1, 1, 1, 1, 101)))
stopifnot(is.na(finalGrade(NA, NA, NA, NA, NA, missingAs = "drop")))

lg = letterGrade(c(95, 85, 59, -1, NA))
stopifnot(identical(lg$letter, c("A", "B", "F", NA, NA)))
stopifnot(identical(lg$points, c(4, 3, 0, NA, NA)))
pm = letterGrade(c(98, 88, 61), "plusminus")
stopifnot(identical(pm$letter, c("A+", "B+", "D-")), identical(pm$points, c(4, 3.3, 0.7)))
pf = letterGrade(c(60, 59.9), "passfail")
stopifnot(identical(pf$letter, c("Pass", "Fail")), all(is.na(pf$points)))
hn = letterGrade(c(85, 90, 10), "honors")
stopifnot(identical(hn$letter, c("Merit", "Distinction", "Fail")))
stopifnot(identical(hn$points, c(3, 4, 0)))
stopifnot(fails(letterGrade("A")))

sc = c(60, 70, 80, NA)
stopifnot(identical(curveScores(sc)$scores, sc), curveScores(sc)$shift == 0)
fl = curveScores(sc, "flat")
stopifnot(identical(fl$scores, c(65, 75, 85, NA)), fl$shift == 5, fl$method == "flat")
stopifnot(identical(curveScores(sc, "linear")$scores, c(75, 87.5, 100, NA)))
stopifnot(identical(curveScores(sc, "sqrt")$scores, c(77.5, 83.7, 89.4, NA)))
stopifnot(identical(curveScores(sc, "zscore")$scores, c(65, 75, 85, NA)))
stopifnot(curveScores(c(90, 100), "flat", target = 99)$scores[2] == 100)

cs = componentSummary(book)
stopifnot(identical(rownames(cs), c("homework", "quizzes", "midterm", "final", "project")))
stopifnot(identical(colnames(cs), c("n", "missing", "mean", "median", "min", "max")))
stopifnot(cs["homework", "n"] == 2, cs["homework", "missing"] == 1, cs["quizzes", "mean"] == 16)
stopifnot(cs["midterm", "median"] == 80, cs["final", "min"] == 100, cs["project", "max"] == 40)
