# skipped: needs dplyr (drop_lowest, course_percentages, summarise_by_student,
# summarise_by_assignment, final_grades)

# apply_late_penalty
stopifnot(all.equal(apply_late_penalty(c(90, 80, 5), c(0, 1.5, 1)), c(90, 60, 0)))
stopifnot(all.equal(apply_late_penalty(c(90, 40), c(2, NA), "cap", rate = 50), c(50, 40)))
stopifnot(all.equal(apply_late_penalty(70, 3, "none"), 70))

# letter_grade
stopifnot(identical(letter_grade(c(95, 85, 72, 60, 10, NA)), c("A", "B", "C", "D", "F", NA)))
stopifnot(identical(letter_grade(c(55, 49), scale = c(P = 50)), c("P", "F")))

# grade_points
stopifnot(identical(grade_points(c("A", "C", NA, "I")), c(4, 2, NA, NA)))
stopifnot(identical(grade_points("P", points = c(P = 1)), 1))
