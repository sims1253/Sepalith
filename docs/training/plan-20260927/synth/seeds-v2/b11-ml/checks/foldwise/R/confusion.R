cm <- conf_mat_build(c("a", "a", "b", "b"), c("a", "b", "b", "b"))

# conf_mat_build
stopifnot(inherits(cm, "fw_conf_mat"), identical(cm$levels, c("a", "b")))
stopifnot(sum(cm$table) == 4, cm$table["b", "a"] == 1)
stopifnot(inherits(try(conf_mat_build("a", c("a", "b")), silent = TRUE), "try-error"))

# class_counts
cc <- class_counts(cm)
stopifnot(identical(cc$tp, c(1L, 2L)), identical(cc$fp, c(0L, 1L)), identical(cc$fn, c(1L, 0L)))

# average_by_class
stopifnot(average_by_class(c(1, 1), c(2, 4), c(1, 1)) == 0.375)
stopifnot(average_by_class(c(1, 1), c(2, 4), c(1, 1), "micro") == 2 / 6)
stopifnot(is.na(average_by_class(0, 0, 0)))

# cm_precision / cm_recall / cm_f1 / cm_accuracy
stopifnot(all.equal(cm_precision(cm), (1 + 2 / 3) / 2), all.equal(cm_recall(cm), (0.5 + 1) / 2))
stopifnot(all.equal(cm_f1(cm, "micro"), 0.75), cm_accuracy(cm) == 0.75)
stopifnot(is.na(cm_accuracy(conf_mat_build(character(), character(), levels = "a"))))

# print and summary
txt <- capture.output(print(cm))
stopifnot(any(grepl("Accuracy: 0.750", txt, fixed = TRUE)))
sm <- summary(cm)
stopifnot(identical(names(sm), c("class", "support", "precision", "recall", "f1")))
stopifnot(identical(sm$support, c(2L, 2L)), sm$recall[2] == 1)
