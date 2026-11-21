# render.section() lives in sections.R; a fixed stub is enough here.
render.section <- function(section, format) {
  c(paste0("[", format, "] ", section$title), "body")
}
sec <- function(title, level = 1L, id = tolower(title)) {
  structure(list(title = title, level = level, id = id, blocks = list()), class = "rl_section")
}

d <- as.Date("2024-05-06")
r <- new.report("Q1 Review", "Ann Lee", date = d)
stopifnot(inherits(r, "rl_report"), identical(r$sections, list()), identical(r$date, d))
stopifnot(inherits(try(new.report(1, "a"), silent = TRUE), "try-error"))
stopifnot(inherits(try(new.report("t", NA_character_), silent = TRUE), "try-error"))
stopifnot(inherits(try(new.report("t", "a", date = "2024-01-01"), silent = TRUE), "try-error"))
stopifnot(inherits(try(new.report("t", "a", d, list(1)), silent = TRUE), "try-error"))

r <- report.add(r, sec("Sales"))
r <- report.add(r, sec("Costs"))
r <- report.add(r, sec("Detail", 2L), after = 1L)
stopifnot(identical(vapply(r$sections, `[[`, "", "title"), c("Sales", "Detail", "Costs")))
stopifnot(inherits(try(report.add(r, list()), silent = TRUE), "try-error"))

fm <- front.matter(r)
stopifnot(identical(fm, c("---", "title: \"Q1 Review\"", "author: \"Ann Lee\"",
                          "date: \"2024-05-06\"", "---")))
stopifnot(identical(front.matter(r, "html")[5], "<p class=\"byline\">Ann Lee, 2024-05-06</p>"))
stopifnot(identical(front.matter(r, "latex")[c(2, 6)], c("\\title{Q1 Review}", "\\maketitle")))
stopifnot(identical(front.matter(r, "text"),
                    c("Q1 Review", strrep("=", 9), "Ann Lee, 2024-05-06")))

stopifnot(identical(back.matter(r), "---"))
stopifnot(identical(back.matter(r, "html", "n"),
                    c("<p class=\"note\">n</p>", "</body>", "</html>")))
stopifnot(identical(back.matter(r, "latex", "n"), c("n", "\\end{document}")))
stopifnot(identical(back.matter(r, "text"), "---------"))

toc <- report.toc(r)
stopifnot(identical(toc, c("- [Sales](#sales)", "  - [Detail](#detail)", "- [Costs](#costs)")))
th <- report.toc(r, "html")
stopifnot(identical(th[c(1, 5)], c("<ul class=\"toc\">", "</ul>")))
stopifnot(identical(th[3], "  <li><a href=\"#detail\">Detail</a></li>"))
stopifnot(identical(report.toc(r, "text")[2], "  2. Detail"))

md <- render.report(r)
stopifnot(identical(md[7:9], toc), "[markdown] Costs" %in% md, identical(tail(md, 1), "---"))
lx <- render.report(r, "latex", note = "End")
stopifnot("\\tableofcontents" %in% lx, identical(tail(lx, 2), c("End", "\\end{document}")))
tx <- render.report(r, "text", toc = FALSE)
stopifnot(identical(tx[4:6], c("", "[text] Sales", "body")))
empty <- render.report(new.report("T", "A", d), "html")
stopifnot(!any(grepl("toc", empty)), identical(tail(empty, 1), "</html>"))

dir <- file.path(tempdir(), "rl-check")
paths <- write.report(r, dir, c("markdown", "text"))
stopifnot(identical(basename(unname(paths)), c("q1_review.md", "q1_review.txt")))
stopifnot(identical(readLines(paths[["markdown"]]), md), file.exists(paths[["text"]]))
stopifnot(inherits(try(write.report(r, dir, "pdf"), silent = TRUE), "try-error"))
stopifnot(identical(basename(write.report(r, dir, "html", stem = "x")), "x.html"))

r2 <- new.report("T", "A", d, subtitle = "Sub", keywords = c("k", "k", "j"))
stopifnot(identical(r2$subtitle, "Sub"), identical(r2$keywords, c("k", "j")))
stopifnot(inherits(try(new.report("T", "A", d, subtitle = 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(new.report("T", "A", d, keywords = NA_character_), silent = TRUE),
                   "try-error"))

stopifnot(identical(report.problems(new.report(" ", "", d)),
                    c("The report has an empty title.", "The report has an empty author.",
                      "The report has no sections.")))
full <- sec("Intro")
full$blocks <- list(list(type = "paragraph", content = "x"))
deep <- sec("Deep", 3L)
deep$blocks <- full$blocks
r3 <- new.report("T", "A", d, sections = list(full, deep, full, sec("Empty")))
stopifnot(identical(report.problems(r3), c("Section `intro` is used more than once.",
                                           "Section `deep` skips a heading level.",
                                           "Section `empty` has no content.")))
stopifnot(identical(report.problems(new.report("T", "A", d, sections = list(full))), character()))
