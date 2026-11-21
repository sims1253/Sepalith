stopifnot(identical(section.id("Results & Discussion!"), "results-discussion"))
stopifnot(identical(section.id("  2. Methods "), "2-methods"))
stopifnot(identical(section.id("!!"), "section"))

s <- new.section("Key Results", level = 2)
stopifnot(inherits(s, "rl_section"), identical(s$id, "key-results"), identical(s$level, 2L))
stopifnot(identical(s$blocks, list()))
stopifnot(inherits(try(new.section(1), silent = TRUE), "try-error"))
stopifnot(inherits(try(new.section("a", level = 5), silent = TRUE), "try-error"))
stopifnot(inherits(try(new.section("a", id = "Bad Id"), silent = TRUE), "try-error"))
stopifnot(inherits(try(new.section("a", blocks = "x"), silent = TRUE), "try-error"))

stopifnot(identical(section.heading("Intro", 2), "## Intro"))
stopifnot(identical(section.heading("Intro", 1, "html", "intro"), "<h1 id=\"intro\">Intro</h1>"))
stopifnot(identical(section.heading("Intro", 3, "html"), "<h3>Intro</h3>"))
stopifnot(identical(section.heading("Intro", 2, "latex", "intro"),
                    "\\subsection{Intro}\\label{intro}"))
stopifnot(identical(section.heading("Intro", 2, "text"), c("Intro", "-----")))

s <- add.paragraph(s, c("First", "sentence."))
s <- add.bullets(s, c("a", "b"))
s <- add.bullets(s, "c", ordered = TRUE)
s <- add.code(s, "x <- 1")
stopifnot(length(s$blocks) == 4L, identical(s$blocks[[1]]$content, "First sentence."))
stopifnot(identical(s$blocks[[2]]$type, "bullets"), identical(s$blocks[[3]]$type, "ordered"))
stopifnot(identical(s$blocks[[4]]$language, "r"))
stopifnot(inherits(try(add.paragraph(list(), "x"), silent = TRUE), "try-error"))

md <- render.section(s)
stopifnot(identical(md, c("## Key Results", "", "First sentence.", "", "- a", "- b", "",
                          "1. c", "", "```r", "x <- 1", "```")))
ht <- render.section(s, "html")
stopifnot(identical(ht[3], "<p>First sentence.</p>"), identical(ht[5:8],
          c("<ul>", "  <li>a</li>", "  <li>b</li>", "</ul>")), "<ol>" %in% ht)
lx <- render.section(s, "latex")
stopifnot(identical(lx[5:6], c("\\begin{itemize}", "  \\item a")), "\\begin{verbatim}" %in% lx)
stopifnot("\\begin{enumerate}" %in% lx)
tx <- render.section(s, "text")
stopifnot(identical(tx[6:7], c("* a", "* b")), identical(tail(tx, 1), "    x <- 1"))
stopifnot(identical(tx[9], "1. c"))
bad <- s
bad$blocks <- list(list(type = "image", content = "x"))
stopifnot(inherits(try(render.section(bad), silent = TRUE), "try-error"))

fl <- section.from.list(list(title = "Setup", level = 2, paragraphs = c("One.", "Two."),
                             bullets = "b", steps = c("s1", "s2"), code = "f()"))
stopifnot(identical(vapply(fl$blocks, `[[`, "", "type"),
                    c("paragraph", "paragraph", "bullets", "ordered", "code")))
stopifnot(identical(fl$level, 2L), identical(fl$blocks[[2]]$content, "Two."))
stopifnot(length(section.from.list(list(title = "Bare"))$blocks) == 0L)
stopifnot(inherits(try(section.from.list(list(level = 1)), silent = TRUE), "try-error"))
stopifnot(inherits(try(section.from.list(list(title = "x", foo = 1)), silent = TRUE), "try-error"))
