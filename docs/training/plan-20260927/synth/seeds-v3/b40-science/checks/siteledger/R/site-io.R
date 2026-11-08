tm <- as.POSIXct("2024-05-01 12:00:00", tz = "UTC") + c(3600, 0, 0)
d <- data.frame(site = c("b", "b", "a"), time = tm, temp = c(1.5, NA, 3.25))
expected <- d[c(3, 2, 1), ]
rownames(expected) <- NULL
tmp <- tempfile("siteio")
dir.create(tmp)

p1 <- write.site.csv(d, file.path(tmp, "x.csv"))
r1 <- read.site.csv(p1)
stopifnot(identical(r1$site, expected$site), r1$time == expected$time)
stopifnot(identical(r1$temp, expected$temp))

p2 <- write.site.tsv(d, file.path(tmp, "x.tsv"))
r2 <- read.site.tsv(p2)
stopifnot(identical(r2$site, expected$site), r2$time == expected$time)
stopifnot(identical(r2$temp, expected$temp), !any(grepl(",", readLines(p2))))

p3 <- write.site.semicolon(d, file.path(tmp, "x2.csv"))
r3 <- read.site.semicolon(p3)
stopifnot(identical(r3$temp, expected$temp), any(grepl("1,5", readLines(p3), fixed = TRUE)))
stopifnot(r3$time == expected$time)

writeLines("a,b\n1,2", file.path(tmp, "bad.csv"))
stopifnot(inherits(try(read.site.csv(file.path(tmp, "bad.csv")), silent = TRUE), "try-error"))
writeLines("a\tb\n1\t2", file.path(tmp, "bad.tsv"))
stopifnot(inherits(try(read.site.tsv(file.path(tmp, "bad.tsv")), silent = TRUE), "try-error"))
writeLines("a;b\n1;2", file.path(tmp, "bad2.csv"))
stopifnot(inherits(try(read.site.semicolon(file.path(tmp, "bad2.csv")), silent = TRUE),
                   "try-error"))

ps <- write.site.split(d, file.path(tmp, "split"))
stopifnot(identical(names(ps), c("a", "b")), all(file.exists(ps)))
stopifnot(nrow(read.site.csv(ps[["b"]])) == 2L)
pt <- write.site.split(d, file.path(tmp, "split2"), "tsv")
stopifnot(grepl("\\.tsv$", pt[["a"]]), nrow(read.site.tsv(pt[["a"]])) == 1L)
pc <- write.site.split(d, file.path(tmp, "split3"), "semicolon")
stopifnot(read.site.semicolon(pc[["a"]])$temp == 3.25)
unlink(tmp, recursive = TRUE)

tmp2 <- tempfile("siteio2")
p4 <- write.site.csv(rbind(d, d[1, ]), file.path(tmp2, "dup.csv"))
stopifnot(file.exists(p4))
w <- tryCatch(read.site.csv(p4), warning = function(w) conditionMessage(w))
stopifnot(grepl("1 duplicated", w), nrow(suppressWarnings(read.site.csv(p4))) == 3L)
p5 <- write.site.tsv(rbind(d, d[1, ]), file.path(tmp2, "dup.tsv"))
stopifnot(nrow(suppressWarnings(read.site.tsv(p5))) == 3L)
p6 <- write.site.semicolon(rbind(d, d[1, ]), file.path(tmp2, "dup2.csv"))
stopifnot(nrow(suppressWarnings(read.site.semicolon(p6))) == 3L)
writeLines("site,time\na,yesterday", file.path(tmp2, "bt.csv"))
stopifnot(inherits(try(read.site.csv(file.path(tmp2, "bt.csv")), silent = TRUE), "try-error"))
stopifnot(inherits(try(write.site.tsv(d[0, ], file.path(tmp2, "e.tsv")), silent = TRUE),
                   "try-error"))
unlink(tmp2, recursive = TRUE)

tmp3 <- tempfile("siteio3")
p7 <- write.site.tsv(data.frame(site = "a", time = tm[1], v = pi), file.path(tmp3, "c.tsv"),
                     digits = 3, comment = c("line one", "line two"))
stopifnot(identical(readLines(p7)[1:2], c("# line one", "# line two")))
stopifnot(read.site.tsv(p7)$v == 3.14)
p8 <- write.site.csv(data.frame(site = "a", time = tm[1], v = pi), file.path(tmp3, "c.csv"),
                     digits = 2, comment = "x")
stopifnot(read.site.csv(p8)$v == 3.1)
p9 <- write.site.semicolon(data.frame(site = "a", time = tm[1], v = pi), file.path(tmp3, "s.csv"),
                           digits = 2)
stopifnot(read.site.semicolon(p9)$v == 3.1)
sp <- write.site.split(d, file.path(tmp3, "sp"))
stopifnot(readLines(sp[["b"]])[1] == "# site b, 2 readings")
stopifnot(inherits(try(write.site.split(d, tmp3, site.col = "zz"), silent = TRUE), "try-error"))
dna <- d
dna$site[1] <- NA
stopifnot(inherits(try(write.site.split(dna, tmp3), silent = TRUE), "try-error"))
unlink(tmp3, recursive = TRUE)
