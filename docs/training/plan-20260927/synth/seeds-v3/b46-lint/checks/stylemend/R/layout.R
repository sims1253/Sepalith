# sys.source() turns parse data off while it evaluates this file
options(keep.parse.data = TRUE)

src <- c("x<-1", "y =x+2*3", "if(x==y&&y!= 1) z <- c(1 ,2,3)", "d %>%head|>tail()")
sa <- fix.spacing(src, "assign")
stopifnot(identical(sa$after[1:2], c("x <- 1", "y = x+2*3")), identical(sa$changed, 1:2))
stopifnot(identical(fix.spacing(src, "comparison")$after[3], "if(x == y&&y != 1) z <- c(1 ,2,3)"))
stopifnot(identical(fix.spacing(src, "arithmetic")$after[2], "y =x + 2 * 3"))
sc <- fix.spacing(src, "comma")
stopifnot(identical(sc$after[3], "if(x==y&&y!= 1) z <- c(1, 2, 3)"), identical(sc$changed, 3L))
stopifnot(identical(fix.spacing(src, "pipe")$after[4], "d %>% head |> tail()"))
stopifnot(identical(fix.spacing(c("x <-", "  1"), "assign")$changed, integer()))
stopifnot(identical(fix.spacing(src[1:2], "pipe")$after, src[1:2]))

br <- c("f( a, b )", "x[ 1 ]; y[[ 2 ]]", "g <- function() {x}", "h <- function() {", "  1}",
  "k( )")
bp <- fix.brackets(br, "paren")
stopifnot(identical(bp$after[c(1, 6)], c("f(a, b)", "k()")), identical(bp$changed, c(1L, 6L)))
bs <- fix.brackets(br, "square")
stopifnot(identical(bs$after[2], "x[1]; y[[2]]"), identical(bs$changed, 2L))
bb <- fix.brackets(br, "brace")
stopifnot(identical(bb$after[c(3, 5)], c("g <- function() { x }", "  1 }")))
stopifnot(identical(bb$after[4], br[4]))
stopifnot(identical(fix.brackets("f <- function() { }", "brace")$after, "f <- function() {}"))
stopifnot(identical(fix.brackets(c("f( # note", "  a )"), "paren")$after, c("f( # note", "  a)")))

ind <- c("f <- function(x) {", "if (x) {", "      y <- c(1,", "2)", "} else {", "s <- 'a",
  "  b'", "}", "", "    }")
fi <- fix.indent(ind)
stopifnot(identical(fi$after, c("f <- function(x) {", "  if (x) {", "    y <- c(1,", "      2)",
  "  } else {", "    s <- 'a", "  b'", "  }", "", "}")))
stopifnot(identical(fi$changed, c(2L, 3L, 4L, 5L, 6L, 8L, 10L)))
stopifnot(identical(fix.indent(ind, 4L)$after[4], "            2)"))
stopifnot(identical(fix.indent(c("x[[", "1]]"))$after, c("x[[", "    1]]")))

kw <- c("if(a) b else{c}", "for  (i in 1) while(TRUE) break", "f <- function (x) x",
  "if (a) 1 else if(b) 2")
fk <- fix.keywords(kw)
stopifnot(identical(fk$after, c("if (a) b else {c}", "for (i in 1) while (TRUE) break",
  "f <- function(x) x", "if (a) 1 else if (b) 2")), identical(fk$changed, 1:4))
stopifnot(identical(fix.keywords(c("if (a) {", "} else {", "}"))$changed, integer()))

pp <- c("d %>% head %>% filter(x)", "d |> tail()")
fn <- fix.pipes(pp)
stopifnot(identical(fn$after, c("d |> head() |> filter(x)", "d |> tail()")))
stopifnot(identical(fn$changed, 1L))
fm <- fix.pipes(pp, "magrittr")
stopifnot(identical(fm$after, c("d %>% head %>% filter(x)", "d %>% tail()")))
stopifnot(identical(fm$changed, 2L), identical(fix.pipes(fn$after, "magrittr")$after[1],
  "d %>% head() %>% filter(x)"))
