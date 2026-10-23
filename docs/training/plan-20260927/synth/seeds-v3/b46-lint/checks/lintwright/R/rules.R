# sys.source() turns parse data off while it evaluates this file
options(keep.parse.data = TRUE)

ri <- rule_info("assign_eq")
stopifnot(identical(ri$message, "Use `<-` instead of `=`."), ri$severity == "style")
stopifnot(rule_info("true_false")$severity == "warning")
stopifnot(rule_info("semicolon")$good[2] == "y <- 2")
stopifnot(rule_info("trailing_space")$severity == "info")
sq <- rule_info("single_quotes")
stopifnot(identical(sq$message, "Use double quotes instead of single quotes."))
stopifnot(rule_info("browser_call")$severity == "error", rule_info()$rule == "assign_eq")

rules <- c("assign_eq", "true_false", "semicolon", "trailing_space", "single_quotes",
  "browser_call")
for (r in rules) {
  info <- rule_info(r)
  stopifnot(nrow(lint_rule(info$bad, r)) == 1L, nrow(lint_rule(info$good, r)) == 0L)
}
src <- c("x = 1; y <- T ", "f <- function() browser()", "s <- 'a'")
stopifnot(identical(lint_rule(src, "assign_eq")$column, 3L))
stopifnot(identical(lint_rule(src, "semicolon")$column, 6L))
stopifnot(identical(lint_rule(src, "true_false")$column, 13L))
stopifnot(identical(lint_rule(src, "trailing_space")$column, 14L))
stopifnot(identical(lint_rule(src, "single_quotes")$line, 3L))
stopifnot(identical(lint_rule(src, "browser_call")$severity, "error"))

lc <- lint_code(src)
stopifnot(nrow(lc) == 6L, identical(lc$line, c(1L, 1L, 1L, 1L, 2L, 3L)))
stopifnot(identical(lc$rule[1:2], c("assign_eq", "semicolon")))
src2 <- c("x = 1 # nolint", "y = T # nolint: assign_eq", "z = 2")
lc2 <- lint_code(src2, rules = c("assign_eq", "true_false"))
stopifnot(identical(lc2$line, c(2L, 3L)), identical(lc2$rule, c("true_false", "assign_eq")))
stopifnot(nrow(lint_code(src2, rules = "assign_eq", nolint = FALSE)) == 3L)

src3 <- c("myVar <- 1", "other_name <- function(a.b, C) a.b", "BIG_ONE = 2")
ln <- lint_names(src3, "snake_case")
stopifnot(identical(ln$name, c("myVar", "a.b", "C", "BIG_ONE")))
stopifnot(identical(ln$suggestion, c("my_var", "a_b", "c", "big_one")))
stopifnot(identical(lint_names(src3, "camelCase")$suggestion, c("otherName", "aB", "c", "bigOne")))
stopifnot(identical(lint_names(src3, "dotted.case")$suggestion,
  c("my.var", "other.name", "c", "big.one")))
stopifnot(identical(lint_names(src3, "UPPER_CASE")$suggestion,
  c("MY_VAR", "OTHER_NAME", "A_B")))
stopifnot(identical(lint_names(src3, "lowercase")$name, c("myVar", "other_name", "a.b", "C",
  "BIG_ONE")))
stopifnot(identical(lint_names(src3, "lowercase")$suggestion[1], "myvar"))

sp <- lint_spacing(c("x<-1", "y <- x+ 2"), "assign")
stopifnot(identical(sp$message, c("Missing space before `<-`.", "Missing space after `<-`.")))
stopifnot(identical(lint_spacing(c("x<-1", "y <- x+ 2"), "arithmetic")$column, 7L))
stopifnot(identical(lint_spacing("if (a==b && c >= d) 1", "comparison")$line, c(1L, 1L)))
cm <- lint_spacing("f(a ,b)", "comma")
stopifnot(identical(cm$message, c("Unexpected space before `,`.", "Missing space after `,`.")))
stopifnot(nrow(lint_spacing("x |> f() %>% g()", "pipe")) == 0L)
stopifnot(nrow(lint_spacing("x|>f()", "pipe")) == 2L)
