x <- list(owner = list(login = "ada"), labels = list(list(name = "bug"), list(name = "ui")))
stopifnot(identical(pluck.path(x, "owner.login"), "ada"))
stopifnot(identical(pluck.path(x, "labels[2].name"), "ui"))
stopifnot(is.null(pluck.path(x, "labels[3].name")), identical(pluck.path(x, "nope", 0), 0))
stopifnot(identical(pluck.path(x, "owner.login.deeper", "d"), "d"))

u <- extract.user(list(id = 7, login = "ada", name = "Ada L", email = "Ada@X.org",
  company = "@acme", followers = 12, created_at = "2020-01-02T03:04:05Z"))
stopifnot(identical(names(u), c("id", "login", "name", "email", "company", "followers",
  "created_at")))
stopifnot(identical(u$id, 7L), identical(u$email, "ada@x.org"), identical(u$company, "acme"))
stopifnot(identical(u$followers, 12L), format(u$created_at, "%H:%M") == "03:04")
e <- extract.user(list())
stopifnot(is.na(e$id), is.na(e$login), is.na(e$name), is.na(e$email), is.na(e$company))
stopifnot(is.na(e$followers), is.na(e$created_at))

r <- extract.repo(list(id = 1, full_name = "ada/eng", owner = list(login = "ada"),
  stargazers_count = 40, forks_count = 3, language = "R", license = list(spdx_id = "MIT"),
  topics = list("stats", "api"), archived = FALSE))
stopifnot(identical(r$owner, "ada"), identical(r$stars, 40L), identical(r$forks, 3L))
stopifnot(identical(r$license, "MIT"), identical(r$topics, "stats,api"))
stopifnot(identical(r$archived, FALSE))
stopifnot(identical(r$full_name, "ada/eng"), identical(r$language, "R"), identical(r$id, 1L))
er <- extract.repo(list(topics = list()))
stopifnot(all(is.na(unlist(er))), length(er) == 9L)

i <- extract.issue(list(number = 5, title = " Crash ", state = "OPEN", user = list(login = "bo"),
  labels = list(list(name = "bug"), list(name = "p1")), assignees = list(list(login = "ada")),
  comments = 2, closed_at = "2021-06-01T00:00:00Z"))
stopifnot(identical(i$number, 5L), identical(i$title, "Crash"), identical(i$state, "open"))
stopifnot(identical(i$author, "bo"), identical(i$labels, "bug,p1"), identical(i$assignees, "ada"))
stopifnot(identical(i$comments, 2L), format(i$closed_at, "%Y-%m-%d") == "2021-06-01")
ei <- extract.issue(list())
stopifnot(length(ei) == 8L, all(is.na(unlist(ei))))

items <- list(list(id = 1, user = list(login = "a"), score = 1.5, ok = TRUE,
  at = "2024-01-01T00:00:00Z", tags = list("x")), list(id = 2))
f <- extract.fields(items, c(id = "id:int", who = "user.login:chr", score = "score:num",
  ok = "ok:lgl", at = "at:time", tags = "tags:list"))
stopifnot(identical(f$id, 1:2), identical(f$who, c("a", NA)), identical(f$score, c(1.5, NA)))
stopifnot(identical(f$ok, c(TRUE, NA)), is.na(f$at[2]), format(f$at[1], "%Y") == "2024")
stopifnot(identical(f$tags[[1]], list("x")), nrow(f) == 2L)
stopifnot(inherits(try(extract.fields(items, c(id = "id:bad")), silent = TRUE), "try-error"))
stopifnot(inherits(try(extract.fields(items, c("id:int")), silent = TRUE), "try-error"))
