res <- data.frame(
  home = c("A", "C", "B", "A"), away = c("B", "A", "C", "C"),
  home_goals = c(2, 0, 3, 1), away_goals = c(1, 0, 1, 2)
)
fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")

stopifnot(identical(check_results(res), res))
stopifnot(fails(check_results(list(a = 1))), fails(check_results(res[1:3])))
bad <- res; bad$home[1] <- NA
stopifnot(fails(check_results(bad)))
bad <- res; bad$away <- factor(bad$away)
stopifnot(fails(check_results(bad)))
bad <- res; bad$home_goals[2] <- -1
stopifnot(fails(check_results(bad)))
bad <- res; bad$away_goals[2] <- 1.5
stopifnot(fails(check_results(bad)))
bad <- res; bad$away[1] <- "A"
stopifnot(fails(check_results(bad)))

tab <- standings_table(res)
stopifnot(identical(tab$team, c("A", "C", "B")))
stopifnot(identical(tab$played, c(3L, 3L, 2L)), identical(tab$won, c(1L, 1L, 1L)))
stopifnot(identical(tab$drawn, c(1L, 1L, 0L)), identical(tab$lost, c(1L, 1L, 1L)))
stopifnot(identical(tab$goals_for, c(3L, 3L, 4L)), identical(tab$goals_against, c(3L, 4L, 3L)))
stopifnot(identical(tab$goal_diff, c(0L, -1L, 1L)), all(tab$points == c(4, 4, 3)))
tab2 <- standings_table(res, points = c(win = 2, draw = 1, loss = 0))
stopifnot(all(tab2$points == c(3, 3, 2)))

idle <- rbind(tab, data.frame(team = "D", played = 0L, won = 0L, drawn = 0L, lost = 0L,
  goals_for = 0L, goals_against = 0L, goal_diff = 0L, points = 0))
pg <- standings_per_game(idle, digits = 3)
stopifnot(pg$won_pg[1] == 0.333, pg$drawn_pg[3] == 0, pg$lost_pg[3] == 0.5)
stopifnot(pg$goals_for_pg[3] == 2, pg$goals_against_pg[2] == 1.333, pg$points_pg[3] == 1.5)
stopifnot(is.na(pg$won_pg[4]), is.na(pg$points_pg[4]))

ld <- column_leaders(tab)
stopifnot(identical(ld$stat, c("points", "won", "goals_for", "lost", "goals_against")))
stopifnot(identical(ld$team, c("A/C", "A/C/B", "B", "A/C/B", "A/B")))
stopifnot(identical(ld$value, c(4, 1, 4, 1, 3)))

before <- standings_table(res[1:2, ])
cmp <- compare_standings(before, tab)
stopifnot(identical(cmp$team, c("A", "C", "B")), identical(cmp$position_change, c(0L, 0L, 0L)))
stopifnot(cmp$points_change[2] == 3, cmp$points_before[3] == 0)
stopifnot(cmp$goal_diff_change[3] == 2, cmp$won_change[2] == 1, cmp$lost_change[1] == 1)
stopifnot(cmp$won_before[1] == 1, cmp$lost_before[3] == 1, cmp$goal_diff_before[2] == 0)
stopifnot(fails(compare_standings(before[1:2, ], tab)))

pl <- format_standings(tab)
stopifnot(length(pl) == 4L, startsWith(pl[1], "Pos  Team"), grepl("^  1  A ", pl[2]))
cp <- format_standings(tab, "compact")
stopifnot(identical(cp[3], "2 C 3 1 1 1 -1 4"))
md <- format_standings(tab, "markdown")
stopifnot(length(md) == 5L, identical(md[2], "|---|---|---|---|---|---|---|---|"))
stopifnot(identical(md[5], "| 3 | B | 2 | 1 | 0 | 1 | 1 | 3 |"))
cs <- format_standings(tab, "csv")
stopifnot(identical(cs[1], "pos,team,played,won,drawn,lost,goal_diff,points"))
stopifnot(identical(cs[2], "1,A,3,1,1,1,0,4"))
