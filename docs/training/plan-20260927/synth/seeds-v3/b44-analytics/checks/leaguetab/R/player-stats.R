fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
ev <- data.frame(
  match_id = c(1, 1, 1, 1, 1, 2, 2, 2),
  player = c("Ann", "Ann", "Bob", "Cid", "Bob", "Ann", "Dee", "Dee"),
  team = c("X", "X", "Y", "X", "Y", "X", "Y", "Y"),
  type = c("goal", "shot", "save", "assist", "yellow", "goal", "shot", "red"),
  minute = c(10, 9, 9, 10, 50, 5, 20, 70)
)

stopifnot(identical(check_events(ev), ev), fails(check_events(ev[-1])))
bad <- ev; bad$match_id[2] <- NA
stopifnot(fails(check_events(bad)))
bad <- ev; bad$player[3] <- ""
stopifnot(fails(check_events(bad)))
bad <- ev; bad$team[3] <- NA
stopifnot(fails(check_events(bad)))
bad <- ev; bad$type[1] <- "own_goal"
stopifnot(fails(check_events(bad)))
bad <- ev; bad$minute[1] <- 200
stopifnot(fails(check_events(bad)))

tot <- player_totals(ev)
stopifnot(identical(tot$player, c("Ann", "Cid", "Bob", "Dee")))
stopifnot(identical(tot$team, c("X", "X", "Y", "Y")))
stopifnot(identical(tot$n_goal, c(2L, 0L, 0L, 0L)), identical(tot$matches_goal, c(2L, 0L, 0L, 0L)))
stopifnot(identical(tot$n_assist, c(0L, 1L, 0L, 0L)), identical(tot$matches_assist[2], 1L))
stopifnot(identical(tot$n_shot, c(1L, 0L, 0L, 1L)), identical(tot$matches_shot, c(1L, 0L, 0L, 1L)))
stopifnot(identical(tot$n_save, c(0L, 0L, 1L, 0L)), identical(tot$matches_save[3], 1L))
stopifnot(identical(tot$n_yellow, c(0L, 0L, 1L, 0L)), identical(tot$matches_yellow[3], 1L))
stopifnot(identical(tot$n_red, c(0L, 0L, 0L, 1L)), identical(tot$matches_red[4], 1L))

mins <- c(Ann = 180, Bob = 90, Cid = 45, Dee = 60)
p90 <- per_ninety(tot, mins, min_minutes = 60)
stopifnot(p90$goal_p90[1] == 1, p90$shot_p90[1] == 0.5, is.na(p90$assist_p90[2]))
stopifnot(p90$save_p90[3] == 1, p90$yellow_p90[3] == 1, p90$red_p90[4] == 1.5)
stopifnot(p90$shot_p90[4] == 1.5, identical(p90$minutes, c(180, 45, 90, 60)))
p2 <- per_ninety(tot, mins[c("Ann", "Bob")])
stopifnot(p2$minutes[2] == 0, is.na(p2$goal_p90[4]), p2$goal_p90[1] == 1)

g <- stat_leaders(tot, "goals")
stopifnot(identical(g$player, "Ann"), g$value == 2, g$rank == 1L)
stopifnot(identical(stat_leaders(tot, "assists")$player, "Cid"))
s <- stat_leaders(tot, "shots", n = 1)
stopifnot(identical(s$player, "Ann"), identical(s$team, "X"))
stopifnot(identical(stat_leaders(tot, "saves")$player, "Bob"))
cd <- stat_leaders(tot, "cards")
stopifnot(identical(cd$player, c("Dee", "Bob")), identical(cd$value, c(2L, 1L)))

fw <- player_rating(tot, mins, "forward")
stopifnot(identical(fw$pts_goal, c(12, 0, 0, 0)), fw$pts_shot[1] == 0.5, fw$total[1] == 12.5)
stopifnot(fw$rating[1] == 6.25, fw$pts_red[4] == -3, fw$total[4] == -2.5)
mf <- player_rating(tot, mins, "midfielder")
stopifnot(mf$pts_assist[2] == 4, mf$rating[2] == 8)
df <- player_rating(tot, mins, "defender")
stopifnot(df$pts_yellow[3] == -1.5, df$pts_shot[4] == 0.25)
kp <- player_rating(tot, c(Bob = 90), "keeper")
stopifnot(kp$pts_save[3] == 1, kp$total[3] == 0, kp$rating[3] == 0, is.na(kp$rating[1]))
