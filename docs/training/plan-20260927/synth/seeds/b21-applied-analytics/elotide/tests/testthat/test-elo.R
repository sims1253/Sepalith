toyResults <- function() {
  data.table::data.table(
    date = as.Date("2026-08-01") + c(0, 0, 7, 7, 14, 14),
    home = c("Ashby", "Corfield", "Ashby", "Brenton", "Corfield", "Brenton"),
    away = c("Brenton", "Dunmore", "Corfield", "Dunmore", "Brenton", "Ashby"),
    goalsHome = c(2L, 0L, 1L, 3L, 1L, 0L),
    goalsAway = c(1L, 0L, 1L, 0L, 2L, 2L)
  )
}

test_that("expectedScore is symmetric without home advantage", {
  p <- expectedScore(1600, 1500, homeAdvantage = 0)
  q <- expectedScore(1500, 1600, homeAdvantage = 0)
  expect_equal(p + q, 1)
  expect_gt(p, 0.5)
})

test_that("matchScore maps results to 1, 0.5 and 0", {
  expect_equal(matchScore(c(2, 1, 0), c(1, 1, 3)), c(1, 0.5, 0))
})

test_that("marginMultiplier methods agree for one-goal wins", {
  expect_equal(marginMultiplier(1, 0, "linear"), 1)
  expect_equal(marginMultiplier(1, 0, "none"), 1)
  expect_gt(marginMultiplier(4, 0, "log"), marginMultiplier(1, 0, "log"))
})

test_that("runElo conserves total rating points", {
  res <- runElo(toyResults(), initial = 1500)
  ratings <- attr(res, "ratings")
  expect_equal(sum(ratings), 1500 * 4)
  expect_equal(nrow(res), 6L)
  expect_true(all(c("eloHomePre", "eloAwayPre", "pHome", "delta") %in% names(res)))
})

test_that("runElo skips unplayed matches", {
  results <- toyResults()
  results$goalsHome[6] <- NA_integer_
  res <- runElo(results)
  expect_equal(res$delta[6], 0)
})

test_that("eloTable ranks teams by rating", {
  tab <- eloTable(runElo(toyResults()))
  expect_equal(tab$rank, 1:4)
  expect_true(all(diff(tab$rating) <= 0))
  expect_equal(tab$team[1], "Ashby")
})

test_that("leagueTable awards points and orders by goal difference", {
  tab <- leagueTable(toyResults())
  expect_equal(tab$team[1], "Ashby")
  expect_equal(tab[team == "Ashby", points], 7L)
  expect_equal(sum(tab$played), 12L)
  expect_equal(sum(tab$goalDifference), 0L)
})

test_that("eloForecast probabilities sum to one", {
  fixtures <- data.table::data.table(home = c("Ashby", "Dunmore"), away = c("Dunmore", "Ashby"))
  ratings <- c(Ashby = 1550, Dunmore = 1450)
  fc <- eloForecast(fixtures, ratings)
  expect_equal(fc$pHome + fc$pDraw + fc$pAway, c(1, 1))
  expect_gt(fc$pHome[1], fc$pHome[2])
})

test_that("roundRobin gives every pairing twice", {
  sched <- roundRobin(c("Ashby", "Brenton", "Corfield", "Dunmore", "Elmstead"))
  expect_equal(nrow(sched), 20L)
  expect_equal(max(sched$round), 10L)
  expect_false(any(duplicated(sched[, .(home, away)])))
})
