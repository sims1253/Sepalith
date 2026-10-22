stopifnot(all.equal(wrap_coord(c(-1, 5, 11), 10), c(9, 5, 1)))
stopifnot(all.equal(reflect_coord(c(-1, 5, 11), 10), c(1, 5, 9)))
stopifnot(all.equal(clamp_coord(c(-1, 5, 11), 10), c(0, 5, 10)))

pop <- data.frame(
  id = 1:3, x = c(-1, 5, 12), y = c(3, -2, 4),
  state = c("S", "I", "R"), days_in_state = integer(3),
  stringsAsFactors = FALSE
)
attr(pop, "arena") <- c(width = 10, height = 10)
b <- apply_boundary(pop, "clamp")
stopifnot(identical(b$x, c(0, 5, 10)), identical(b$y, c(3, 0, 4)))
stopifnot(all.equal(apply_boundary(pop, "wrap")$x, c(9, 5, 2)))

set.seed(3)
inside <- apply_boundary(pop, "wrap")
m <- move_random(inside, step_sd = 1)
stopifnot(m$x[3] == inside$x[3], all(m$x >= 0 & m$x < 10))
l <- move_levy(inside, min_step = 0.5)
stopifnot(l$y[3] == inside$y[3], all(l$y >= 0 & l$y < 10))

t1 <- move_toward(inside, target = c(5, 5), pull = 1, step_sd = 0)
stopifnot(all.equal(t1$x[1:2], c(5, 5)), t1$x[3] == inside$x[3])
stopifnot(inherits(try(move_toward(inside, target = 1), silent = TRUE), "try-error"))

snap2 <- inside
snap2$x <- snap2$x + 1
msd <- mean_squared_displacement(list(inside, snap2, snap2[0, ]))
stopifnot(identical(msd$step, 0:2), all.equal(msd$msd[1:2], c(0, 1)), is.na(msd$msd[3]))
stopifnot(inherits(try(mean_squared_displacement(list()), silent = TRUE), "try-error"))
