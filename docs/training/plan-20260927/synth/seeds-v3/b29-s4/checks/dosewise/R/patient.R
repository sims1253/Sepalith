p = patient("P001", "M", 60, 80, 180, scr.mgdl = 1)
stopifnot(is(p, "Patient"), p@sex == "male", p@scr.mgdl == 1)
stopifnot(identical(patient("P2", "f", 30, 60, 165)@sex, "female"))
stopifnot(inherits(try(patient("P3", "m", 130, 60, 165), silent = TRUE), "try-error"))
stopifnot(inherits(try(patient("P3", "m", 30, "60", 165), silent = TRUE), "try-error"))
imp = patient("P5", "male", 50, 176, 70, units = "imperial")
stopifnot(imp@weight.kg == 79.83, imp@height.cm == 177.8)

stopifnot(isTRUE(valid.patient(p)))
bad = p
bad@id = NA_character_
bad@sex = "x"
bad@age.y = -1
bad@weight.kg = 400
bad@height.cm = NA_real_
bad@scr.mgdl = c(1, 2)
msgs = valid.patient(bad)
stopifnot(length(msgs) == 6L, grepl("^`id`", msgs[1]), grepl("^`sex`", msgs[2]))
stopifnot(identical(msgs[3], "`age.y` must lie between 0 and 120."),
  grepl("^`weight.kg`", msgs[4]))
stopifnot(identical(msgs[5], "`height.cm` must lie between 30 and 250."),
  grepl("^`scr.mgdl`", msgs[6]))

weight(p) = 90
stopifnot(p@weight.kg == 90, inherits(try(set.patient.weight(p, 0.1), silent = TRUE), "try-error"))
stopifnot(inherits(try(set.patient.weight(p, "90"), silent = TRUE), "try-error"))
height(p) = 175
stopifnot(p@height.cm == 175,
  inherits(try(set.patient.height(p, 300), silent = TRUE), "try-error"))
stopifnot(inherits(try(set.patient.height(p, c(170, 171)), silent = TRUE), "try-error"))

q = patient("P4", "female", 40, 64, 160)
stopifnot(isTRUE(all.equal(body.surface.area(q), sqrt(64 * 160 / 3600))))
stopifnot(abs(body.surface.area(q, "dubois") - 1.67) < 0.01)
stopifnot(abs(body.surface.area(q, "haycock") - 1.70) < 0.01)
stopifnot(abs(body.surface.area(q, "gehan") - 1.70) < 0.01)

stopifnot(isTRUE(all.equal(ideal.body.weight(p), 50 + 2.3 * (175 / 2.54 - 60))))
stopifnot(isTRUE(all.equal(ideal.body.weight(q), 45.5 + 2.3 * (160 / 2.54 - 60))))
stopifnot(ideal.body.weight(patient("K", "female", 8, 25, 120)) == 45.5)
io = (175 / 2.54 - 60)
stopifnot(isTRUE(all.equal(ideal.body.weight(p, "robinson"), 52 + 1.9 * io)))
stopifnot(isTRUE(all.equal(ideal.body.weight(p, "miller"), 56.2 + 1.41 * io)))
stopifnot(isTRUE(all.equal(ideal.body.weight(p, "hamwi"), 48 + 2.7 * io)))
iq = (160 / 2.54 - 60)
stopifnot(isTRUE(all.equal(ideal.body.weight(q, "robinson"), 49 + 1.7 * iq)))
stopifnot(isTRUE(all.equal(ideal.body.weight(q, "miller"), 53.1 + 1.36 * iq)))
stopifnot(isTRUE(all.equal(ideal.body.weight(q, "hamwi"), 45.5 + 2.2 * iq)))

stopifnot(isTRUE(all.equal(creatinine.clearance(p), 80 * 90 / 72)))
stopifnot(isTRUE(all.equal(creatinine.clearance(p, "ideal"), 80 * ideal.body.weight(p) / 72)))
adj = ideal.body.weight(p) + 0.4 * (90 - ideal.body.weight(p))
stopifnot(isTRUE(all.equal(creatinine.clearance(p, "adjusted"), 80 * adj / 72)))
stopifnot(isTRUE(all.equal(creatinine.clearance(q), 0.85 * 100 * 64 / 72)))
stopifnot(isTRUE(all.equal(creatinine.clearance(q, scr.floor = 1.25), 0.85 * 100 * 64 / 90)))
stopifnot(inherits(try(creatinine.clearance(q, scr.floor = -1), silent = TRUE), "try-error"))

out = capture.output(res <- show.patient(p))
stopifnot(length(out) == 4L, grepl("P001, male, 60 y", out[1]), identical(res, p))
stopifnot(grepl("CrCl:   100 mL/min \\(normal\\)", out[4]),
  identical(capture.output(show(p)), out))
renal = function(scr) capture.output(show(patient("R", "male", 60, 90, 175, scr)))[4]
stopifnot(grepl("(mildly reduced)", renal(1.5), fixed = TRUE))
stopifnot(grepl("(moderately reduced)", renal(2.5), fixed = TRUE))
stopifnot(grepl("(severely reduced)", renal(4), fixed = TRUE))
