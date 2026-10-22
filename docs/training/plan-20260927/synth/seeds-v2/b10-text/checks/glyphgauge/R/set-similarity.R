# qgrams
stopifnot(identical(qgrams("ab"), c("#a", "ab", "b#")))
stopifnot(identical(qgrams("abc", q = 2L, pad = FALSE), c("ab", "bc")))
stopifnot(identical(qgrams("a", q = 3L, pad = FALSE), character()))

# jaccardSim / diceSim / overlapSim
stopifnot(jaccardSim("abc", "abc") == 1, all.equal(jaccardSim("ab", "ac"), 1 / 5))
stopifnot(all.equal(diceSim("ab", "ac"), 1 / 3), is.na(diceSim(NA, "a")))
stopifnot(all.equal(overlapSim("ab", "xaby"), 1 / 3), overlapSim("abc", "abcd") < 1)
stopifnot(overlapSim("bc", "abcd", q = 2L) > 0)

# qgramCosine
stopifnot(all.equal(qgramCosine("abab", "abab"), 1), qgramCosine("ab", "cd") == 0)

# jaroWinklerSim
stopifnot(abs(jaroWinklerSim("MARTHA", "MARHTA") - 0.9611111) < 1e-6)
stopifnot(jaroWinklerSim("", "") == 1, jaroWinklerSim("abc", "xyz") == 0)
