# R environment: the 2026-10-31 drift, rig and rv

This note records why the C18b GRPO pilot stopped resuming on 2026-10-31,
the interim pacman pin the coordinator is applying, and the durable
rig-based layout for every card after C18b. It is an ops note, not a card.

## What happened

On 2026-10-31 a routine `pacman -Syu` upgraded the system R from 4.6.1 to
4.6.2. `runtime.reward.rscript` in the bound recipes is `/usr/bin/Rscript`, so
the reward probe silently started reporting a different parser identity.

Reward v2.1 refuses that mismatch by design:

- `sepalith.training.rl.campaign_r_parse_probe.RParseOnlyProbe.r_version()`
  records the identity line, e.g. `R 4.6.1 (2026-06-24)`.
- The reward buffers store the parser identity that verified each row as
  `parser_r_version` when the C14/C18 pools are built
  (`training/grpo/pool.py`).
- `RewardV21Scorer.build_references()` compares the two
  (`training/rl/reward_v21.py`, around line 416) and raises
  `RewardV21Error: row ...: reward-buffer parser 'R 4.6.1 (2026-06-24)'
  differs from the probe's 'R 4.6.2 (2026-10-29)'`.

The buffers are not corrupt. The parser that verified them is gone from
`/usr/bin`. A patch bump is enough: the identity string includes the patch
version and release date.

## Interim fix (coordinator, this box)

Downgrade to the cached `r-4.6.1` package and keep pacman from moving it
again:

```ini
# /etc/pacman.conf
IgnorePkg = r
```

This restores `/usr/bin/Rscript` = 4.6.1 so the already-bound C18b run can
resume, and it is deliberately temporary. It is a system change, needs sudo,
and is not performed by the scripts in this pull request. Leave `IgnorePkg`
in place until the last recipe bound to `/usr/bin/Rscript` has finished;
removing it is a user decision.

## Durable fix: rig owns the interpreter

[rig](https://github.com/r-lib/rig) installs R outside the distro package
manager:

```
/opt/R/4.6.1/bin/R              the interpreter
/opt/R/4.6.1/bin/Rscript        the script front end
/usr/local/bin/R-4.6.1          versioned quick links (rig, admin mode)
/usr/local/bin/Rscript-4.6.1
```

Arch Linux is not one of rig's supported Linux distributions, so rig's
admin mode falls back to the portable manylinux build under `/opt/R/4.6.1`.
That is exactly what we want here: pacman cannot see or move it.

The installer is [`scripts/env/install_r_rig.sh`](../../../scripts/env/install_r_rig.sh).
It pins:

| Tool | Version | Installs to | Needs sudo |
| --- | --- | --- | --- |
| rig | 0.11.1 | `~/.local/bin/rig` | no (tarball, checksum-verified) |
| R | 4.6.1 | `/opt/R/4.6.1` | yes (`rig add --admin`) |
| rv | 0.23.3 | `~/.local/bin/rv` | no (optional, `--with-rv`) |

Run it as your normal user; it calls `sudo` itself for the R install:

```bash
scripts/env/install_r_rig.sh --check        # read-only status
scripts/env/install_r_rig.sh                # rig + R 4.6.1
scripts/env/install_r_rig.sh --with-rv      # + rv and an rv project
```

The script adds R with `--no-default`, so `/usr/local/bin/R` and `Rscript`
are not replaced and `/usr/bin/Rscript` keeps its current meaning. Existing
recipes cannot change behaviour just because the new R was installed. The
versioned links are the only new names on `PATH`. If rig does not leave the
`/usr/local/bin` links, the script creates equivalent `R-4.6.1` /
`Rscript-4.6.1` links under `~/.local/bin`.

## The rv layer is a no-op today

`rv` manages R packages declaratively (`rproject.toml` with `r_version`,
repositories and dependencies). It discovers rig installs, including
`/opt/R/4.6.1`.

The reward and execution stacks use **no external R packages**:

- `exec_env/r_pool.R`: `library(codetools)` (recommended, ships with R)
- `exec_env/r_worker.R`: `stats`, `utils`, `methods`, `tools`, `parallel`
  (base/recommended, ship with R)
- `exec_env/r_diff.R`, `r_tokens.R`, `testlib.R`: base R only
- `training/rl/reward_parse_only.R`: `base::parse` only

So the campaign has nothing for rv to lock. `--with-rv` writes a minimal
`rproject.toml` with `r_version = "4.6.1"` and an empty `dependencies` list,
as a place to record the next CRAN dependency honestly. Do not pretend the
rv library is load-bearing; the version pin is carried by rig and by the
recipe binding.

## Recipe-binding convention for future cards

Every card bound after C18b must bind the interpreter explicitly, never via
`/usr/bin` or a bare `Rscript`:

```json
"runtime": {
  "reward": {
    "air": "/home/m0hawk/.local/bin/air",
    "rscript": "/opt/R/4.6.1/bin/Rscript",
    "rscript_version": "R 4.6.1 (2026-06-24)",
    "workers": 8,
    "exec_workers": 8,
    "exec_nice": 10
  }
}
```

Rules:

1. `runtime.reward.rscript` is the absolute rig path
   `/opt/R/4.6.1/bin/Rscript`. New cards bump the version only together
   with newly built reward buffers, never alone.
2. Record the interpreter identity in the recipe as
   `runtime.reward.rscript_version`, copied verbatim from
   `/opt/R/4.6.1/bin/Rscript --version` (or the harness `--version`). The
   bound recipe is the card's receipt; the run's own reward identity
   (`RewardV21Scorer.identity()["parser"]["r_version"]`) must match it.
3. Build the reward pools and run the training with the same interpreter.
   Pool evidence is not portable across an R patch bump.
4. For today's exec harness this needs one extra step: `ExecHarness` is
   constructed without the recipe's `rscript` and therefore resolves
   `Rscript` from `PATH` (`training/grpo_exec/trainer.py`,
   `exec_env/harness.py`). A future card must either pass the rig path into
   `ExecHarness` or launch the trainer with
   `PATH=/opt/R/4.6.1/bin:$PATH`. This pull request does not change
   `grpo_exec` or `exec_env`; record the choice in the card's recipe.

Recommended launcher guard for future cards:

```bash
RSCRIPT=/opt/R/4.6.1/bin/Rscript
test "$("$RSCRIPT" --version | head -n1)" = "Rscript (R) version 4.6.1 (2026-06-24)"
PATH=/opt/R/4.6.1/bin:$PATH <trainer command>
```

## Do not rebind bound recipes

C18 and C18b are already bound and their reward buffers were verified by the
R 4.6.1 parser. Do **not** edit their `runtime.reward.rscript`, their
`source-manifest.json`, or any other field mid-run:

| Recipe | SHA-256 |
| --- | --- |
| `docs/training/plan-20260927/c18b/bound/grpo-exec-pilot-1-arm-a-803.json` | `2cef84e46fd5dd3f997a58d36fe94feb7768c3be9bcfe29db0b5860dcc235169` |
| `docs/training/plan-20260927/c18b/bound/grpo-exec-smoke-arm-a-803.json` | `3ed82221305b79b093e4f2732647db317aca37211f8cb7e118019de5cc8fd92d` |
| `docs/training/plan-20260927/c18/bound/grpo-exec-pilot-1-arm-a-803.json` | `cca09f08d7b7b20dad091426795fb34927aaf372b020c7f453af8539ccf85d57` |

Rebinding is a new-boundary decision: it changes the parser, invalidates
the pool evidence and requires rebuilding the buffers. It is not a fix for a
run that is already in flight. The interim pacman pin keeps those recipes
valid until they finish.

## Verification without system changes

```bash
bash -n scripts/env/install_r_rig.sh
shellcheck scripts/env/install_r_rig.sh
scripts/env/install_r_rig.sh --check   # exits 1 until the install is done
```
