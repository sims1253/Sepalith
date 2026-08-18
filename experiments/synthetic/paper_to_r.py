#!/usr/bin/env python3
"""Synthetic-data family pilot: statistical method paper -> R implementation,
verified by SIMULATION (coverage / type-I error / bias / FDR control are
checkable properties, so implementations get real verifiable rewards without
ground-truth code).

12 curated biostatistical methods. Per method, ONE glm-5.3 call returns JSON
{name, implementation, validator, property}; the implementation + validator are
concatenated and run under Rscript (timeout 120s). On failure, ONE retry feeds
the failure output back. Outputs examples.jsonl + stats.json to the NAS.

Usage:
  paper_to_r.py                      # full pilot (12 methods, <=24 API calls)
  paper_to_r.py --only km_greenwood  # subset (comma-separated ids)
  paper_to_r.py --dry-run            # mechanics smoke test, no API
  paper_to_r.py --corrupt-run ID --find "x <- i" --replace "x <- i + 1" \
      --corruption-desc "off-by-one"   # discrimination check on saved example
"""
import argparse, json, os, re, statistics, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENDPOINT = "https://api.z.ai/api/coding/paas/v4/chat/completions"
MODEL = "glm-5.3"
MAX_TOKENS = 3000
TEMPERATURE = 0.3
R_TIMEOUT_S = 120          # hard kill for a validator run
CALL_BUDGET = 40

OUT_DIR = Path("/mnt/h/sepalith/datasets/paper_to_r_pilot")
WORK = Path("/tmp/paper_to_r")

# --------------------------------------------------------------------------
# Method list: id, name, paper-style description (~150-250 words, canonical
# source cited by name+year). Written for the prompt; no code is given.
# --------------------------------------------------------------------------
METHODS = [
    dict(
        id="km_greenwood",
        name="Kaplan-Meier estimator with Greenwood-variance confidence intervals",
        description="""The product-limit estimator of Kaplan & Meier (1958) is the nonparametric maximum likelihood estimator of the survival function S(t) from right-censored survival data. At each distinct event time t_i (times with at least one death), let d_i be the number of events and n_i the number at risk (subjects with observed time >= t_i); then S-hat(t) = prod over event times t_i <= t of (1 - d_i/n_i), with S-hat(t) = 1 below the first event time. Its variance is estimated by the Greenwood (1926) formula: Var-hat(S-hat(t)) = S-hat(t)^2 * sum over event times t_i <= t of d_i / (n_i * (n_i - d_i)). Pointwise two-sided 95% limits are usually formed with the complementary log-log (log-minus-log) transform of Kalbfleisch & Prentice (1980): with g = sqrt(Var-hat) / (S-hat * log(S-hat)) when 0 < S-hat < 1, the limits are S-hat^exp(-1.959964 * g) and S-hat^exp(+1.959964 * g). Implement functions that take vectors of observed times (time) and event indicators (status: 1 = event, 0 = censored) and return, for a query grid of times, the KM survival estimate plus lower and upper 95% confidence limits. Handle tied event times, the at-risk definition above, and return NA for query times beyond the last observation when that observation is censored. Compute the estimator, the Greenwood variance, and the interval from scratch; do not use the survival package.""",
    ),
    dict(
        id="logrank",
        name="Two-sample log-rank test",
        description="""The log-rank test of Mantel (1966), in the formulation of Peto & Peto (1972), compares survival between two groups under the null hypothesis of equal hazard functions. Pool both groups and consider the distinct event times t_i. At t_i let d_i be the total number of events, n_i the total number at risk (observed time >= t_i), and d1_i, n1_i the corresponding counts in group 1. Conditional on the margins of the 2x2 table at t_i, under the null, d1_i has hypergeometric mean E1_i = d_i * n1_i / n_i and variance V1_i = d_i * (n_i - d_i) * n1_i * (n_i - n1_i) / (n_i^2 * (n_i - 1)), with V1_i taken as 0 when n_i = 1. The test aggregates over event times: U = sum_i (d1_i - E1_i), Var(U) = sum_i V1_i, and the statistic chi2 = U^2 / Var(U) is referred to the chi-square distribution with 1 degree of freedom (equivalently sqrt(chi2) signed by U is a standard normal z). Implement a function taking parallel vectors time, status (1 = event), and group (two-level factor or 0/1) and returning the chi-square statistic, degrees of freedom, and the two-sided p-value; base R's pchisq may be used for the reference distribution, but the statistic, variance, and risk-set bookkeeping must be computed from scratch.""",
    ),
    dict(
        id="cox_nr",
        name="Cox proportional-hazards partial likelihood fitted by Newton-Raphson",
        description="""Cox's (1972) proportional hazards model, in the notation of Kalbfleisch & Prentice (2002), with Breslow's method for tied event times. For right-censored data (time_i, delta_i) and covariate matrix X (n x p, rows x_i), the log partial likelihood is l(beta) = sum over i with delta_i = 1 of [ x_i' beta - log( sum over j in R_i of exp(x_j' beta) ) ], where the risk set R_i = { j : time_j >= time_i }. The score is U(beta) = sum over events i of [ x_i - ( sum_{j in R_i} w_j x_j / sum_{j in R_i} w_j ) ] with w_j = exp(x_j' beta), and the observed information is I(beta) = sum over events i of [ (sum_{j in R_i} w_j x_j x_j')/(sum_{j in R_i} w_j) - z_i z_i' ] where z_i = (sum_{j in R_i} w_j x_j)/(sum_{j in R_i} w_j). Estimate beta by Newton-Raphson starting from beta = 0, using step halving whenever a step decreases l(beta), and iterate until the maximum absolute score component and relative log-likelihood change are below 1e-8, with an iteration cap (e.g. 200). Return the coefficient vector, standard errors sqrt(diag(solve(I(beta-hat)))), the log partial likelihood at convergence, and the iteration count. All risk-set sums, score, and information must be your own base-R code; do not call survival::coxph.""",
    ),
    dict(
        id="gee_exchangeable",
        name="GEE for Gaussian clustered data with exchangeable working correlation",
        description="""Generalized estimating equations of Liang & Zeger (1986) for clustered Gaussian outcomes with an exchangeable working correlation. For cluster i of size n_i with outcome vector y_i and design matrix X_i (including the intercept column), the GEE solves sum_i D_i' V_i^{-1} (y_i - X_i beta) = 0, where V_i = sigma^2 R_i(rho) and R_i has 1 on the diagonal and rho off-diagonal (exchangeable). Fit by iterated feasible generalized least squares: given current (beta, rho, sigma^2), update beta = (sum_i X_i' V_i^{-1} X_i)^{-1} (sum_i X_i' V_i^{-1} y_i); then with residuals r_ij = y_ij - x_ij' beta, update rho = ( sum_i sum_{j != k} r_ij r_ik ) / ( sum_i n_i (n_i - 1) ) and sigma^2 = ( sum_i ||r_i||^2 ) / ( N - p ), N = total observations, p = number of coefficients. Iterate to convergence of beta. The model-based covariance is (sum X_i' V_i^{-1} X_i)^{-1}; the robust sandwich covariance is that matrix times sum_i X_i' V_i^{-1} r_i r_i' V_i^{-1} X_i times the same matrix again. Implement from scratch, taking a data frame with a cluster id, outcome, and covariates, and returning beta, robust standard errors, rho-hat, and sigma-hat. Use base R matrix algebra only.""",
    ),
    dict(
        id="bh_qvalues",
        name="Benjamini-Hochberg step-up FDR procedure and q-values",
        description="""The step-up procedure of Benjamini & Hochberg (1995) controls the false discovery rate when testing m hypotheses. Given p-values p_1..p_m and a level q, sort them p_(1) <= ... <= p_(m) and let k = max{ i : p_(i) <= (i/m) * q } (k = 0 if the set is empty); reject the hypotheses corresponding to p_(1), ..., p_(k). FDR is controlled at level q when all m null p-values are independent or positively dependent (PRDS). The procedure is equivalent to comparing each p-value to its BH adjusted p-value (q-value): q-hat_(i) = min over j >= i of min(1, (m / j) * p_(j)), which is monotone nondecreasing in i, so q-hat_(i) = min(1, min_{j >= i} m p_(j) / j) after enforcing monotonicity from the largest to the smallest. Implement two functions: (a) given a numeric vector of p-values, return the BH adjusted p-values (q-values); (b) given the p-values and a level q, return the logical rejection vector. Handle ties and clip at 1; assume no NAs. The step-up logic, sorting, and monotonicity enforcement must be written from scratch with base R; do not call p.adjust, and use the plain BH constant m, not the Benjamini & Yekutieli (2001) sum_{j<=m} 1/j correction.""",
    ),
    dict(
        id="boot_percentile",
        name="Nonparametric bootstrap percentile confidence interval",
        description="""Efron's (1979) bootstrap, in the percentile-interval form of Efron & Tibshirani (1993, ch. 12-13). Given data x_1, ..., x_n and a statistic theta-hat = t(x), draw B independent bootstrap resamples, each formed by sampling n indices from 1..n with replacement with probability 1/n each and taking the corresponding data values; recompute the statistic on each resample to obtain replicates theta-hat*_1, ..., theta-hat*_B. The bootstrap estimate of standard error is se-boot = sd(replicates) (with the 1/(B-1) denominator), and the two-sided 1 - alpha percentile confidence interval is the empirical alpha/2 and 1 - alpha/2 quantiles of the replicates, e.g. quantile(replicates, c(alpha/2, 1 - alpha/2), type = 7). Implement a function taking a numeric vector of data, a statistic function (defaulting to the mean), B, the confidence level, and a seed, and returning the original statistic value, se-boot, the percentile interval limits, and the B replicates. The resampling loop (index sampling with replacement via sample.int and numeric indexing) must be written from scratch and reproducible via the seed argument; base R only.""",
    ),
    dict(
        id="jackknife_bias",
        name="Jackknife bias estimation",
        description="""The jackknife of Quenouille (1949, 1956), as popularized by Tukey (1958). For a statistic theta-hat = t(x_1, ..., x_n), let theta-hat_(i) be the statistic recomputed on the sample with the i-th observation deleted, for i = 1..n, and let theta-bar_(.) = (1/n) sum_i theta-hat_(i) be their mean. The jackknife estimate of the bias of theta-hat is bias-jack = (n - 1) * (theta-bar_(.) - theta-hat), and the bias-corrected estimator is theta-jack = n * theta-hat - (n - 1) * theta-bar_(.). For statistics that are smooth functions of sample means, this removes the O(1/n) leading bias term; for statistics linear in the sample mean it recovers the finite-sample bias exactly (up to numerical error). Implement a function taking a numeric vector x and a statistic function t (with a sensible default) and returning theta-hat, the n leave-one-out values, bias-jack, and theta-jack. Compute all leave-one-out values from scratch by index deletion; base R only. A canonical check: the maximum-likelihood variance estimator s2_n = (1/n) sum (x_i - x-bar)^2 for normal data with true variance sigma^2 has exact bias -sigma^2/n, which the jackknife must recover exactly because s2_n is a function of the sample mean of x^2 and x.""",
    ),
    dict(
        id="em_mixture",
        name="EM algorithm for a two-component normal mixture",
        description="""The EM algorithm of Dempster, Laird & Rubin (1977) applied to the two-component univariate normal mixture, as treated in McLachlan & Peel (2000, ch. 2). The model is f(x) = pi * phi(x; mu1, sigma1^2) + (1 - pi) * phi(x; mu2, sigma2^2), where phi is the normal density. The E-step computes responsibilities tau_ij = ( pi_i * phi(x_j; mu_i, sigma_i^2) ) / ( sum_l pi_l * phi(x_j; mu_l, sigma_l^2) ) for components i = 1, 2, evaluated on the log scale for numerical stability (subtract the log-sum-exp of the two log component terms). The M-step updates pi_i = (1/n) sum_j tau_ij, mu_i = sum_j tau_ij x_j / sum_j tau_ij, and sigma_i^2 = sum_j tau_ij (x_j - mu_i)^2 / sum_j tau_ij. Iterate from starting values (argument with reasonable defaults, e.g. quantile-based) until the absolute change in the observed-data log-likelihood L = sum_j log( sum_i pi_i phi(x_j; mu_i, sigma_i^2) ) is below 1e-8 * (1 + |L|) or a maximum of 500 iterations. Implement from scratch, returning pi1, mu1, sigma1, mu2, sigma2, the final log-likelihood, the iteration count, and the final responsibility matrix; guard sigma_i^2 with a small floor (e.g. 1e-6) to avoid degenerate components. Base R only.""",
    ),
    dict(
        id="lasso_cd",
        name="Coordinate-descent lasso for Gaussian regression",
        description="""The cyclical coordinate-descent lasso of Friedman, Hastie & Tibshirani (2010), "Regularization Paths for Generalized Linear Models via Coordinate Descent" (Journal of Statistical Software), the algorithm behind glmnet. The objective is (1/(2n)) * sum_i (y_i - x_i' beta)^2 + lambda * sum_j |beta_j|. Standardize each predictor to zero mean and unit variance using the denominator-n standard deviation, center the response, and fit without an intercept; afterwards transform coefficients back to the original scale and recover the intercept as y-bar - sum_j beta_j * x-bar_j. The update for coordinate j, holding all other coefficients fixed: with partial residual r_i^(j) = y_i - sum_{k != j} x_ik beta_k (computable incrementally from the full residual), the unregularized solution is z_j = (1/n) sum_i x_ij r_i^(j), and the lasso update is beta_j <- S(z_j, lambda) where S(z, gamma) = sign(z) * max(|z| - gamma, 0) is the soft-thresholding operator. Iterate full cycles until max_j |change in beta_j| < 1e-7 or 10000 iterations. Implement a function taking X (n x p matrix), y, and lambda, returning the coefficients on the original scale, the intercept, and the number of iterations, all written from scratch; do not call glmnet or lars.""",
    ),
    dict(
        id="bland_altman",
        name="Bland-Altman limits of agreement",
        description="""The agreement-analysis method of Bland & Altman (1986), "Statistical methods for assessing agreement between two methods of clinical measurement" (The Lancet). Two measurement methods are applied to each of n subjects, giving pairs (m1_i, m2_i). The analysis works on the differences d_i = m1_i - m2_i: the estimated bias of method 1 relative to method 2 is the mean difference d-bar = (1/n) sum d_i, and the agreement is summarized by the "limits of agreement" d-bar +/- 1.96 * s_d, where s_d = sqrt( (1/(n-1)) sum (d_i - d-bar)^2 ) is the sample standard deviation of the differences. Under the assumption that differences are independent draws from an approximately normal distribution with constant mean and variance, these limits are predicted to contain 95% of the population of differences between the methods; 1.96 should be computed as qnorm(0.975) rather than hard-coded. Implement a function taking two paired numeric vectors and a confidence level, returning the mean difference, s_d, the lower and upper limits of agreement, and per-observation standardized differences (d_i - d-bar)/s_d. Handle recycling-free equal-length input; base R, written from scratch.""",
    ),
    dict(
        id="newcombe_ci",
        name="Newcombe hybrid score confidence interval for a difference of proportions",
        description="""Method 10 of Newcombe (1998), "Interval estimation for the difference between independent proportions: comparison of eleven methods" (Statistics in Medicine 17:873-890): the "square-and-add" hybrid score interval combining two Wilson (1927) intervals. For arm a with x_a successes in n_a trials, p-hat_a = x_a/n_a and z = qnorm(1 - alpha/2), the Wilson score interval is [ l_a, u_a ] with l_a = ( 2*n_a*p-hat_a + z^2 - z*sqrt(z^2 + 4*n_a*p-hat_a*(1 - p-hat_a)) ) / (2*(n_a + z^2)) and u_a = ( 2*n_a*p-hat_a + z^2 + z*sqrt(z^2 + 4*n_a*p-hat_a*(1 - p-hat_a)) ) / (2*(n_a + z^2)). The hybrid score interval for the difference p1 - p2 is then: lower = (p-hat_1 - p-hat_2) - sqrt( (p-hat_1 - l_1)^2 + (u_2 - p-hat_2)^2 ) and upper = (p-hat_1 - p-hat_2) + sqrt( (u_1 - p-hat_1)^2 + (p-hat_2 - l_2)^2 ). Implement a function taking x1, n1, x2, n2 and a confidence level, returning the difference estimate and the two limits, computed from scratch with base R (qnorm allowed for z). The interval has close-to-nominal two-sided coverage even for small samples and extreme probabilities, which is the property Newcombe demonstrated.""",
    ),
    dict(
        id="delong_auc",
        name="DeLong variance estimator for the ROC AUC",
        description="""The nonparametric variance estimator of DeLong, DeLong & Clarke-Pearson (1988), "Comparing the areas under two or more correlated receiver operating characteristic curves" (Biometrics 44:837-845), for the area under the empirical ROC curve. With continuous scores for n1 cases (x_i) and n0 controls (y_j), the AUC is the two-sample Mann-Whitney statistic A-hat = (1/(n1*n0)) * sum_i sum_j psi(x_i, y_j), where psi(a, b) = 1 if a > b, 0.5 if a = b (tie), and 0 if a < b. The variance uses the structural components: for each case i, V1_i = (1/n0) * sum_j psi(x_i, y_j), and for each control j, V0_j = (1/n1) * sum_i psi(x_i, y_j). Then, with S1^2 = (1/(n1-1)) * sum_i (V1_i - V1-bar)^2 and S0^2 = (1/(n0-1)) * sum_j (V0_j - V0-bar)^2, the DeLong variance estimate is Var-hat(A-hat) = S1^2/n1 + S0^2/n0. Implement a function taking a vector of case scores and a vector of control scores and returning A-hat and its standard error sqrt(Var-hat(A-hat)), with the Mann-Whitney kernel and structural components computed from scratch; base R only (no pROC, no auc functions).""",
    ),
]

PROMPT_TEMPLATE = """You are implementing a statistical method from its primary-source description, then verifying your implementation by simulation in R.

METHOD: {{NAME}}

PAPER-STYLE DESCRIPTION:
{{DESCRIPTION}}

Return ONE JSON object with exactly these keys:

"name": short name of the method.

"implementation": R source code implementing the method from scratch. The core estimator/test must be your own code (no call to an existing package function that already performs the method, e.g. no survival::survfit, survival::coxph, p.adjust, glmnet). Recommended-CRAN packages are allowed only for peripheral helpers such as simulation (mvtnorm is installed if you need multivariate normals). Define named functions; include NO top-level executable statements (no demo code, no printing, nothing that runs when the file is sourced except function definitions).

"validator": a self-contained base-R script that verifies a statistical property of your implementation by simulation. It will be executed in the SAME Rscript process directly AFTER the implementation code (appended into one file), so it can simply call your functions. It must:
- begin with set.seed() with a fixed seed of your choosing;
- simulate data with known ground truth (known parameter values, true null, known population quantity);
- apply your implementation;
- check a checkable statistical property (bias, coverage, type-I error, FDR control, variance-estimate accuracy, etc.) with an explicit numeric tolerance that accommodates Monte Carlo noise: use at most 2000 total replications, keep per-replication sample sizes modest, and pick tolerance bands a correct implementation passes reliably but a grossly wrong implementation fails;
- run in under 30 seconds on one laptop CPU core;
- end by printing exactly one verdict line: PASS alone on its line, or FAIL: <reason>.

"property": one sentence stating the property checked and its nominal value.

Be economical with comments and keep the code compact so everything fits. Return ONLY the JSON object."""

RETRY_TEMPLATE = """You are implementing a statistical method from its primary-source description, then verifying your implementation by simulation in R.

METHOD: {{NAME}}

PAPER-STYLE DESCRIPTION:
{{DESCRIPTION}}

Your previous attempt FAILED verification:

- Failure category: {{CATEGORY}}
- Failure detail / R output (truncated):
{{OUTPUT}}

Requirements (same as before): "implementation" = from-scratch R functions, no top-level executable code; "validator" = self-contained base-R simulation script executed after the implementation in one Rscript process, beginning with a fixed set.seed(), at most 2000 replications, under 30 seconds, ending by printing exactly PASS or FAIL: <reason>; "property" = one sentence.

Fix the problem: if the implementation is wrong, correct the math; if the validator had a runtime error, fix the script; if the tolerance was unreasonably tight or loose for the Monte Carlo noise, fix the tolerance; if the output was truncated, be more concise. Return ONLY the corrected JSON object with the same four keys."""

# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
CALL_COUNT = {"n": 0}


def api_call(prompt, max_retries=3):
    key = os.environ.get("ZAI_API_KEY")
    if not key:
        raise RuntimeError("ZAI_API_KEY not set (source ~/.zshrc)")
    if CALL_COUNT["n"] >= CALL_BUDGET:
        raise RuntimeError(f"call budget {CALL_BUDGET} exhausted")
    CALL_COUNT["n"] += 1
    body = json.dumps({
        "model": MODEL,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }).encode()
    last = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(ENDPOINT, data=body, headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
            msg = data["choices"][0]["message"]
            return dict(content=msg.get("content") or "",
                        finish=data["choices"][0].get("finish_reason", ""),
                        usage=data.get("usage", {}),
                        latency=round(time.time() - t0, 1),
                        transport_retries=attempt)
        except Exception as e:  # transport-level retry
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"transport failure after {max_retries} attempts: {last}")


def fake_call(prompt):
    """Smoke-test responder: known-good KM-lite + validator."""
    impl = """km_mean <- function(time, status) {
  ot <- sort(unique(time[status == 1])); n <- length(time)
  S <- 1; t_out <- c(0); s_out <- c(1)
  for (t in ot) { d <- sum(time == t & status == 1); nr <- sum(time >= t)
    S <- S * (1 - d / nr); t_out <- c(t_out, t); s_out <- c(s_out, S) }
  list(t = t_out, s = s_out)
}"""
    val = """set.seed(1)
t1 <- rexp(200, 1); ct <- rexp(200, 0.7); obs <- pmin(t1, ct); st <- as.integer(t1 <= ct)
f <- km_mean(obs, st); est <- approx(f$t, f$s, xout = log(2))$y
if (abs(est - 0.5) < 0.08) cat("PASS\\n") else cat("FAIL: bias too large\\n")"""
    time.sleep(0.05)
    return dict(content=json.dumps({"name": "KM-lite", "implementation": impl,
                                    "validator": val,
                                    "property": "KM S(log 2) near true median 0.5"}),
                finish="stop", usage={}, latency=0.05, transport_retries=0)


def parse_model_json(raw):
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
        txt = re.sub(r"\n?```\s*$", "", txt)
    try:
        return json.loads(txt)
    except Exception:
        pass
    a, b = txt.find("{"), txt.rfind("}")
    if a >= 0 and b > a:
        return json.loads(txt[a:b + 1])
    raise ValueError("no JSON object found in model output")


# --------------------------------------------------------------------------
# R execution
# --------------------------------------------------------------------------
def classify_failure(stdout, stderr, returncode):
    out = (stdout or "") + "\n" + (stderr or "")
    # A printed FAIL verdict is the meaningful outcome even if the script then
    # dies (e.g. top-level `else` parse error after the verdict line) -> it
    # wins over compile classification.
    if re.search(r"^FAIL\b", out, re.M):
        return "property-violation"
    if re.search(r"unexpected (symbol|numeric|constant|string|input|end|')", out) \
            or re.search(r"(parse|syntax) error", out, re.I):
        return "compile"
    return "simulation-error"


def run_validator(impl_code, validator_code, tag):
    """Concatenate implementation + validator into one Rscript run."""
    WORK.mkdir(parents=True, exist_ok=True)
    impl_path = WORK / f"{tag}_impl.R"
    runner = WORK / f"{tag}_run.R"
    impl_path.write_text(impl_code + "\n")
    runner.write_text(
        impl_code + "\n\n# ---- validator (appended after implementation) ----\n"
        + validator_code + "\n")
    t0 = time.time()
    try:
        proc = subprocess.run(["Rscript", str(runner)], capture_output=True,
                              text=True, timeout=R_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return dict(ok=False, category="simulation-error",
                    reason=f"validator exceeded {R_TIMEOUT_S}s timeout",
                    verdict=None, runtime_s=float(R_TIMEOUT_S),
                    output=f"[timeout after {R_TIMEOUT_S}s]")
    runtime = round(time.time() - t0, 2)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if re.search(r"^PASS\s*$", proc.stdout or "", re.M):
        return dict(ok=True, category=None, reason=None, verdict="PASS",
                    runtime_s=runtime, output=out)
    if proc.returncode != 0 or re.search(r"^FAIL\b", out, re.M):
        cat = classify_failure(proc.stdout, proc.stderr, proc.returncode)
        vm = re.search(r"^FAIL\b.*$", out, re.M)
        return dict(ok=False, category=cat,
                    reason=(vm.group(0).strip() if vm
                            else (out.strip().splitlines() or ["R error"])[-1]),
                    verdict=("FAIL" if vm else None), runtime_s=runtime,
                    output=out)
    return dict(ok=False, category="simulation-error",
                reason="validator finished but printed no PASS/FAIL verdict",
                verdict=None, runtime_s=runtime, output=out)


# --------------------------------------------------------------------------
# per-method pipeline
# --------------------------------------------------------------------------
def process_method(m, dry):
    rec = dict(method=m["id"], name=m["name"], description=m["description"],
               implementation="", validator="", property="",
               passed=False, first_try_passed=False, retries=0, model=MODEL,
               full_prompt=None,
               validator_runtime_s=None, failure_category=None,
               failure_reason=None, api_calls=0,
               latency_s=[], usage_total_tokens=[])
    last_fail = None
    for attempt in (1, 2):
        if attempt == 1:
            prompt = (PROMPT_TEMPLATE.replace("{{NAME}}", m["name"])
                      .replace("{{DESCRIPTION}}", m["description"]))
        else:
            prompt = (RETRY_TEMPLATE.replace("{{NAME}}", m["name"])
                      .replace("{{DESCRIPTION}}", m["description"])
                      .replace("{{CATEGORY}}", last_fail["category"])
                      .replace("{{OUTPUT}}", last_fail["output"][-1800:]))
        try:
            resp = fake_call(prompt) if dry else api_call(prompt)
        except RuntimeError as e:
            rec["failure_category"] = rec["failure_category"] or "budget"
            rec["failure_reason"] = str(e)
            rec["api_calls"] = CALL_COUNT["n"]
            return rec
        rec["api_calls"] = CALL_COUNT["n"]
        rec["latency_s"].append(resp["latency"])
        rec["usage_total_tokens"].append(resp["usage"].get("total_tokens", 0))
        try:
            obj = parse_model_json(resp["content"])
            missing = [k for k in ("implementation", "validator", "property")
                       if not obj.get(k)]
            if missing:
                raise ValueError(f"missing/empty keys: {missing}; "
                                 f"finish_reason={resp['finish']}")
        except ValueError as e:
            last_fail = dict(category="invalid-output", output=str(e),
                             reason=str(e))
            rec["failure_category"] = last_fail["category"]
            rec["failure_reason"] = str(e)[:300]
            rec["retries"] = attempt - 1
            continue
        rec["full_prompt"] = prompt  # prompt that produced this code
        rec["implementation"] = obj["implementation"]
        rec["validator"] = obj["validator"]
        rec["property"] = obj.get("property", "")
        if not rec["name"]:
            rec["name"] = obj.get("name", m["name"])
        res = run_validator(obj["implementation"], obj["validator"],
                            f"{m['id']}_try{attempt}")
        rec["validator_runtime_s"] = res["runtime_s"]
        if res["ok"]:
            rec["passed"] = True
            rec["first_try_passed"] = (attempt == 1)
            rec["retries"] = attempt - 1
            rec["failure_category"] = None
            rec["failure_reason"] = None
            break
        last_fail = res
        rec["failure_category"] = res["category"]
        rec["failure_reason"] = (res["reason"] or "")[:300]
        rec["retries"] = attempt - 1
    return rec


def write_outputs(records, journal=False):
    import datetime
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    generator = str(Path(__file__).resolve())
    slim = []
    for r in records:
        slim.append(dict(
            method=r["method"], description=r["description"],
            implementation=r["implementation"], validator=r["validator"],
            property=r["property"], passed=r["passed"], retries=r["retries"],
            model=r["model"],
            full_prompt=r.get("full_prompt"),
            generator=generator, generated_at=generated_at,
            license=None, source_url=None,  # pure synthetic, no external source
            first_try_passed=r["first_try_passed"],
            validator_runtime_s=r["validator_runtime_s"],
            failure_category=r["failure_category"],
            failure_reason=r["failure_reason"]))
    if not journal:
        (OUT_DIR / "examples.jsonl").write_text(
            "\n".join(json.dumps(s) for s in slim) + "\n")

    runtimes = [r["validator_runtime_s"] for r in records
                if r["validator_runtime_s"] is not None]
    pass_runtimes = [r["validator_runtime_s"] for r in records if r["passed"]]
    fails = [r for r in records if not r["passed"]]
    cats = {}
    for r in fails:
        cats[r["failure_category"]] = cats.get(r["failure_category"], 0) + 1
    n = len(records)
    stats = dict(
        model=MODEL,
        generator=generator,
        generated_at=generated_at,
        license=None, source_url=None,  # pure synthetic
        n_methods=n,
        api_calls_used=CALL_COUNT["n"],
        pass_first_try=sum(1 for r in records if r["first_try_passed"]),
        pass_rate_first_try=round(sum(1 for r in records if r["first_try_passed"]) / n, 3),
        pass_final=sum(1 for r in records if r["passed"]),
        pass_rate_after_retry=round(sum(1 for r in records if r["passed"]) / n, 3),
        mean_validator_runtime_s=(round(statistics.mean(runtimes), 2)
                                  if runtimes else None),
        mean_validator_runtime_passing_s=(round(statistics.mean(pass_runtimes), 2)
                                          if pass_runtimes else None),
        max_validator_runtime_s=(round(max(runtimes), 2) if runtimes else None),
        failures_by_category=cats,
        failed_methods=[dict(method=r["method"], category=r["failure_category"],
                             reason=r["failure_reason"]) for r in fails],
        per_method=[dict(method=r["method"], passed=r["passed"],
                         first_try=r["first_try_passed"], retries=r["retries"],
                         runtime_s=r["validator_runtime_s"])
                    for r in records])
    if not journal:
        (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=1))
    else:
        (OUT_DIR / "journal.jsonl").write_text(
            "\n".join(json.dumps(s) for s in slim) + "\n")
    return stats


# --------------------------------------------------------------------------
# corruption discrimination check
# --------------------------------------------------------------------------
def corrupt_run(method_id, find, replace, desc):
    recs = [json.loads(l) for l in
            (OUT_DIR / "examples.jsonl").read_text().splitlines() if l.strip()]
    rec = next((r for r in recs if r["method"] == method_id), None)
    if rec is None:
        sys.exit(f"method {method_id} not found in examples.jsonl")
    if find not in rec["implementation"]:
        sys.exit("snippet not found in implementation:\n" + find)
    corrupted = rec["implementation"].replace(find, replace, 1)
    res = run_validator(corrupted, rec["validator"], f"{method_id}_corrupt")
    entry = dict(method=method_id, corruption=desc,
                 original_snippet=find, corrupted_snippet=replace,
                 passed_before=rec["passed"],
                 passed_after_corruption=res["ok"],
                 discriminating=rec["passed"] and not res["ok"],
                 failure_category=res["category"],
                 failure_reason=(res["reason"] or "")[:300],
                 runtime_s=res["runtime_s"])
    path = OUT_DIR / "corruption_check.json"
    existing = json.loads(path.read_text()) if path.exists() else []
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=1))
    print(json.dumps(entry, indent=1))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated method ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--corrupt-run", default=None, metavar="METHOD_ID")
    ap.add_argument("--find", default=None)
    ap.add_argument("--replace", default=None)
    ap.add_argument("--corruption-desc", default="intentional corruption")
    args = ap.parse_args()

    if args.corrupt_run:
        corrupt_run(args.corrupt_run, args.find, args.replace,
                    args.corruption_desc)
        return

    selected = METHODS
    if args.only:
        ids = [s.strip() for s in args.only.split(",")]
        selected = [m for m in METHODS if m["id"] in ids]
        missing = set(ids) - {m["id"] for m in selected}
        if missing:
            sys.exit(f"unknown method ids: {missing}")
    if args.dry_run:
        # never let a mechanics smoke test clobber the real dataset dir
        global OUT_DIR
        OUT_DIR = Path("/tmp/paper_to_r_dryrun")

    print(f"{len(selected)} methods, budget {CALL_BUDGET} calls"
          f"{' [DRY-RUN]' if args.dry_run else ''}", flush=True)
    t0 = time.time()
    records = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for rec in ex.map(lambda m: process_method(m, args.dry_run), selected):
            records.append(rec)
            status = "PASS" if rec["passed"] else (
                f"FAIL[{rec['failure_category']}]")
            print(f"  {rec['method']:18s} {status:22s} "
                  f"retries={rec['retries']} "
                  f"runtime={rec['validator_runtime_s']}s "
                  f"calls={CALL_COUNT['n']} ({time.time()-t0:.0f}s elapsed)",
                  flush=True)
            write_outputs([rec], journal=True)  # crash-safe partial state
    # stable order + final outputs
    order = {m["id"]: i for i, m in enumerate(selected)}
    records.sort(key=lambda r: order[r["method"]])
    stats = write_outputs(records, journal=False)
    (OUT_DIR / "journal.jsonl").unlink(missing_ok=True)
    print(json.dumps(stats, indent=1))
    print(f"outputs: {OUT_DIR}/examples.jsonl, {OUT_DIR}/stats.json "
          f"(api calls used: {CALL_COUNT['n']}/{CALL_BUDGET})")


if __name__ == "__main__":
    main()
