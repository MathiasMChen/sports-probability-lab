# Modelling decisions

## 1. Relative strength and time weighting

For participant i against j, the reference strength model is

$$p_{ij}=\operatorname{logit}^{-1}(\theta_i-\theta_j),\qquad w_t=2^{-(T-t)/h}.$$

The fitting objective is weighted binomial negative log likelihood plus an L2 penalty. Pairwise differences determine probabilities; centring the fitted vector makes the exported strength representation identifiable. Penalization reduces extreme estimates for sparse participants, but it does not make sparse estimates reliable.

The code requires every training observation to occur strictly before cutoff T. The demo refits only at the start of each test block and holds that fit fixed during the block. Production systems may update online, but must replay only observations available before each forecast.

Frame/game totals in real first-to-N competitions are stopping-time observations. A binomial likelihood applied to aggregate played frames is a simplifying assumption and can be affected by match selection and within-match dependence. The synthetic fitting example uses one binary trial per match, avoiding a claim that this issue has been solved.

## 2. Series length and latent match conditions

For an odd best-of N format, the iid mapping sums the Binomial(N,p) mass above N/2. Equivalently it is the first-to-majority probability when hypothetical unplayed units can be sampled independently.

The beta-binomial variant draws a shared latent match probability q from Beta(kappa*p, kappa*(1-p)). Finite kappa allows positive dependence between units. It usually reduces the amplification of a moderate favourite relative to the iid mapping. This is not a generic cure for psychological uncertainty or unfamiliar opponents; kappa requires appropriate validation.

## 3. Football score distributions

The larger project uses regularized attack/defensive-weakness goal intensities, with time decay and sport-specific context. This public module starts from supplied goal rates rather than claiming to retrain that entire model.

For independent Poisson home and away goals, Dixon–Coles factors modify only (0,0), (0,1), (1,0), and (1,1). The code rejects infeasible rho values instead of silently clipping them. It chooses a score-grid bound by a Poisson tail criterion, reports discarded mass, and normalizes the retained distribution. H/D/A are mutually exclusive; double chance should be derived by summing that same distribution.

The outcome target must be defined first: regulation-time football results and results including extra time are not interchangeable.

## 4. Basketball margins

The reference fit minimizes time-weighted squared margin error with ridge shrinkage. The first design column is an unpenalized intercept. A Gaussian residual assumption converts the conditional mean to a win probability.

The demonstration includes anonymous team effects and a synthetic rest differential. The broader project's player-minute, normalized-duration and expected-clutch components are described in the case studies, not implemented here.

Residual scale is estimated in sample in this simplified implementation. That can make confidence optimistic. A stronger extension would estimate scale/calibration from earlier out-of-fold residuals, permit conditional variance, and compare its added complexity on later untouched periods. Full-sample standardization before splitting would also leak future information; any learned transformation belongs inside each fold.

## 5. Proper scores and paired comparisons

All public-code Brier scores use

$$\mathrm{Brier}=\frac12\sum_k(p_k-y_k)^2.$$

This reduces to (p-y)^2 for binary outcomes. Three-class scores from sources using the unhalved sum must be divided by two before comparison. Log loss uses natural logarithms. Valid zero probabilities are accepted, actual zero-probability outcomes are disclosed, and log loss uses a 1e-15 numerical floor. A missing probability is never replaced with zero.

Model/comparator comparisons require both complete distributions and an outcome. When no pair is available, N is zero and the scores are None. Invalid distributions fail validation. The standalone model sample can be larger, but its score cannot be compared directly to a market score from a smaller sample.

The bootstrap resamples whole date clusters and estimates the observation-weighted mean model-minus-baseline loss difference. It does not model dependence across dates, resolve multiple testing, or turn a consumed historical test into new evidence.

## 6. Risk as a separate research layer

For decimal price O and event probability p, the non-negative Kelly fraction is

$$f^*=\max\left(0,\frac{pO-1}{O-1}\right).$$

The example multiplies this by a fixed fractional policy, samples hypothetical outcomes from supplied p, and tracks the maximum peak-to-trough equity drawdown including initial equity. It reports a drawdown quantile, threshold exceedance rate, and an exact one-sided 95% binomial upper bound for Monte Carlo uncertainty.

This assumes known, calibrated p and independent sequential events with no overlapping positions, costs, limits, or rounding. It does not capture parameter uncertainty, shared-event correlation, changing prices, liquidity, or the larger system's daily starting-balance and cash constraints. It is a methodological illustration, not a recommended betting fraction.
