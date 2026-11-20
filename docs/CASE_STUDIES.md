# Selected project case studies

These notes describe research decisions in the broader project. The historical numbers below are selected summaries from reviewed project documentation; they are separate from the reproducible synthetic report in this repository. They are not an independently audited live track record.

## Football: evaluate the probability distribution, not only the predicted winner

**Question.** Can a time-decayed, regularized Poisson/Dixon–Coles model improve a simple relative-strength benchmark while preserving coherent three-way probabilities?

The national-team project fitted team attack and defensive weakness, a venue effect, and a low-score correction. Source scores that included extra time had to be reconciled to regulation-time goals. Event-year coverage filtering removed unresolved observations, improving target consistency while introducing a possible selection effect.

A historical snapshot documented quarterly rolling evaluation on 2,815 matches after 2021-01-01:

| Historical method | Log loss | Brier, half-sum convention | Highest-probability accuracy |
| --- | ---: | ---: | ---: |
| Poisson / Dixon–Coles model | 0.842217 | 0.2475745 | 61.28% |
| Internal Davidson strength baseline | 0.873047 | 0.2558865 | 60.04% |

The source report used summed three-class Brier (0.495149 and 0.511773); the table divides each by two to match this repository's convention. No market comparison or profit conclusion follows from beating that internal baseline. Original publication timestamps and later source revisions were not fully recoverable. Development selection was not rerun after a later data supplement. These are retrospective, consumed research results.

**Public implementation:** goal-rate-to-outcome distribution with explicit rho feasibility and numerical-tail checks. The original observations, fitted team coefficients and historical evaluation rows are not distributed, so this repository does not reproduce that historical table.

## Basketball: distinguish workload, scoring rate and realized clutch information

**Question.** Does raw whole-game margin confound strength with overtime duration, and can a pregame expected-clutch contribution improve forecasts?

A reviewed basketball research snapshot documented 6,213 eligible base games and 6,011 accepted clutch-analysis games. The design normalized whole-game margin and strength minutes to a 48-minute equivalent, while preserving actual overtime minutes for workload. It separated the event triggering a clutch interval from subsequent scoring.

At forecast time, expected clutch occurrence/duration came from earlier out-of-fold predictions, not the target game's realized trigger, minutes or lineup. This is the central leakage-control issue: explaining a close game's ending with information observed only after tip-off does not establish a pregame forecasting improvement.

The selected candidate had lower historical log loss, but uncertainty intervals included zero. The appropriate conclusion was that a stable improvement had not been established, not that the added features necessarily created an edge. Team/player effects were regularized associations, not causal measures of composure.

**Public implementation:** a simpler time-weighted ridge margin model and Gaussian win mapping, with synthetic chronological evaluation. It excludes the player-minute and clutch pipelines and explicitly documents the in-sample residual-scale limitation.

## Snooker / table tennis / esports: a unit probability is not a match probability

**Question.** How do format, dependence and sparse histories affect the mapping from a frame/game/map win probability to match outcomes?

The broader research uses sport-specific strength and state adjustments. The snooker model family includes a Bradley–Terry frame-strength component and beta-binomial match mapping. A longer series can amplify a favourite under independence; shared match-level conditions can reduce that amplification. Applying the wrong best-of length changes the answer even when the underlying strength estimate is unchanged.

Historical model comparisons and data-format repairs are distinct exercises. Correcting a match identity, score or format can change the fitted state; it should not be described as a fresh prospective validation. A useful next experiment is an untouched, date-bounded comparison of iid and overdispersed mappings, stratified by format and history depth.

**Public implementation:** weighted Bradley–Terry fitting plus iid and beta-binomial series mappings, including symmetry, limiting-case, and invalid-format tests. Sport-specific production feature weights are excluded.

## Scope map

| Broader research domain | Public representation | Not distributed |
| --- | --- | --- |
| NBA and FIBA basketball | Margin-model reference; NBA design case study | Player states, live lineups, fitted production coefficients |
| Club and national-team football | Poisson/DC distribution; national-team historical summary | Production team/player parameters and replay state |
| Snooker | Strength and beta-binomial series reference | Workload/travel coefficients and online player corrections |
| Table tennis | Strength/format research discussion | Current fitted strengths and decision filters |
| League of Legends | Generic series-probability machinery | Team/player/version interaction state |

This map describes research coverage, not seven complete publicly runnable production models.
