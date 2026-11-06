
# Paper-reported aggregate results

This file summarizes the BHI results reported for the NeurIPS 2026 study.
Values are specific to the corresponding experimental snapshots and are not
expected results for arbitrary datasets.

The April benchmark-level results refer to the original 117-model/158-benchmark
submission snapshot. The June tables and temporal analysis refer to the
133-model/185-benchmark snapshot. The associated result tables and figures are
organized under `data/output/`.

The numerical results below have been checked against the included experiment
tables.

## Paper baselines

| Snapshot used by paper experiments | Models | Raw benchmarks | Ranked benchmarks | CRITIC weights: Disc / Sat / Imp |
|---|---:|---:|---:|---|
| April 2026 original submission | 117 | 416 | 158 | 0.3262540581 / 0.3574271392 / 0.3163188027 |
| June 2026 plus-one-model temporal snapshot | 133 | 486 | 185 | 0.3271684584 / 0.3745101946 / 0.2983213470 |

The June row is the 185-benchmark snapshot used in the paper's temporal
robustness analysis.

## Parameter sensitivity

### EDR threshold

| EDR | Spearman | Top-10 | wDisc | wSat | wImp |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.9582176558 | 10 | 0.3642880440 | 0.3386293449 | 0.2970826110 |
| 1.0 | 0.9870238332 | 9 | 0.3254842835 | 0.3597784973 | 0.3147372192 |
| 1.5 | 0.9962760075 | 9 | 0.3167772362 | 0.3644105491 | 0.3188122147 |
| 2.0 (default) | 1.0000000000 | 10 | 0.3262540575 | 0.3574271395 | 0.3163188030 |
| 2.5 | 0.9954727934 | 10 | 0.3382989447 | 0.3465352267 | 0.3151658286 |
| 3.0 | 0.9864822722 | 10 | 0.3603163936 | 0.3354116104 | 0.3042719961 |
| 4.0 | 0.9763842893 | 9 | 0.3785299271 | 0.3264851357 | 0.2949849373 |
| 5.0 | 0.9645794764 | 9 | 0.3913995197 | 0.3196101114 | 0.2889903689 |

### RCV percentile range

| RCV | Spearman | Kendall | Top-10 | Top-20 | Median / max abs rank change | wDisc | wSat | wImp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0-P100 | 0.9934982255 | 0.9463033137 | 9 | 17 | 2 / 29 | 0.3226851980 | 0.3754869579 | 0.3018278441 |
| P5-P95 | 0.9977576940 | 0.9716197694 | 9 | 19 | 1 / 21 | 0.3293331093 | 0.3620693233 | 0.3085975674 |
| P10-P90 (default) | 1.0000000000 | 1.0000000000 | 10 | 20 | 0 / 0 | 0.3262540581 | 0.3574271392 | 0.3163188027 |
| P15-P85 | 0.9986156727 | 0.9740385391 | 10 | 18 | 1 / 9 | 0.3291810190 | 0.3546247590 | 0.3161942220 |
| P20-P80 | 0.9962182004 | 0.9546883818 | 10 | 18 | 2 / 14 | 0.3272844560 | 0.3536643676 | 0.3190511764 |
| P25-P75 | 0.9922203849 | 0.9374344917 | 9 | 18 | 3 / 25 | 0.3160426693 | 0.3544274257 | 0.3295299051 |

### Open-upper-bound headroom factor

| Factor | Spearman | Kendall | Top-10 | Top-20 | Median / max abs rank change | wDisc | wSat | wImp |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.10 | 0.9978307135 | 0.9706522615 | 9 | 19 | 1 / 18 | 0.3150286183 | 0.3599980867 | 0.3249732950 |
| 1.20 | 0.9996562000 | 0.9888736596 | 9 | 19 | 1 / 5 | 0.3226218563 | 0.3580228619 | 0.3193552818 |
| 1.25 (default) | 1.0000000000 | 1.0000000000 | 10 | 20 | 0 / 0 | 0.3262540581 | 0.3574271392 | 0.3163188027 |
| 1.33 | 0.9997079222 | 0.9937111989 | 10 | 20 | 0 / 8 | 0.3238412716 | 0.3600075124 | 0.3161512160 |
| 1.50 | 0.9983966143 | 0.9832298637 | 9 | 19 | 1 / 20 | 0.3197800409 | 0.3658410612 | 0.3143788979 |

### Dynamic window and anti-saturation weighting

| Window (days) | Spearman | Kendall | Top-10 | Top-20 | Mean / max abs rank shift |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.9999056832 | 0.9966137225 | 10 | 20 | 0.2405063291 / 4 |
| 25 | 0.9999543628 | 0.9982262356 | 10 | 20 | 0.1392405063 / 3 |
| 30 (default) | 1.0000000000 | 1.0000000000 | 10 | 20 | 0 / 0 |
| 35 | 0.9999452354 | 0.9972587277 | 10 | 20 | 0.2151898734 / 2 |
| 40 | 0.9998904708 | 0.9954849633 | 10 | 20 | 0.3291139241 / 3 |

| Static / dynamic | Spearman | Kendall | Top-10 | Top-20 | Mean / max abs rank shift |
|---|---:|---:|---:|---:|---:|
| 0.6 / 0.4 | 0.9995253735 | 0.9888736596 | 10 | 20 | 0.7721518987 / 8 |
| 0.7 / 0.3 | 0.9997991965 | 0.9930661937 | 10 | 20 | 0.5063291139 / 3 |
| 0.8 / 0.2 (default) | 1.0000000000 | 1.0000000000 | 10 | 20 | 0 / 0 |
| 0.9 / 0.1 | 0.9997839841 | 0.9938724502 | 9 | 20 | 0.4177215190 / 6 |

## Gaussian-noise robustness

Each row summarizes 100 iterations with seed 42.

| Weighting | Sigma | Mean Spearman | SD | Mean Kendall | Mean BHI Pearson | Mean BHI MAE | Mean top-10 | Top-20 rate | Mean rank shift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CRITIC | 0.01 | 0.9910504002 | 0.0046300157 | 0.9353043618 | 0.9899441487 | 0.0101795328 | 9.67 | 0.9695 | 3.6997468354 |
| CRITIC | 0.02 | 0.9826836477 | 0.0071927193 | 0.9027009594 | 0.9808984862 | 0.0147554561 | 9.45 | 0.9545 | 5.5297468354 |
| CRITIC | 0.05 | 0.9543759802 | 0.0201656671 | 0.8329388051 | 0.9529015428 | 0.0244965202 | 8.90 | 0.9060 | 9.4439240506 |
| CRITIC | 0.10 | 0.9184221103 | 0.0286450676 | 0.7673514472 | 0.9156099668 | 0.0349637156 | 8.67 | 0.8660 | 12.9950632911 |
| CRITIC | 0.15 | 0.8854289361 | 0.0377966932 | 0.7188680158 | 0.8800495191 | 0.0456690906 | 8.44 | 0.8405 | 15.6905063291 |
| CRITIC | 0.20 | 0.8575418911 | 0.0318577621 | 0.6803999032 | 0.8500684591 | 0.0541838425 | 8.36 | 0.8105 | 17.7450632911 |
| Equal | 0.01 | 0.9908699204 | 0.0032137747 | 0.9305716359 | 0.9895559479 | 0.0107514041 | 9.19 | 0.9580 | 3.9920253165 |
| Equal | 0.02 | 0.9818626048 | 0.0055606481 | 0.8967475611 | 0.9796420813 | 0.0154901187 | 9.14 | 0.9290 | 5.9149367089 |
| Equal | 0.05 | 0.9523018016 | 0.0149657685 | 0.8257985971 | 0.9503149019 | 0.0256013683 | 8.78 | 0.8940 | 9.8949367089 |
| Equal | 0.10 | 0.9106154780 | 0.0230463369 | 0.7527307909 | 0.9081455918 | 0.0381277582 | 8.36 | 0.8670 | 13.8146835443 |
| Equal | 0.15 | 0.8747850109 | 0.0267224687 | 0.7013125857 | 0.8703950602 | 0.0496495454 | 7.93 | 0.8405 | 16.6749367089 |
| Equal | 0.20 | 0.8424731996 | 0.0264252067 | 0.6594598081 | 0.8365225285 | 0.0594751322 | 7.72 | 0.7950 | 18.8959493671 |

## Model-dropout robustness

Each row summarizes 100 iterations with seed 42.

| Dropout | CRITIC / equal Spearman | CRITIC / equal Kendall | CRITIC / equal top-10 | Models retained |
|---:|---:|---:|---:|---:|
| 0.05 | 0.9961180911 / 0.9955736326 | 0.9667463188 / 0.9585881191 | 9.31 / 8.83 | 112 |
| 0.10 | 0.9911257061 / 0.9909055199 | 0.9389490226 / 0.9353003646 | 9.14 / 8.90 | 106 |
| 0.15 | 0.9850891254 / 0.9853517537 | 0.9134413009 / 0.9130811532 | 9.04 / 8.81 | 100 |
| 0.20 | 0.9814368307 / 0.9814531034 | 0.8986117781 / 0.8983618071 | 8.91 / 8.75 | 94 |
| 0.25 | 0.9768915163 / 0.9768164624 | 0.8835099669 / 0.8833407822 | 8.75 / 8.69 | 88 |
| 0.30 | 0.9716041801 / 0.9717000901 | 0.8684032354 / 0.8691584514 | 8.48 / 8.55 | 82 |
| 0.35 | 0.9645813225 / 0.9651645929 | 0.8509413336 / 0.8527741256 | 8.47 / 8.63 | 77 |
| 0.40 | 0.9591303076 / 0.9600150859 | 0.8388053109 / 0.8410476285 | 8.29 / 8.49 | 71 |
| 0.45 | 0.9548650105 / 0.9553507531 | 0.8284663984 / 0.8302777400 | 8.25 / 8.46 | 65 |
| 0.50 | 0.9474493201 / 0.9479027762 | 0.8139537954 / 0.8151691509 | 7.86 / 8.03 | 59 |
| 0.55 | 0.9342943095 / 0.9366707540 | 0.7923130134 / 0.7966716262 | 7.74 / 8.16 | 53 |
| 0.60 | 0.9263999020 / 0.9300756634 | 0.7791179557 / 0.7856307139 | 7.55 / 7.86 | 47 |

## Targeted outliers

The zero-score experiment used 10 targets, 20 perturbation runs, 2,000
bootstrap resamples, and seed 20260726; all 220 invariants passed.

| Metric | Low abs effect | High abs effect | Median low/high ratio | Ratios below 1 |
|---|---:|---:|---:|---:|
| S_Static | 0.1046555410 | 0.2959867216 | 0.3256633810 | 10/10 |
| S_Sat | 0.1005237179 | 0.2871111479 | 0.3421335915 | 10/10 |
| BHI | 0.2047088407 | 0.2899069161 | 0.7331761056 | 10/10 |
| Rank percentile | 0.5636942675 | 0.7738853503 | 0.8179530750 | 10/10 |

| Zeroed group | Mean / min Spearman | Mean Kendall | Mean top-10 / top-20 | Mean BHI MAE |
|---|---:|---:|---:|---:|
| Low capability | 0.9845661199 / 0.9743792965 | 0.9661856003 | 9.4 / 19.2 | 0.0054940714 |
| High capability | 0.9752500536 / 0.9649293613 | 0.9610416835 | 9.4 / 18.9 | 0.0063401909 |

The fixed-decrement experiment used 100 runs, 2,000 bootstrap resamples, and
the same seed; all 1,300 invariants passed.

| Decrement | Low / high median abs BHI effect | Median ratio | Ratios below 1 | Low / high mean Spearman |
|---:|---:|---:|---:|---:|
| 10 | 0.0366044933 / 0.0355061060 | 0.9413004165 | 5/10 | 0.9995488006 / 0.9996796271 |
| 15 | 0.0463925064 / 0.0508166618 | 0.8839327698 | 5/10 | 0.9992917112 / 0.9993069236 |
| 20 | 0.0570596840 / 0.0678729863 | 0.8683295176 | 6/10 | 0.9989050123 / 0.9987425440 |
| 25 | 0.0694962470 / 0.0822363501 | 0.8612223079 | 6/10 | 0.9983540196 / 0.9979594103 |
| 30 | 0.0833792254 / 0.0972854269 | 0.8563995400 | 6/10 | 0.9977236183 / 0.9970369311 |

## Cold-start, coverage, correlation, and ablation

Cold-start used 157 verified dates, 10,000 bootstrap samples, and seed 42.

| Outcome | Method | N | Coefficient | p | Bootstrap 95% CI |
|---|---|---:|---:|---:|---|
| NUsage | Pearson | 157 | -0.0297233668 | 0.7117265732 | [-0.1184869899, 0.0693615462] |
| NUsage | Spearman | 157 | 0.0201437489 | 0.8022711453 | [-0.1331600242, 0.1735713787] |
| SImp | Pearson | 157 | -0.2192442085 | 0.0058010937 | [-0.3288566305, -0.1108126907] |
| SImp | Spearman | 157 | -0.2714247506 | 0.0005848435 | [-0.4047769531, -0.1211395478] |
| BHI | Pearson | 157 | -0.4030487902 | 1.662867e-07 | [-0.5122439122, -0.2782464858] |
| BHI | Spearman | 157 | -0.4396391593 | 8.375609e-09 | [-0.5640696165, -0.2916798013] |

Recent benchmarks (age at most 12 months) were top-quartile at 15/41 versus
25/116 for older benchmarks: risk difference 0.150336, odds ratio 2.1, Fisher
two-sided p=0.0639623. This is observational, not causal.

| Metric | Pearson / Spearman with coverage |
|---|---:|
| S_Disc | 0.2807326598 / 0.2423596994 |
| S_Sat | -0.0274881577 / -0.0332841868 |
| S_Imp | 0.9130048972 / 0.8862366756 |
| BHI | 0.4736981163 / 0.4000294186 |

| Pair | Pearson / Spearman |
|---|---:|
| S_Disc - S_Sat | 0.3861868146 / 0.4560540405 |
| S_Disc - S_Imp | 0.3028656102 / 0.2518171225 |
| S_Sat - S_Imp | 0.0759529362 / 0.1252071739 |

| Ablation | Spearman | Mean / max abs rank shift | Top-10 shared |
|---|---:|---:|---:|
| Without discrimination | 0.9073352612 | 15.5316455696 / 49 | 5 |
| Without anti-saturation | 0.8896402118 | 16.1772151899 / 82 | 5 |
| Without impact | 0.9416483231 | 11.3417721519 / 66 | 7 |

## April-June temporal robustness

The latest paper-era analysis used the 133-model, 185-benchmark June snapshot.

| Matching | Common | Spearman | Kendall | Top-10 / top-20 | Mean / median abs rank change | Median rank-percentile change | Top-quartile / half retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact names | 151 | 0.9848065528 | 0.9159381898 | 7 / 14 | 4.8609271523 / 3 | 0.0205621711 | 0.8947368421 / 0.9605263158 |
| Conservative aliases | 155 | 0.9849344204 | 0.9155425220 | 7 / 15 | 4.9548387097 / 3 | 0.0205621711 | 0.9230769231 / 0.9615384615 |

Exact-name BHI statistics: Pearson 0.9861882758; mean signed change
0.0078040124; mean, median, and maximum absolute changes 0.0121577576,
0.0064433664, and 0.0887155605. April-to-June CRITIC-weight changes were
+0.091440, +1.708306, and -1.799746 percentage points; total-variation
distance was 0.0179975.

Exact-name 95% bootstrap intervals (2,000 resamples, seed 20260726): Spearman
[0.9723032881, 0.9914212408], Kendall [0.8874654157, 0.9393939394], median
absolute rank-percentile change [0.0146081418, 0.0255815564], and median
absolute BHI change [0.0054914006, 0.0083584030].

## External-validation cohorts

These values use sample standard deviation.

| Cohort | Mean BHI +/- sample SD |
|---|---:|
| Known-saturated | 0.225 +/- 0.036 |
| Active frontier | 0.605 +/- 0.045 |

