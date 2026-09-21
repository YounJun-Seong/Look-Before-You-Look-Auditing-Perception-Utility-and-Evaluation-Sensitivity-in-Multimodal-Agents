# Deployable-action restricted oracle (robustness check of the structural oracle)

full = oracle over the individual arms {STOP, CROP#0..7, OCR, HIGHRES}; dep = oracle over the
deployable decision space {STOP, CROP-family (uniform-grid expectation), OCR, HIGHRES}. Both:
select on replay 1 (ties → cheaper family), score on replay 2; headroom H = oracle − the SAME
cross-fitted best fixed arm (audit rule, 11 fixed arms). 10k state-paired bootstrap CIs. pp.

## textvqa / Gate-0 (Qwen, n=200)  (n=200, always-HIGHRES 308 tok)

| scorer | best fixed | full oracle | dep oracle | H_full [CI] | H_dep [CI] | full − dep [CI] | dep/full | sym: H_full / H_dep |
|---|--:|--:|--:|:--|:--|:--|--:|:--|
| strict | 0.694 | 0.783 | 0.740 | +8.9 [+4.0, +13.9] | +4.6 [-0.1, +9.3] | +4.3 [+1.7, +7.1] | 0.51 | +11.4 / +7.3 |
| lenient | 0.801 | 0.883 | 0.854 | +8.3 [+4.7, +12.1] | +5.3 [+2.6, +8.2] | +3.0 [+1.2, +5.0] | 0.64 | +8.4 / +5.9 |

Budget-constrained oracle accuracy (Lagrangian, cross-replay symmetrised), full / dep:

| scorer | ¼× | ½× | 1× |
|---|:--|:--|:--|
| strict | 0.806 / 0.767 | 0.806 / 0.767 | 0.840 / 0.792 |
| lenient | 0.878 / 0.858 | 0.878 / 0.858 | 0.900 / 0.867 |

## mme / Gate-0 (Qwen, n=196)  (n=196, always-HIGHRES 994 tok)

| scorer | best fixed | full oracle | dep oracle | H_full [CI] | H_dep [CI] | full − dep [CI] | dep/full | sym: H_full / H_dep |
|---|--:|--:|--:|:--|:--|:--|--:|:--|
| strict | 0.413 | 0.495 | 0.443 | +8.2 [+3.3, +13.2] | +3.0 [-1.1, +7.2] | +5.2 [+2.1, +8.5] | 0.37 | +7.7 / +1.8 |

Budget-constrained oracle accuracy (Lagrangian, cross-replay symmetrised), full / dep:

| scorer | ¼× | ½× | 1× |
|---|:--|:--|:--|
| strict | 0.495 / 0.439 | 0.510 / 0.454 | 0.510 / 0.454 |

## textvqa / held-out 250  (n=250, always-HIGHRES 307 tok)

| scorer | best fixed | full oracle | dep oracle | H_full [CI] | H_dep [CI] | full − dep [CI] | dep/full | sym: H_full / H_dep |
|---|--:|--:|--:|:--|:--|:--|--:|:--|
| strict | 0.771 | 0.851 | 0.816 | +7.9 [+4.2, +11.6] | +4.4 [+1.1, +7.6] | +3.5 [+1.8, +5.5] | 0.56 | +8.0 / +4.7 |
| lenient | 0.862 | 0.919 | 0.898 | +5.7 [+2.1, +9.1] | +3.6 [+0.4, +6.6] | +2.1 [+0.7, +3.7] | 0.64 | +5.7 / +3.9 |

Budget-constrained oracle accuracy (Lagrangian, cross-replay symmetrised), full / dep:

| scorer | ¼× | ½× | 1× |
|---|:--|:--|:--|
| strict | 0.851 / 0.821 | 0.851 / 0.821 | 0.866 / 0.826 |
| lenient | 0.921 / 0.905 | 0.921 / 0.905 | 0.935 / 0.908 |

## textvqa / prospective 600  (n=600, always-HIGHRES 307 tok)

| scorer | best fixed | full oracle | dep oracle | H_full [CI] | H_dep [CI] | full − dep [CI] | dep/full | sym: H_full / H_dep |
|---|--:|--:|--:|:--|:--|:--|--:|:--|
| strict | 0.709 | 0.807 | 0.765 | +9.8 [+7.1, +12.6] | +5.6 [+3.1, +8.2] | +4.2 [+2.8, +5.7] | 0.57 | +10.8 / +6.5 |
| lenient | 0.827 | 0.899 | 0.865 | +7.2 [+5.1, +9.6] | +3.9 [+2.1, +5.7] | +3.4 [+2.2, +4.7] | 0.53 | +7.2 / +4.0 |

Budget-constrained oracle accuracy (Lagrangian, cross-replay symmetrised), full / dep:

| scorer | ¼× | ½× | 1× |
|---|:--|:--|:--|
| strict | 0.819 / 0.777 | 0.819 / 0.777 | 0.847 / 0.807 |
| lenient | 0.899 / 0.868 | 0.899 / 0.868 | 0.906 / 0.876 |

## textvqa / v3-prospective 300  (n=300, always-HIGHRES 307 tok)

| scorer | best fixed | full oracle | dep oracle | H_full [CI] | H_dep [CI] | full − dep [CI] | dep/full | sym: H_full / H_dep |
|---|--:|--:|--:|:--|:--|:--|--:|:--|
| strict | 0.698 | 0.812 | 0.765 | +11.4 [+8.0, +15.1] | +6.7 [+4.0, +9.4] | +4.7 [+2.5, +7.1] | 0.59 | +12.3 / +7.6 |
| lenient | 0.818 | 0.889 | 0.848 | +7.1 [+3.6, +10.7] | +3.1 [+0.5, +5.7] | +4.0 [+2.0, +6.2] | 0.43 | +7.5 / +3.6 |

Budget-constrained oracle accuracy (Lagrangian, cross-replay symmetrised), full / dep:

| scorer | ¼× | ½× | 1× |
|---|:--|:--|:--|
| strict | 0.816 / 0.772 | 0.816 / 0.772 | 0.826 / 0.781 |
| lenient | 0.887 / 0.848 | 0.887 / 0.848 | 0.890 / 0.854 |

## textvqa / strong 400  (n=400, always-HIGHRES 308 tok)

| scorer | best fixed | full oracle | dep oracle | H_full [CI] | H_dep [CI] | full − dep [CI] | dep/full | sym: H_full / H_dep |
|---|--:|--:|--:|:--|:--|:--|--:|:--|
| strict | 0.710 | 0.803 | 0.773 | +9.4 [+6.3, +12.5] | +6.3 [+3.6, +9.0] | +3.1 [+1.2, +5.0] | 0.67 | +10.4 / +7.3 |
| lenient | 0.839 | 0.909 | 0.888 | +7.0 [+4.2, +9.9] | +4.9 [+2.7, +7.3] | +2.1 [+0.7, +3.6] | 0.70 | +6.3 / +4.5 |

Budget-constrained oracle accuracy (Lagrangian, cross-replay symmetrised), full / dep:

| scorer | ¼× | ½× | 1× |
|---|:--|:--|:--|
| strict | 0.810 / 0.777 | 0.810 / 0.777 | 0.835 / 0.802 |
| lenient | 0.901 / 0.882 | 0.901 / 0.882 | 0.908 / 0.890 |

## textvqa / hidden 400  (n=400, always-HIGHRES 307 tok)

| scorer | best fixed | full oracle | dep oracle | H_full [CI] | H_dep [CI] | full − dep [CI] | dep/full | sym: H_full / H_dep |
|---|--:|--:|--:|:--|:--|:--|--:|:--|
| strict | 0.716 | 0.794 | 0.757 | +7.8 [+4.7, +10.9] | +4.1 [+1.5, +6.8] | +3.7 [+2.1, +5.4] | 0.53 | +9.8 / +6.0 |
| lenient | 0.811 | 0.877 | 0.854 | +6.6 [+4.2, +9.3] | +4.3 [+2.2, +6.5] | +2.4 [+1.2, +3.7] | 0.64 | +7.5 / +5.1 |

Budget-constrained oracle accuracy (Lagrangian, cross-replay symmetrised), full / dep:

| scorer | ¼× | ½× | 1× |
|---|:--|:--|:--|
| strict | 0.809 / 0.777 | 0.809 / 0.777 | 0.837 / 0.806 |
| lenient | 0.887 / 0.865 | 0.887 / 0.865 | 0.887 / 0.869 |

## textvqa / Gate-0 Phi-4 (n=200)  (n=200, always-HIGHRES 308 tok)

| scorer | best fixed | full oracle | dep oracle | H_full [CI] | H_dep [CI] | full − dep [CI] | dep/full | sym: H_full / H_dep |
|---|--:|--:|--:|:--|:--|:--|--:|:--|
| strict | 0.335 | 0.433 | 0.409 | +9.9 [+5.7, +13.9] | +7.5 [+4.1, +10.8] | +2.4 [+0.4, +4.5] | 0.76 | +6.8 / +4.8 |
| lenient | 0.624 | 0.737 | 0.698 | +11.2 [+6.5, +16.0] | +7.4 [+3.6, +11.2] | +3.9 [+1.2, +6.7] | 0.66 | +10.5 / +6.8 |

Budget-constrained oracle accuracy (Lagrangian, cross-replay symmetrised), full / dep:

| scorer | ¼× | ½× | 1× |
|---|:--|:--|:--|
| strict | 0.418 / 0.405 | 0.418 / 0.405 | 0.418 / 0.405 |
| lenient | 0.739 / 0.708 | 0.739 / 0.708 | 0.739 / 0.717 |

## mme / Gate-0 Phi-4 (n=196)  (n=196, always-HIGHRES 994 tok)

| scorer | best fixed | full oracle | dep oracle | H_full [CI] | H_dep [CI] | full − dep [CI] | dep/full | sym: H_full / H_dep |
|---|--:|--:|--:|:--|:--|:--|--:|:--|
| strict | 0.349 | 0.372 | 0.365 | +2.3 [-2.9, +7.6] | +1.5 [-3.0, +6.1] | +0.8 [-1.4, +2.9] | 0.66 | +1.0 / +0.1 |

Budget-constrained oracle accuracy (Lagrangian, cross-replay symmetrised), full / dep:

| scorer | ¼× | ½× | 1× |
|---|:--|:--|:--|
| strict | 0.403 / 0.328 | 0.403 / 0.383 | 0.403 / 0.383 |
