# Sequential prospective evaluation (preregistered, 1 pass; n=300)

Frozen controller (strict-trained heads primary; lenient-trained
variant disclosed) vs frozen mixture (= always→CROP-TR). Same-replay
decision+outcome, mean over replays; paired 10k bootstrap.

## strict scorer

### a1 = center

| policy | acc |
|---|--:|
| stop after a1 | 0.741 |
| frozen mixture (= always→TR) | 0.756 |
| always→HIGHRES | 0.748 |
| controller (strict-heads) | 0.739 (look 1.00) |
| controller (lenient-heads) | 0.736 (look 1.00) |
| sequential oracle (cross-replay) | 0.784 |

strict-heads − mixture = -0.0167 [-0.0356, +0.0017]; lenient-heads − mixture = -0.0206 [-0.0395, -0.0017]

### a1 = ocr

| policy | acc |
|---|--:|
| stop after a1 | 0.614 |
| frozen mixture (= always→TR) | 0.696 |
| always→HIGHRES | 0.659 |
| controller (strict-heads) | 0.697 (look 1.00) |
| controller (lenient-heads) | 0.693 (look 1.00) |
| sequential oracle (cross-replay) | 0.754 |

strict-heads − mixture = +0.0006 [-0.0139, +0.0150]; lenient-heads − mixture = -0.0033 [-0.0311, +0.0244]

## lenient scorer

### a1 = center

| policy | acc |
|---|--:|
| stop after a1 | 0.808 |
| frozen mixture (= always→TR) | 0.815 |
| always→HIGHRES | 0.817 |
| controller (strict-heads) | 0.803 (look 1.00) |
| controller (lenient-heads) | 0.803 (look 1.00) |
| sequential oracle (cross-replay) | 0.842 |

strict-heads − mixture = -0.0122 [-0.0289, +0.0044]; lenient-heads − mixture = -0.0122 [-0.0289, +0.0044]

### a1 = ocr

| policy | acc |
|---|--:|
| stop after a1 | 0.741 |
| frozen mixture (= always→TR) | 0.773 |
| always→HIGHRES | 0.803 |
| controller (strict-heads) | 0.781 (look 1.00) |
| controller (lenient-heads) | 0.800 (look 1.00) |
| sequential oracle (cross-replay) | 0.835 |

strict-heads − mixture = +0.0083 [-0.0050, +0.0233]; lenient-heads − mixture = +0.0272 [+0.0039, +0.0511]

## Oracle-gap CIs (state-paired 10k bootstrap; appended, same log)

| scorer | a1 | oracle − mixture(always→TR) [95% CI] | oracle − stop-after-a1 [95% CI] |
|---|---|:--|:--|
| strict | center | +0.0283 [+0.0111, +0.0478] | +0.0439 [+0.0194, +0.0689] |
| strict | ocr | +0.0578 [+0.0339, +0.0844] | +0.1400 [+0.1033, +0.1778] |
| lenient | center | +0.0272 [+0.0100, +0.0467] | +0.0344 [+0.0117, +0.0589] |
| lenient | ocr | +0.0622 [+0.0378, +0.0883] | +0.0939 [+0.0606, +0.1289] |
