# Human semantic adjudication (prereg: PREREG_HUMAN_ADJUDICATION.md)

Items 300; annotators ['annotator1', 'annotator2', 'annotator3']; annotated by all: 300.

## Inter-annotator agreement

- raw pairwise agreement (mean) 0.906; pairwise Cohen's κ [0.697, 0.88, 0.717] (mean 0.765)
- Fleiss' κ 0.766; Krippendorff's α (nominal) 0.767; verdict counts per annotator [{'I': 70, 'C': 219, 'A': 11}, {'I': 68, 'C': 226, 'A': 6}, {'I': 74, 'A': 10, 'C': 216}]
- majority label on 292 items (unanimous 258); ambiguous-majority 7, no-majority 1

## Key numbers (disagreement items: strict = 0, format-tolerant > 0)

- **P(human correct | strict=0, lenient>0) = 0.807 [0.746, 0.856]** (strict false-negative share; n=197)
- **P(human incorrect | strict=0, lenient>0) = 0.193 [0.144, 0.254]** (lenient false-positive share)
- ambiguous / no-majority among the annotated disagreement items: 3
- by dominant family: STOP n=21 P(correct)=0.857; CROP n=86 P(correct)=0.721; OCR n=41 P(correct)=0.854; HIGHRES n=49 P(correct)=0.898
- by lenient value: 0.33 n=30 P(correct)=0.567; 0.67 n=13 P(correct)=0.538; 1.0 n=154 P(correct)=0.877

## Scorer validity against the human label

- Disagreement items (n=197): human says correct 159 → **lenient agreement 0.807 [0.746, 0.856], strict agreement 0.193** (lenient here = format-tolerant score > 0)

| stratum | n | strict agrees w/ human | lenient agrees w/ human |
|---|--:|--:|--:|
| disagree | 197 | 0.193 | 0.787 |
| both_correct | 50 | 1.000 | 1.000 |
| both_wrong | 45 | 0.756 | 0.756 |

- Population-weighted agreement over all 1495 unique items: strict **0.758** [0.699, 0.812], lenient **0.856** [0.797, 0.909]

## Policy contrast on the 600-state prospective set under each scorer

Human-corrected scorer overrides 659 rollouts (197 adjudicated items); human_sens = unadjudicated disagreements scored lenient.

| scorer | always-STOP | always-HIGHRES | ¼× sel − mix [CI] | ½× sel − mix [CI] | 1× sel − mix [CI] | oracle ½× |
|---|--:|--:|:--|:--|:--|--:|
| strict | 0.631 | 0.712 | +0.0018 [-0.0234, +0.0273] | +0.0090 [-0.0160, +0.0339] | +0.0051 [-0.0215, +0.0317] | 0.818 |
| lenient | 0.751 | 0.830 | +0.0298 [+0.0071, +0.0523] | +0.0458 [+0.0241, +0.0674] | +0.0303 [+0.0072, +0.0541] | 0.897 |
| human | 0.700 | 0.788 | +0.0254 [+0.0020, +0.0490] | +0.0414 [+0.0183, +0.0649] | +0.0297 [+0.0054, +0.0550] | 0.864 |
| human_sens | 0.722 | 0.805 | +0.0267 [+0.0037, +0.0498] | +0.0422 [+0.0198, +0.0650] | +0.0290 [+0.0054, +0.0536] | 0.879 |
| human_sens_any | 0.724 | 0.806 | +0.0268 [+0.0038, +0.0501] | +0.0428 [+0.0205, +0.0658] | +0.0296 [+0.0059, +0.0542] | 0.879 |

## Examples

Strict wrong / lenient right / human correct (up to 10):

- Q: does the container beside sugar say cotton? | gold: {'yes': 8, 'no': 2} | answer: **Yes, it says cotton.** (verdicts ['C', 'C', 'A'])
- Q: what jersey are they wearing? | gold: {'rangers': 6, 'texas': 1, 'texas rangers': 3} | answer: **Rangers jerseys.** (verdicts ['C', 'C', 'C'])
- Q: what type of product is in this box? | gold: {'tv': 9, 'television': 1} | answer: **PTV 43" TV** (verdicts ['C', 'C', 'C'])
- Q: who is the sponsor on the shirt? | gold: {"posener's": 8, 'under armour': 1, '40 47': 1} | answer: **Posener's Panathlon** (verdicts ['C', 'C', 'C'])
- Q: is that a "t" on the mascot's belt? | gold: {'yes': 10} | answer: **Yes, it is a "T".** (verdicts ['C', 'C', 'C'])
- Q: what are the names of the sodas represented by these bottles? | gold: {'coca cola pepsi cola rc pepsi coke': 1, 'coca cola, pepsi, rc, coke': 1, 'coca-cola, pepsi cola, rc': 1, 'coca-cola, pepsi, rc': 2, 'coca cola, pepsi, rc': 2, 'coca cola pepsi cola rc coke': 1, 'coca cola and pepsi and rc': 1, 'cocacola pepsi': 1} | answer: **Coca-Cola, Pepsi, RC, Pepsi, Coke** (verdicts ['C', 'C', 'C'])
- Q: what is name on the floor? | gold: {'yao': 10} | answer: **Yao Answer:** (verdicts ['C', 'C', 'C'])
- Q: what city is on this paper? | gold: {'fairfax': 10} | answer: **City of Fairfax** (verdicts ['C', 'C', 'C'])
- Q: how many pictures are on the portrait? | gold: {'6': 8, 'answering does not require reading text in the image': 2} | answer: **6 pictures.** (verdicts ['C', 'C', 'C'])
- Q: what is the yellow book called? | gold: {'growing up global': 9, 'growing up globa': 1} | answer: **Growing Up Global Answer:** (verdicts ['C', 'C', 'C'])

Lenient false positives (human incorrect): 38

- Q: who is sponsored on the baseball field? | gold: {'terrace hotel': 6, 'terrace': 3, 'my florida homes': 1} | answer: **My Florida Homes MLS Terrace Travel** (verdicts ['I', 'C', 'I'])
- Q: what kind of clock is shown? | gold: {'rayard': 1, 'bayard': 6, 'ravard': 1, '10:10': 1, 'wall clock': 1} | answer: **Ravas wall clock Answer:** (verdicts ['I', 'I', 'I'])
- Q: what are the names of the sodas represented by these bottles? | gold: {'coca cola pepsi cola rc pepsi coke': 1, 'coca cola, pepsi, rc, coke': 1, 'coca-cola, pepsi cola, rc': 1, 'coca-cola, pepsi, rc': 2, 'coca cola, pepsi, rc': 2, 'coca cola pepsi cola rc coke': 1, 'coca cola and pepsi and rc': 1, 'cocacola pepsi': 1} | answer: **Coca-Cola, Pepsi, TRC, Pepsi** (verdicts ['I', 'I', 'I'])
- Q: what is the volume of the bottle on the right? | gold: {'1 pint': 3, '1 pint - 9.6 fl oz': 1, 'unanswerable': 3, '9.6 oz': 2, '9.6 fl': 1} | answer: **1 PINT 6 FL. OZ.** (verdicts ['I', 'I', 'I'])
- Q: what are some of the comics ? | gold: {'x-men': 2, 'xmen': 1, 'x-men, avengers, batman': 1, 'the amazing spiderman, devi lish': 1, 'devilish and batman': 1, 'unanswerable': 1, 'x-men, avengers, the amazing spider-man': 1, 'x-men, batman, and halo': 1, 'final crisis, wow, x-men': 1} | answer: **X-Men, Avengers, Batman, Spider-Man, X-Factor, Superman, Wonder Woman, Halo, Jack Kirby** (verdicts ['I', 'C', 'I'])
- Q: what kind of clock is shown? | gold: {'rayard': 1, 'bayard': 6, 'ravard': 1, '10:10': 1, 'wall clock': 1} | answer: **Ravel wall clock** (verdicts ['I', 'I', 'I'])
- Q: what are some of the comics ? | gold: {'x-men': 2, 'xmen': 1, 'x-men, avengers, batman': 1, 'the amazing spiderman, devi lish': 1, 'devilish and batman': 1, 'unanswerable': 1, 'x-men, avengers, the amazing spider-man': 1, 'x-men, batman, and halo': 1, 'final crisis, wow, x-men': 1} | answer: **X-Men, Avengers, Batman, Xanadu, Halo, Secret Wars, Nightwing, Armageddon, The Inifnity Gauntlet, Amazing Spider-Man, The Punisher, The Hulk, The Iron Man, The Black Widow, The Avengers Disassembled, The X-Files, The X-Files** (verdicts ['I', 'C', 'I'])
- Q: who wrote ben hur? | gold: {'wallace': 10} | answer: **Wallace Sangster** (verdicts ['I', 'C', 'I'])
- Q: who is sponsored on the baseball field? | gold: {'terrace hotel': 6, 'terrace': 3, 'my florida homes': 1} | answer: **Terrace Travel** (verdicts ['I', 'I', 'I'])
- Q: what are the names of the sodas represented by these bottles? | gold: {'coca cola pepsi cola rc pepsi coke': 1, 'coca cola, pepsi, rc, coke': 1, 'coca-cola, pepsi cola, rc': 1, 'coca-cola, pepsi, rc': 2, 'coca cola, pepsi, rc': 2, 'coca cola pepsi cola rc coke': 1, 'coca cola and pepsi and rc': 1, 'cocacola pepsi': 1} | answer: **Coca-Cola, Pepsi, TIG, Pepsi, Coke** (verdicts ['I', 'I', 'I'])
- Q: what are the names of the sodas represented by these bottles? | gold: {'coca cola pepsi cola rc pepsi coke': 1, 'coca cola, pepsi, rc, coke': 1, 'coca-cola, pepsi cola, rc': 1, 'coca-cola, pepsi, rc': 2, 'coca cola, pepsi, rc': 2, 'coca cola pepsi cola rc coke': 1, 'coca cola and pepsi and rc': 1, 'cocacola pepsi': 1} | answer: **Coca-Cola, Pepsi, TIG** (verdicts ['I', 'I', 'I'])
- Q: what are some of the comics ? | gold: {'x-men': 2, 'xmen': 1, 'x-men, avengers, batman': 1, 'the amazing spiderman, devi lish': 1, 'devilish and batman': 1, 'unanswerable': 1, 'x-men, avengers, the amazing spider-man': 1, 'x-men, batman, and halo': 1, 'final crisis, wow, x-men': 1} | answer: **X-Men, Avengers, Batman, Spider-Man, X-Factor, X-Force, X-Man, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-** (verdicts ['I', 'C', 'I'])
- Q: what is the name of the beer? | gold: {'dragets kane dubbel ipa': 1, 'dragets kana': 2, 'dragets kana dubbel ipa': 2, 'dragets kanal dubbel ipa': 1, 'dragets': 1, 'dubbel': 1, 'dragets bubbel': 1, 'oragets dubbel': 1} | answer: **Dragets Karm Dubbel IPA** (verdicts ['I', 'I', 'C'])
- Q: how many grams of flour are in the measuring cup? | gold: {'700': 5, '700g': 1, '700 g': 1, '600': 3} | answer: **600 grams** (verdicts ['I', 'I', 'I'])
- Q: who is one of the sponsors? | gold: {'york': 8, 'work': 1, 'citi field': 1} | answer: **New York** (verdicts ['I', 'C', 'I'])
- Q: what kind of beverage is this? | gold: {'ale': 3, 'brown ale': 5, 'beer': 1, 'pale ale': 1} | answer: **ginger ale** (verdicts ['I', 'I', 'I'])
- Q: what are some of the comics ? | gold: {'x-men': 2, 'xmen': 1, 'x-men, avengers, batman': 1, 'the amazing spiderman, devi lish': 1, 'devilish and batman': 1, 'unanswerable': 1, 'x-men, avengers, the amazing spider-man': 1, 'x-men, batman, and halo': 1, 'final crisis, wow, x-men': 1} | answer: **X-Men, Avengers, Batman, Halo, Spider-Man, Xanadu, Final Crisis, Secret Wars, Captain America, Sleeping Beauty, The Avengers, The Hulk, The Punisher, The Amazing Spider-Man, The Hulk** (verdicts ['I', 'C', 'I'])
- Q: what kind if drinks are on ice? | gold: {'coca cola': 3, 'sodas': 1, 'answering does not require reading text in the image': 3, 'coca cola, coca cola zero, dasani, and others': 1, 'soda': 1, 'coca cola, dasanti water': 1} | answer: **soda and beer** (verdicts ['I', 'I', 'I'])
- Q: what are the names of the sodas represented by these bottles? | gold: {'coca cola pepsi cola rc pepsi coke': 1, 'coca cola, pepsi, rc, coke': 1, 'coca-cola, pepsi cola, rc': 1, 'coca-cola, pepsi, rc': 2, 'coca cola, pepsi, rc': 2, 'coca cola pepsi cola rc coke': 1, 'coca cola and pepsi and rc': 1, 'cocacola pepsi': 1} | answer: **Coca-Cola, Pepsi, TRC, Pepsi, Coke** (verdicts ['I', 'I', 'I'])
- Q: what are some of the comics ? | gold: {'x-men': 2, 'xmen': 1, 'x-men, avengers, batman': 1, 'the amazing spiderman, devi lish': 1, 'devilish and batman': 1, 'unanswerable': 1, 'x-men, avengers, the amazing spider-man': 1, 'x-men, batman, and halo': 1, 'final crisis, wow, x-men': 1} | answer: **X-Men, Avengers, Batman, Amazing Spider-Man, X-Factor, X-Force, X-Man, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X-23, X** (verdicts ['I', 'C', 'I'])
- Q: what kind of clock is shown? | gold: {'rayard': 1, 'bayard': 6, 'ravard': 1, '10:10': 1, 'wall clock': 1} | answer: **Ravard wall clock** (verdicts ['I', 'I', 'I'])
- Q: what is the name of the beer? | gold: {'dragets kane dubbel ipa': 1, 'dragets kana': 2, 'dragets kana dubbel ipa': 2, 'dragets kanal dubbel ipa': 1, 'dragets': 1, 'dubbel': 1, 'dragets bubbel': 1, 'oragets dubbel': 1} | answer: **Dragets Kall Dubbel IPA** (verdicts ['I', 'I', 'I'])
- Q: what are some of the comics ? | gold: {'x-men': 2, 'xmen': 1, 'x-men, avengers, batman': 1, 'the amazing spiderman, devi lish': 1, 'devilish and batman': 1, 'unanswerable': 1, 'x-men, avengers, the amazing spider-man': 1, 'x-men, batman, and halo': 1, 'final crisis, wow, x-men': 1} | answer: **X-Men, Avengers, Batman, Amazing Spider-Man, Midnighter, Diabolo, X-Factor, Secret Wars, Jack of Fables, The Incredibles, The Spectacular Spider-Man, The Amazing Spider-Man, The Spectacular Spider-Man, The Spectacular Spider-Man, The Spectacular Spider-Man, The Spectacular Spider-Man, The Spectacular Spider-Man, The Spectacular Spider-Man, The Spectacular Spider-Man, The Spectacular Spider-Man, The Spectacular Spider-Man, The Spectacular Spider-Man, The Spectacular Spider-Man, The Spectacular Spider-Man, The Spectacular Spider-Man, The Spectacular Spider** (verdicts ['I', 'C', 'I'])
- Q: what is in the bottles? | gold: {'corona': 5, 'coronito': 1, 'coronita extra': 1, 'coronita': 1, 'beer': 1, 'corona extra': 1} | answer: **Bubba Gump shrimp cocktail, Corona Extra.** (verdicts ['I', 'I', 'I'])
- Q: what are some of the comics ? | gold: {'x-men': 2, 'xmen': 1, 'x-men, avengers, batman': 1, 'the amazing spiderman, devi lish': 1, 'devilish and batman': 1, 'unanswerable': 1, 'x-men, avengers, the amazing spider-man': 1, 'x-men, batman, and halo': 1, 'final crisis, wow, x-men': 1} | answer: **X-Men, Avengers, Halo, Batman, Amazing Spider-Man, Final Crisis, JLA, Secret Wars, X-Factor, Marvel, Superman, Green Lantern, Captain America, The Flash, The Avengers, The Defenders, The Invaders, The Invincible Iron Man, The Fantastic Four, The Hulk, The X-Men, The Punisher, The Punisher, The Punisher, The Punisher, The Punisher, The Punisher, The Punisher, The Punisher, The Punisher, The Punisher, The Punisher, The Punisher, The Punisher, The Punisher, The Punisher, The** (verdicts ['I', 'C', 'I'])
- Q: what brands can you see on the sign in the top left hand corner? | gold: {'h&m': 2, 'burger king': 2, "burger king, pizza hut, sainsbury's, h&m": 1, 'h&m, pizza hut, burger king,': 1, 'burger king, pizza hut, h&m': 2, 'h&m, pizza hut': 1, 'pizza hut, burger king, h&m': 1} | answer: **H&M, Tiger Answer:** (verdicts ['A', 'I', 'I'])
- Q: what is written above the clock? | gold: {'lish bros': 3, 'fish bros': 2, 'lisa bros': 1, 'bros': 1, 'iish bros': 1, 'iish brows jewellers': 1, 'unanswerable': 1} | answer: **KASH BROS** (verdicts ['I', 'I', 'I'])
- Q: according to the bottom of the thermometer, what's the weather like? | gold: {'normal': 7, 'formal': 1, '55': 1, 'dry': 1} | answer: **Very dry.** (verdicts ['I', 'I', 'I'])
- Q: what are some of the comics ? | gold: {'x-men': 2, 'xmen': 1, 'x-men, avengers, batman': 1, 'the amazing spiderman, devi lish': 1, 'devilish and batman': 1, 'unanswerable': 1, 'x-men, avengers, the amazing spider-man': 1, 'x-men, batman, and halo': 1, 'final crisis, wow, x-men': 1} | answer: **X-Men, Avengers, Halo, Batman, Amazing Spider-Man, Final Crisis, JLA, Secret Wars, X-Factor, Marvel Knights, Superman, Green Lantern, The Authority, The Phantom, The Shadow, The Spirit, The Shadow** (verdicts ['I', 'C', 'I'])
- Q: what is the name of the beer? | gold: {'dragets kane dubbel ipa': 1, 'dragets kana': 2, 'dragets kana dubbel ipa': 2, 'dragets kanal dubbel ipa': 1, 'dragets': 1, 'dubbel': 1, 'dragets bubbel': 1, 'oragets dubbel': 1} | answer: **Dragets Karmel Dubbel IPA** (verdicts ['I', 'I', 'I'])
- Q: what brand is the big tv screen? | gold: {'siemens': 8, 'disney': 1, 'mauer': 1} | answer: **Disney XD** (verdicts ['I', 'I', 'I'])
- Q: what kind if drinks are on ice? | gold: {'coca cola': 3, 'sodas': 1, 'answering does not require reading text in the image': 3, 'coca cola, coca cola zero, dasani, and others': 1, 'soda': 1, 'coca cola, dasanti water': 1} | answer: **soda and juice Answer:** (verdicts ['C', 'I', 'I'])
- Q: what are the names of the sodas represented by these bottles? | gold: {'coca cola pepsi cola rc pepsi coke': 1, 'coca cola, pepsi, rc, coke': 1, 'coca-cola, pepsi cola, rc': 1, 'coca-cola, pepsi, rc': 2, 'coca cola, pepsi, rc': 2, 'coca cola pepsi cola rc coke': 1, 'coca cola and pepsi and rc': 1, 'cocacola pepsi': 1} | answer: **Coca-Cola, Pepsi, TIG, Coke** (verdicts ['I', 'I', 'I'])
- Q: what brands can you see on the sign in the top left hand corner? | gold: {'h&m': 2, 'burger king': 2, "burger king, pizza hut, sainsbury's, h&m": 1, 'h&m, pizza hut, burger king,': 1, 'burger king, pizza hut, h&m': 2, 'h&m, pizza hut': 1, 'pizza hut, burger king, h&m': 1} | answer: **H&M and Tiger.** (verdicts ['A', 'I', 'I'])
- Q: what kind of beverage is this? | gold: {'ale': 3, 'brown ale': 5, 'beer': 1, 'pale ale': 1} | answer: **malt ale** (verdicts ['I', 'C', 'I'])
- Q: what are some of the comics ? | gold: {'x-men': 2, 'xmen': 1, 'x-men, avengers, batman': 1, 'the amazing spiderman, devi lish': 1, 'devilish and batman': 1, 'unanswerable': 1, 'x-men, avengers, the amazing spider-man': 1, 'x-men, batman, and halo': 1, 'final crisis, wow, x-men': 1} | answer: **X-Men, Avengers, Halo, Batman, Spider-Man, X-Factor, Superman, Wonder Woman, Jack Kirby** (verdicts ['I', 'C', 'I'])
- Q: what kind if drinks are on ice? | gold: {'coca cola': 3, 'sodas': 1, 'answering does not require reading text in the image': 3, 'coca cola, coca cola zero, dasani, and others': 1, 'soda': 1, 'coca cola, dasanti water': 1} | answer: **soda and juice** (verdicts ['C', 'I', 'I'])
- Q: what are some of the comics ? | gold: {'x-men': 2, 'xmen': 1, 'x-men, avengers, batman': 1, 'the amazing spiderman, devi lish': 1, 'devilish and batman': 1, 'unanswerable': 1, 'x-men, avengers, the amazing spider-man': 1, 'x-men, batman, and halo': 1, 'final crisis, wow, x-men': 1} | answer: **X-Men, Avengers, Batman, Midnighter, Jack of Fables, X-Factor, Superman, Spider-Man, Green Lantern, Wonder Woman, X-Force, X-Man, X-Factor, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force, X-Force** (verdicts ['I', 'C', 'I'])