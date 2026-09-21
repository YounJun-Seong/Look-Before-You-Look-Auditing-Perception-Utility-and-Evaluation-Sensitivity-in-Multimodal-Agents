# Look Before You Look: Auditing Perception Utility and Evaluation Sensitivity in Multimodal Agents

Code, preregistrations, frozen artifacts, evaluation manifests, raw rollout
logs, and result files for the paper *Look Before You Look: Auditing
Perception Utility and Evaluation Sensitivity in Multimodal Agents*
(AAMAS 2027 submission). The supplementary document is
[`supplementary.pdf`](supplementary.pdf) at the repository root (source in
`paper/supplementary/`).

The audit asks, for a frozen vision-language reader under a fixed visual-token
budget, *when* additional perception (Crop / OCR / HighRes vs. Stop) actually
helps, separates unrestricted structural opportunity from action-matched
counterfactual opportunity and from realized prospective policy performance,
and shows how the scoring rule changes the apparent utility of adaptive
perception (with a blinded human adjudication of scorer disagreements).

## Repository layout

| Path | Contents |
|---|---|
| `ppu/` | The audit library (budget accounting, actions, regions, OCR, prompts, replays, evaluator, cache, runner, audit, features) |
| `scripts/` | Every collection driver, analysis, and figure script named in the result files |
| `tests/` | Offline test suite with stub readers and stub OCR (no GPU, no data): `python -m pytest tests -q` |
| `preregistrations/` | All 12 preregistration documents, verbatim (including the withdrawn original human-audit protocol) |
| `results/` | Machine-readable results (JSON), Markdown summaries, per-state CSVs, and the results digest that indexes them |
| `models/` | Freeze artifacts (pickle + JSON manifest) for every learned component; SHA-256 prefixes match the preregistrations |
| `manifests/` | Evaluation-set manifests (ids, questions, references, native sizes, relative image paths) |
| `logs/` | Gzipped raw rollout logs for every core collection (served question, views, token costs, raw generation, score, Stop log-probabilities) |
| `data/hidden/` | Mean-pooled Stop-time hidden states used by the representation-head controllers |
| `data/human_adjudication/` | Frozen blinded adjudication sample, the three annotators' verdict files, and stratum sizes |
| `paper/` | Supplementary LaTeX/PDF and figures |

Images are from the public TextVQA, VQAv2, and MME-RealWorld-Lite releases
and are not redistributed; manifests reference them by relative path.

## Reproduction

Offline (no GPU): every audit, policy comparison, rescoring, action-matched
oracle, and adjudication analysis is recomputable from `logs/`, `results/`,
`manifests/`, and `models/`; the evaluator is deterministic and raw
generations are preserved. Install `requirements.txt` (Python 3.10+) and run
the analysis scripts in `scripts/` (e.g. `scripts/deployable_oracle.py`,
`scripts/human_adjudication_analysis.py`,
`scripts/hidden_controller_eval.py`).

Full re-collection needs the two readers served with vLLM
(`scripts/serve.sh`, `scripts/serve_phi4.sh`; `--no-enable-prefix-caching`
is mandatory) and the public benchmark images. Set the `$PROJECT_ROOT`,
`$SERVE_VENV`, and `$ANALYSIS_PY` placeholders in the shell drivers to your
environment.

## Library layout


| Module | Role |
|---|---|
| `ppu/schema.py` | Frozen record schema, action definitions, cost ordering |
| `ppu/imaging.py` | Visual-token budget accounting (Qwen smart-resize), view rendering |
| `ppu/datasets.py` | Manifest loader, inclusion rule (native >= 4x coarse), seeded draws, burn list |
| `ppu/evaluator.py` | vqa_acc / ANLS / exact / MCQ scoring; answer extraction; **no LLM judge** |
| `ppu/vlm.py` | Frozen reader client (greedy, fixed seed) + StubVLM for tests |
| `ppu/ocr.py` | External OCR (RapidOCR, CPU); pre-pass and per-campaign caches with true-cost accounting |
| `ppu/regions.py` | Crop candidates: 5-box grid + up to 3 VLM proposals in the structural audit; 5 grid + 3 fixed fallback boxes (no proposals) in held-out/prospective collections; K=8 always; size-matched random boxes |
| `ppu/actions.py` | STOP / CROP / OCR / HIGHRES + dummy actions + full-view reference behind one interface |
| `ppu/prompts.py` | Versioned prompt templates (p0.1 and the p0.2-oneword intervention) |
| `ppu/replay.py` | R2 construction: MCQ option permutation / fixed syntactic rewrites |
| `ppu/replay_multi.py` | Replays 3–5 for the multi-replay robustness check |
| `ppu/cache.py` | SQLite generation cache, invalidated by prompt/decoding changes |
| `ppu/runner.py` | Brute-force campaign runner (no policy) |
| `ppu/audit.py` | Cross-replay oracle, luck floor, damage, preregistered verdicts |
| `ppu/gate_features.py` | The 16 frozen pre-action features (F0 / v2.2) |

`scripts/` contains every collection driver, analysis, and figure script
named in the result files (`collect_*.py` = GPU collection,
everything else = offline analysis over the logs). `tests/` exercises the
full pipeline with stub readers and stub OCR — no GPU, no data:

```
python -m pytest tests -q
```

## Reproduction

Offline (no GPU): every audit, policy comparison, and rescoring analysis is
recomputable from the released result files and manifests; the evaluator is
deterministic and raw generations are preserved.

Full re-collection:

1. Serve a reader with vLLM (`scripts/serve.sh` for Qwen2.5-VL-7B-Instruct,
   `scripts/serve_phi4.sh` for Phi-4-multimodal-instruct). Set
   `$SERVE_VENV` / `$PROJECT_ROOT` placeholders to your environment.
   `--no-enable-prefix-caching` is mandatory: the protocol issues many
   near-identical requests per image, and prefix caching would couple arms
   that must be independent.
2. Build manifests from the public datasets (`scripts/prepare_data.py`), or
   use the released manifests directly (relative image paths under
   `data/`).
3. Run collections (e.g. `scripts/collect_prospective.py`,
   `scripts/collect_seq_prospective.py`); each driver enforces a
   zero-transport-error integrity gate and STOP determinism checks.
4. Run the corresponding analysis scripts to regenerate the result files.

Scripts added in the 09-20/21 round: `strong_controller_common.py` (pooled
state loader), `train_strong_controllers.py` / `strong_controller_eval.py`
(F0 x ridge/xgb/mlp), `extract_hidden_states.py` /
`train_hidden_controllers.py` / `hidden_controller_eval.py` (3x3
representation-head test), `build_strong_manifest.py` /
`build_hidden_manifest.py`, `run_strong_controller.sh` /
`run_hidden_pipeline.sh` / `wait_gpu_and_run.sh` (drivers),
`replay_pairwise_robustness.py`, `cross_reader_conditional.py`,
`deployable_oracle.py` (action-matched oracle), and the human adjudication
package/analysis (`make_adjudication_package.py`, `make_adjudication_csv.py`,
`human_adjudication_analysis.py`). `$ANALYSIS_PY` denotes the analysis
interpreter (numpy 2, scikit-learn 1.7, xgboost 3.2); `$SERVE_VENV` the
vLLM/transformers environment.

Environment: Python 3.11+; analysis venv needs only
pillow / numpy / openai / rapidocr-onnxruntime / scikit-learn / xgboost / scipy / matplotlib.
Serving used vLLM 0.11.0, torch 2.8.0, CUDA 12.8.
