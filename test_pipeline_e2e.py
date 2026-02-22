#!/usr/bin/env python3
"""
test_pipeline_e2e.py
End-to-end smoke test for the Biomedical Semantic Leakage Detection pipeline.

Tests each component independently, then the full pipeline on 3 questions.
Prints PASS / FAIL / WARN for each check with real output — no mocked results.

Usage:
    python test_pipeline_e2e.py                  # full test (uses OpenRouter)
    python test_pipeline_e2e.py --heuristic-nli  # skip HuggingFace model download
    python test_pipeline_e2e.py --no-api         # offline only (skips LLM calls)
    python test_pipeline_e2e.py --model openai/gpt-4o-mini  # specific OpenRouter model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# ── Path setup ────────────────────────────────────────────────────────────────
_here = Path(__file__).parent.resolve()
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--heuristic-nli", action="store_true",
                    help="Skip transformer model download — use fast heuristic NLI")
parser.add_argument("--no-api", action="store_true",
                    help="Skip LLM API calls (test offline components only)")
parser.add_argument("--model", default=None,
                    help="OpenRouter model slug, e.g. 'openai/gpt-4o-mini'")
ARGS = parser.parse_args()

# Must be set BEFORE any imports that use it
if ARGS.heuristic_nli:
    os.environ["FORCE_HEURISTIC_NLI"] = "1"

# ── Test harness ──────────────────────────────────────────────────────────────
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

_results: List[Dict[str, Any]] = []

def check(name: str, condition: bool, detail: str = "", warn_only: bool = False):
    status = PASS if condition else (WARN if warn_only else FAIL)
    label  = "PASS" if condition else ("WARN" if warn_only else "FAIL")
    _results.append({"name": name, "status": label})
    print(f"  [{status}]  {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")

def section(title: str):
    bar = "─" * 64
    print(f"\n{bar}\n  {title}\n{bar}")

def summarise() -> bool:
    section("SUMMARY")
    passed = sum(1 for r in _results if r["status"] == "PASS")
    warned = sum(1 for r in _results if r["status"] == "WARN")
    failed = sum(1 for r in _results if r["status"] == "FAIL")
    total  = len(_results)
    print(f"  {PASS}: {passed}/{total}   {WARN}: {warned}   {FAIL}: {failed}")
    if failed:
        print(f"\n  Failed checks:")
        for r in _results:
            if r["status"] == "FAIL":
                print(f"    • {r['name']}")
    return failed == 0

# ── Test questions ─────────────────────────────────────────────────────────────
QUESTIONS = [
    "Does aspirin reduce the risk of myocardial infarction in patients with cardiovascular disease?",
    "What is the mechanism by which metformin lowers blood glucose in type 2 diabetes?",
    "How do statins reduce LDL cholesterol and cardiovascular risk?",
]

# Fixed steps for offline component tests — no API needed
KNOWN_STEPS = [
    "Aspirin inhibits COX-1 and COX-2 enzymes, reducing thromboxane A2 synthesis.",
    "Reduced thromboxane A2 leads to decreased platelet aggregation.",
    "This antiplatelet effect lowers the risk of myocardial infarction.",
]

# =============================================================================
# SECTION 1 — Imports
# =============================================================================
section("1. Module Imports")

_imports_ok = {}

for mod, sym in [
    ("utils.cot_generator",    "generate"),
    ("utils.concept_extractor","extract_concepts"),
    ("utils.hybrid_checker",   "build_entailment_records"),
    ("utils.guards",           "derive_guards"),
    ("utils.umls_api_linker",  "is_configured"),
]:
    try:
        parts = mod.split(".")
        m = __import__(mod, fromlist=[sym])
        _imports_ok[mod] = True
        check(f"import {mod}", True)
    except Exception as e:
        _imports_ok[mod] = False
        check(f"import {mod}", False, str(e))

# Provider readiness
if _imports_ok.get("utils.cot_generator"):
    from utils.cot_generator import generate as generate_cot, OPENROUTER_READY, ANTHROPIC_READY
    any_provider = OPENROUTER_READY or ANTHROPIC_READY
    check("At least one LLM provider configured",
          any_provider or ARGS.no_api,
          f"OPENROUTER_READY={OPENROUTER_READY}  ANTHROPIC_READY={ANTHROPIC_READY}\n"
          f"  → Set OPENROUTER_API_KEY env var or edit config.py",
          warn_only=True)

if _imports_ok.get("utils.concept_extractor"):
    from utils.concept_extractor import extract_concepts

if _imports_ok.get("utils.hybrid_checker"):
    from utils.hybrid_checker import build_entailment_records

if _imports_ok.get("utils.guards"):
    from utils.guards import derive_guards, GuardConfig, lexical_jaccard
    GUARD_CFG = GuardConfig()

if _imports_ok.get("utils.umls_api_linker"):
    from utils.umls_api_linker import is_configured as umls_configured
    umls_ok = umls_configured()
    check("UMLS API configured",
          umls_ok,
          "UMLS_API_KEY not set — concept candidates will have no CUI linking\n"
          "  → Set UMLS_API_KEY env var (free at https://uts.nlm.nih.gov/uts/signup-login)",
          warn_only=True)

# =============================================================================
# SECTION 2 — Guard Signals (fully offline, deterministic)
# =============================================================================
section("2. Guard Signals (offline — deterministic)")

if not _imports_ok.get("utils.guards"):
    check("guard tests", False, "skipped — import failed")
else:
    # 2a. Lexical duplicate detection
    s1 = "Aspirin inhibits COX enzymes and reduces platelet aggregation."
    s2 = "Aspirin inhibits COX enzymes and reduces platelet aggregation."
    s3 = "Metformin activates AMPK in the liver to suppress gluconeogenesis."

    j_same = lexical_jaccard(s1, s2)
    j_diff = lexical_jaccard(s1, s3)
    check("lexical_jaccard: identical sentences → 1.0",
          abs(j_same - 1.0) < 1e-6, f"got {j_same:.4f}")
    check("lexical_jaccard: different sentences < 0.4",
          j_diff < 0.4, f"got {j_diff:.4f}")

    # 2b. Duplicate step → lexical_duplicate guard
    probs_dup = {"entailment": 0.85, "neutral": 0.10, "contradiction": 0.05}
    g_dup = derive_guards(premise=s1, hypothesis=s2, probs=probs_dup, config=GUARD_CFG)
    check("identical steps → 'lexical_duplicate' guard fires",
          "lexical_duplicate" in g_dup, f"guards={g_dup}")

    # 2c. Near-contradiction (close margins) → caution_band
    probs_close = {"entailment": 0.38, "neutral": 0.30, "contradiction": 0.32}
    g_close = derive_guards(premise=s1, hypothesis=s3, probs=probs_close, config=GUARD_CFG)
    check("near-contradiction (close margins) → 'caution_band' guard fires",
          "caution_band" in g_close, f"guards={g_close}")

    # 2d. Clear contradiction — guards are about uncertainty, NOT about detecting contradiction.
    # When p_contra=0.85 there is NO ambiguity so no guard should fire. final_label handles it.
    probs_strong = {"entailment": 0.05, "neutral": 0.10, "contradiction": 0.85}
    g_strong = derive_guards(
        premise="Aspirin reduces platelet aggregation.",
        hypothesis="Aspirin increases platelet aggregation.",
        probs=probs_strong, config=GUARD_CFG
    )
    check("clear contradiction (p=0.85) → no uncertainty guards (correct)",
          "caution_band" not in g_strong,
          f"guards={g_strong}  (guards flag uncertainty, not certainty)")

    print(f"\n         Guard results on test pairs:")
    print(f"           duplicate pair      → {g_dup}")
    print(f"           borderline pair     → {g_close}")
    print(f"           clear contra pair   → {g_strong}")

# =============================================================================
# SECTION 3 — Concept Extractor (surface generation, offline)
# =============================================================================
section("3. Concept Extractor")

concepts = [[] for _ in KNOWN_STEPS]

if not _imports_ok.get("utils.concept_extractor"):
    check("concept extractor tests", False, "skipped — import failed")
else:
    try:
        concepts = extract_concepts(KNOWN_STEPS, scispacy_when="never", top_k=5)
        check("returns list of length == steps",
              len(concepts) == len(KNOWN_STEPS),
              f"expected {len(KNOWN_STEPS)}, got {len(concepts)}")

        n_cands = [len(c) for c in concepts]
        has_any = any(n > 0 for n in n_cands)
        check("at least some concepts extracted",
              has_any,
              f"candidates per step: {n_cands}\n"
              f"  → If all zeros, UMLS may not be configured (surface candidates still expected)",
              warn_only=not has_any)

        if has_any:
            for i, (step, cands) in enumerate(zip(KNOWN_STEPS, concepts)):
                names = [c.get("name") or c.get("surface", "?") for c in cands[:4]]
                print(f"         step[{i}]: {names}")
        else:
            print("         No candidates returned — UMLS linking likely disabled")
            print("         Surface n-gram generation may also be inactive without UMLS")

    except Exception as e:
        check("extract_concepts runs without error", False, traceback.format_exc())

# =============================================================================
# SECTION 4 — Hybrid NLI Checker
# =============================================================================
section("4. Hybrid NLI Entailment Checker" +
        (" [heuristic mode]" if ARGS.heuristic_nli else " [transformer mode]"))

pairs = []
if not _imports_ok.get("utils.hybrid_checker"):
    check("NLI tests", False, "skipped — import failed")
else:
    try:
        pairs = build_entailment_records(KNOWN_STEPS, concepts)
        expected = len(KNOWN_STEPS) - 1

        check("returns N-1 adjacent pairs",
              len(pairs) == expected,
              f"expected {expected}, got {len(pairs)}")

        all_probs_valid = all(
            abs(sum(p.get("probs", {}).values()) - 1.0) < 0.05
            for p in pairs
        )
        check("all pair probabilities sum to ≈ 1.0", all_probs_valid)

        valid_labels = {"entailment", "neutral", "contradiction"}
        check("all pairs have valid final_label",
              all(p.get("final_label") in valid_labels for p in pairs),
              f"labels found: {[p.get('final_label') for p in pairs]}")

        print(f"\n         NLI results (heuristic={ARGS.heuristic_nli}):")
        for p in pairs:
            i, j = p["step_pair"]
            probs = p.get("probs", {})
            src   = (p.get("meta") or {}).get("nli_source", "?")
            print(f"           pair({i}→{j})  {p.get('final_label'):15s}  "
                  f"E={probs.get('entailment',0):.3f}  "
                  f"N={probs.get('neutral',0):.3f}  "
                  f"C={probs.get('contradiction',0):.3f}  "
                  f"src={src}")

    except Exception as e:
        check("build_entailment_records runs without error", False, traceback.format_exc())

# =============================================================================
# SECTION 5 — CoT Generator (live LLM call)
# =============================================================================
section("5. CoT Generator — live LLM call" +
        (" [SKIPPED --no-api]" if ARGS.no_api else ""))

cot_steps = None
if not ARGS.no_api:
    if not _imports_ok.get("utils.cot_generator"):
        check("CoT generator", False, "skipped — import failed")
    else:
        q = QUESTIONS[0]
        try:
            t0 = time.time()
            result = generate_cot(q, prefer="openrouter", model=ARGS.model)
            elapsed = round(time.time() - t0, 2)

            provider = result.get("provider", "?")
            model_id = result.get("model", "?")
            cot_steps = result.get("steps", [])

            check("returns ≥ 3 steps",
                  len(cot_steps) >= 3,
                  f"provider={provider}  model={model_id}  "
                  f"n_steps={len(cot_steps)}  elapsed={elapsed}s")
            check("provider is not 'local' (real LLM responded)",
                  provider != "local",
                  f"provider='{provider}' — 'local' means all API calls failed\n"
                  f"  → Check your OPENROUTER_API_KEY",
                  warn_only=(provider == "local"))
            check("steps are non-trivial strings (len > 15 chars each)",
                  all(len(s) > 15 for s in cot_steps),
                  f"step lengths: {[len(s) for s in cot_steps]}")

            print(f"\n         Question : {q[:70]}")
            print(f"         Provider : {provider}  ({model_id})  [{elapsed}s]")
            for i, s in enumerate(cot_steps[:6]):
                print(f"         Step {i+1}   : {s[:88]}")
            if len(cot_steps) > 6:
                print(f"         ... ({len(cot_steps) - 6} more)")

        except Exception as e:
            check("generate_cot runs without error", False, traceback.format_exc())
else:
    check("CoT generator (skipped)", True, "--no-api flag set", warn_only=True)

# =============================================================================
# SECTION 6 — Full Pipeline End-to-End (3 questions)
# =============================================================================
section("6. Full Pipeline End-to-End" +
        (" [SKIPPED --no-api]" if ARGS.no_api else ""))

e2e_results = []

if not ARGS.no_api:
    all_imports_ready = all(_imports_ok.get(m) for m in [
        "utils.cot_generator", "utils.concept_extractor",
        "utils.hybrid_checker", "utils.guards"
    ])

    if not all_imports_ready:
        check("Full pipeline", False, "skipped — one or more imports failed")
    else:
        for qi, question in enumerate(QUESTIONS):
            print(f"\n  ── Q{qi+1}: {question[:65]}...")
            rec: Dict[str, Any] = {"question": question, "ok": False}
            try:
                t0 = time.time()

                # Step 1: CoT
                cot = generate_cot(question, prefer="openrouter", model=ARGS.model)
                steps = cot.get("steps", [])
                rec["provider"] = cot.get("provider", "?")
                rec["model"]    = cot.get("model", "?")
                rec["n_steps"]  = len(steps)
                print(f"     CoT    : {len(steps)} steps  "
                      f"provider={rec['provider']}  model={rec['model']}")

                if len(steps) < 2:
                    print("     ERROR  : Too few steps — skipping NLI")
                    e2e_results.append(rec)
                    continue

                # Step 2: Concepts
                step_concepts = extract_concepts(steps, scispacy_when="never", top_k=3)
                n_cands = sum(len(c) for c in step_concepts)
                rec["n_concepts"] = n_cands
                print(f"     UMLS   : {n_cands} candidates across {len(steps)} steps")

                # Step 3: NLI
                nli_pairs = build_entailment_records(steps, step_concepts)
                label_counts = Counter(p.get("final_label", "?") for p in nli_pairs)
                rec["pairs"]        = len(nli_pairs)
                rec["label_counts"] = dict(label_counts)
                print(f"     NLI    : {len(nli_pairs)} pairs → {dict(label_counts)}")

                # Step 4: Guards
                all_guards: List[str] = []
                for p in nli_pairs:
                    i_idx, j_idx = p["step_pair"]
                    g = derive_guards(
                        premise    = steps[i_idx] if i_idx < len(steps) else "",
                        hypothesis = steps[j_idx] if j_idx < len(steps) else "",
                        probs      = p["probs"],
                        config     = GUARD_CFG,
                    )
                    all_guards.extend(g)
                rec["guards"] = dict(Counter(all_guards))
                print(f"     Guards : {rec['guards'] or 'none'}")

                rec["elapsed_s"] = round(time.time() - t0, 2)
                rec["ok"] = len(steps) >= 3 and len(nli_pairs) > 0
                print(f"     Status : {'OK' if rec['ok'] else 'INCOMPLETE'}  "
                      f"({rec['elapsed_s']}s)")

            except Exception as e:
                print(f"     ERROR  : {e}")
                traceback.print_exc()
                rec["error"] = str(e)

            e2e_results.append(rec)
            time.sleep(0.5)

        n_ok = sum(1 for r in e2e_results if r.get("ok"))
        check(f"All {len(QUESTIONS)} questions completed end-to-end",
              n_ok == len(QUESTIONS),
              f"{n_ok}/{len(QUESTIONS)} succeeded")

        # Save results
        out_path = _here / "e2e_test_results.json"
        with open(out_path, "w") as f:
            json.dump(e2e_results, f, indent=2, ensure_ascii=False)
        check("Results saved to e2e_test_results.json",
              out_path.exists(), str(out_path))

        # Print JSON summary
        print(f"\n  Results saved → {out_path}")
        for r in e2e_results:
            print(f"    {r.get('provider','?'):12s} | steps={r.get('n_steps','?'):2} | "
                  f"pairs={r.get('pairs','?'):2} | "
                  f"contradiction={r.get('label_counts',{}).get('contradiction',0):2} | "
                  f"guards={r.get('guards',{})} | "
                  f"ok={r.get('ok')}")
else:
    check("Full pipeline (skipped)", True, "--no-api flag set", warn_only=True)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
all_passed = summarise()
sys.exit(0 if all_passed else 1)
