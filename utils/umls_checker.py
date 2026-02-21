# utils/umls_checker.py — Validate concepts and relations using UMLS ontologies
from __future__ import annotations
from typing import List, Dict, Any, Iterable, Optional, Sequence
from dataclasses import dataclass

# Define type buckets for coarse relation compatibility (TUI buckets)
TUI_BUCKETS = {
    # Example mapping of Semantic Type TUIs to broad categories (for demonstration)
    "T047": "Disease",   # e.g., Disease or Syndrome
    "T121": "Pharmaco",  # e.g., Pharmacologic Substance
    "T129": "Pharmaco",  # e.g., Immunologic Factor (also treat as Pharmaco)
    "T195": "Pharmaco",  # e.g., Antibiotic
    "T200": "ClinicalDrug",
    # ... (other mappings as needed)
}
# Pairs of allowed relations between categories
ALLOWED_RELATIONS = {
    ("Pharmaco", "Disease"): "treats",
    ("ClinicalDrug", "Disease"): "treats",
    ("Pharmaco", "Pharmaco"): "interacts",
    # ... (other allowed type relations)
}
SYMMETRIC_KEYS = {("Pharmaco", "Pharmaco")}  # treat this pair as symmetric relation

@dataclass
class CheckerConfig:
    allowed_sources: Iterable[str] = ()
    main_sources: Iterable[str] = ()
    secondary_sources: Iterable[str] = ()
    allowed_tuis: Iterable[str] = ()
    min_score: float = 0.0
    enable_relation_check: bool = False
    require_main_source: bool = False
    ban_generic: bool = False
    upgrade_bioprocess_terms: bool = False
    allow_missing_score: bool = True

class ConceptRecord:
    """Internal wrapper for a concept dict for easier validation."""
    def __init__(self, cdict: Dict[str, Any]):
        self.text = cdict.get("text", "")
        self.cui = cdict.get("cui", "").strip()
        self.canonical = cdict.get("canonical", "")
        self.semantic_types = [d.get("name") for d in (cdict.get("semantic_types") or [])]
        self.kb_sources = [s for s in (cdict.get("kb_sources") or [])]
        self.score = None
        # If scores present, consider 'confidence' or 'api' as primary
        scores = cdict.get("scores") or {}
        if "confidence" in scores:
            self.score = float(scores["confidence"])
        elif "api" in scores:
            self.score = float(scores["api"])
        self.valid = False
        self.reasons: List[str] = []

class UMLSChecker:
    def __init__(self, config: CheckerConfig = CheckerConfig()):
        self.cfg = config

    def validate_concept(self, cdict: Dict[str, Any]) -> Dict[str, Any]:
        c = ConceptRecord(cdict)
        reasons: List[str] = []
        # Allowed source check
        src_ok = True
        if self.cfg.allowed_sources:
            if not any(src.upper() in self.cfg.allowed_sources for src in c.kb_sources):
                src_ok = False
                reasons.append("source not allowed")
        # Semantic type filter
        allowed_tui = True
        if self.cfg.allowed_tuis:
            tuis = [t for t in c.semantic_types if t in TUI_BUCKETS]  # treat semantic_types as TUI codes if matching
            if not tuis or not any(t in self.cfg.allowed_tuis for t in tuis):
                allowed_tui = False
                reasons.append("semantic type not allowed")
        # Main source requirement
        if self.cfg.require_main_source:
            main_ok = any(src.upper() in (s.upper() for s in self.cfg.main_sources) for src in c.kb_sources)
            if not main_ok:
                reasons.append("missing main source")
        # Generic term ban (optional simplistic check via text length or certain words)
        if self.cfg.ban_generic and len(c.text.split()) <= 1:
            reasons.append("generic/uninformative span")

        # Bioprocess upgrade (not elaborated here)
        if self.cfg.upgrade_bioprocess_terms and "generic/uninformative span" in reasons:
            # If a term is a known bioprocess, remove that reason (example logic)
            reasons = [r for r in reasons if r != "generic/uninformative span"]

        # Score threshold check
        score_ok = (c.score is None and self.cfg.allow_missing_score) or (c.score is not None and c.score >= self.cfg.min_score)

        c.valid = bool(score_ok and src_ok and allowed_tui and c.cui)
        cdict_out = {**cdict}
        cdict_out["valid"] = c.valid
        cdict_out["reasons"] = sorted(set(reasons)) if not c.valid else [f"ok{''}"]
        return cdict_out

    def validate_step_concepts(self, step_concepts: Iterable[Iterable[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        out: List[List[Dict[str, Any]]] = []
        for group in step_concepts:
            validated = [self.validate_concept(c) for c in (group or [])]
            out.append(validated)
        return out

    def _best_bucket(self, concept: Dict[str, Any]) -> Optional[str]:
        # Determine broad bucket for a concept's semantic types
        stypes = concept.get("semantic_types") or []
        tuis = [obj["name"] for obj in stypes if obj.get("name")]
        for t in tuis:
            if t in TUI_BUCKETS:
                return TUI_BUCKETS[t]
        return None

    def _pair_allowed(self, a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        ba = self._best_bucket(a)
        bb = self._best_bucket(b)
        if not ba or not bb:
            return False, None
        verb = ALLOWED_RELATIONS.get((ba, bb))
        if verb:
            return True, verb
        # symmetric support for selected pairs
        if (bb, ba) in ALLOWED_RELATIONS and (ba, bb) in SYMMETRIC_KEYS:
            return True, ALLOWED_RELATIONS[(bb, ba)]
        return False, None

    def validate_relations_adjacent(self, step_concepts: Sequence[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not self.cfg.enable_relation_check:
            return []
        diagnostics: List[Dict[str, Any]] = []
        for s in range(len(step_concepts) - 1):
            left = step_concepts[s] or []
            right = step_concepts[s + 1] or []
            for ca in left:
                if not ca.get("valid"):
                    continue
                for cb in right:
                    if not cb.get("valid"):
                        continue
                    ok, verb = self._pair_allowed(ca, cb)
                    diagnostics.append({
                        "i": s, "j": s + 1, "a": ca, "b": cb,
                        "allowed": bool(ok),
                        "verb": (verb if ok else None),
                        "reason": (f"type-compatible: {verb}" if ok else "no supported relation between types"),
                    })
        return diagnostics

# Convenience top-level functions
_DEFAULT_CHECKER = UMLSChecker()

def validate_concepts(per_step_concepts: Sequence[Sequence[Dict[str, Any]]],
                      checker: Optional[UMLSChecker] = None) -> List[List[Dict[str, Any]]]:
    ch = checker or _DEFAULT_CHECKER
    return ch.validate_step_concepts(per_step_concepts)

# Back-compat alias
def validate_step_concepts(per_step_concepts: Sequence[Sequence[Dict[str, Any]]],
                           checker: Optional[UMLSChecker] = None) -> List[List[Dict[str, Any]]]:
    return validate_concepts(per_step_concepts, checker)

def validate_relations(per_step_concepts: Sequence[Sequence[Dict[str, Any]]],
                       checker: Optional[UMLSChecker] = None) -> List[Dict[str, Any]]:
    ch = checker or _DEFAULT_CHECKER
    return ch.validate_relations_adjacent(per_step_concepts)

def make_checker(
    allowed_sources: Optional[Iterable[str]] = None,
    main_sources: Optional[Iterable[str]] = None,
    secondary_sources: Optional[Iterable[str]] = None,
    allowed_tuis: Optional[Iterable[str]] = None,
    min_score: Optional[float] = None,
    enable_relation_check: Optional[bool] = None,
    require_main_source: Optional[bool] = None,
    ban_generic: Optional[bool] = None,
    upgrade_bioprocess_terms: Optional[bool] = None,
) -> UMLSChecker:
    cfg = CheckerConfig()
    if allowed_sources is not None:
        cfg.allowed_sources = {s.upper() for s in allowed_sources}
    if main_sources is not None:
        cfg.main_sources = {s.upper() for s in main_sources}
    if secondary_sources is not None:
        cfg.secondary_sources = {s.upper() for s in secondary_sources}
    if allowed_tuis is not None:
        cfg.allowed_tuis = {str(t).upper() for t in allowed_tuis}
    if min_score is not None:
        cfg.min_score = float(min_score)
    if enable_relation_check is not None:
        cfg.enable_relation_check = bool(enable_relation_check)
    if require_main_source is not None:
        cfg.require_main_source = bool(require_main_source)
    if ban_generic is not None:
        cfg.ban_generic = bool(ban_generic)
    if upgrade_bioprocess_terms is not None:
        cfg.upgrade_bioprocess_terms = bool(upgrade_bioprocess_terms)
    return UMLSChecker(cfg)

# Relation helper shims (compatibility with older code)
def _best_bucket_for_concept(concept: Dict[str, Any]) -> Optional[str]:
    stypes = concept.get("semantic_types") or []
    tuis: List[str] = []
    for st in stypes:
        if isinstance(st, dict):
            t = st.get("name")
        else:
            t = str(st)
        if t and t in TUI_BUCKETS:
            return TUI_BUCKETS[t]
    return None

def has_supported_relation(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """
    Back-compat helper used in older code.
    Returns True if the coarse type buckets of (a -> b) are supported.
    """
    ba = _best_bucket_for_concept(a)
    bb = _best_bucket_for_concept(b)
    if not ba or not bb:
        return False
    if (ba, bb) in ALLOWED_RELATIONS:
        return True
    if (bb, ba) in ALLOWED_RELATIONS and (ba, bb) in SYMMETRIC_KEYS:
        return True
    return False

# (Optional) provisional_support placeholder for older code:
def provisional_support(concept_a: Dict[str, Any], concept_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Back-compat stub for older provisional support check.
    This could implement a check if concept_a and concept_b appear together in known knowledge (not implemented here).
    """
    return {"allowed": False, "evidence": []}
