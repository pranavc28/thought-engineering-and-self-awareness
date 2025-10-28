"""SciFact Retrieval: NAIVE vs OVERTHINKING vs POST-HOC"""

import os, json, random, re
from typing import Dict, List, Tuple
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import matplotlib.pyplot as plt

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_client = None
_corpus = None
_debug_log_list = None

def get_client():
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("Set OPENAI_API_KEY")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

def log_debug(msg: str):
    if _debug_log_list is not None:
        _debug_log_list.append(msg)

def init_worker(debug_list):
    global _debug_log_list
    _debug_log_list = debug_list

LABELS = {"SUPPORT", "CONTRADICT", "NOINFO"}
SAMPLES = 200
MAX_WORKERS = 200
MODELS = ["o3", "gpt-5"]

THRESHOLDS = {
    "gpt-5": {
        "classification": 0.75,
        "posthoc_refinement": 0.7
    },
    "o3": {
        "classification": 0.75,
        "posthoc_refinement": 0.7
    }
}
DEFAULT_THRESHOLDS = {"classification": 0.55, "posthoc_refinement": 0.75}

def load_corpus() -> Dict:
    global _corpus
    if _corpus is not None:
        return _corpus
    _corpus = {}
    try:
        with open('data/corpus.jsonl', 'r') as f:
            for line in f:
                doc = json.loads(line)
                doc_id = str(doc['doc_id'])
                abstract_text = ' '.join(doc.get('abstract', [])) if isinstance(doc.get('abstract', []), list) else doc.get('abstract', '')
                _corpus[doc_id] = {'doc_id': doc_id, 'title': doc.get('title', ''), 'abstract': abstract_text}
    except FileNotFoundError:
        pass
    return _corpus

def normalize_label(x: str) -> str:
    x = (x or "").upper().strip()
    return x if x in LABELS else "NOINFO"

def corpus_search(query: str, top_k: int = 5) -> List[Dict]:
    corpus = load_corpus()
    if not corpus:
        return []
    query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
    scores = []
    for doc_id, doc in corpus.items():
        combined = (doc['title'] + " " + doc['abstract']).lower()
        doc_terms = set(re.findall(r'\b\w{4,}\b', combined))
        overlap = len(query_terms.intersection(doc_terms))
        if overlap > 0:
            title_overlap = len(query_terms.intersection(set(re.findall(r'\b\w{4,}\b', doc['title'].lower()))))
            scores.append((overlap + title_overlap * 2, doc))
    scores.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scores[:top_k]]

def build_naive_evidence(claim: str, model: str) -> Tuple[str, List[Dict]]:
    prompt = (
        f"Generate 2-3 scientific search queries to find research papers that could verify or refute this claim.\n\n"
        f"Claim: {claim}\n\n"
        f"Focus on:\n"
        f"- Key entities (genes, proteins, drugs, diseases)\n"
        f"- Core mechanisms or relationships mentioned\n"
        f"- Specific phenomena or effects\n\n"
        f"Return JSON: {{\"search_queries\": [\"query1\", \"query2\", \"query3\"]}}"
    )
    try:
        resp = get_client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        queries = json.loads(resp.choices[0].message.content or "{}").get("search_queries", [])[:3]
        if not queries:
            queries = [claim]
        
        all_papers, seen_ids = [], set()
        for query in queries:
            for p in corpus_search(query, top_k=5):
                if p["doc_id"] not in seen_ids:
                    seen_ids.add(p["doc_id"])
                    all_papers.append(p)
        
        evidence = "\n\n".join([f"Paper {i}: {p['title']}\n{p['abstract']}" 
                               for i, p in enumerate(all_papers[:10], 1)])
        return evidence or "No papers found.", all_papers
    except Exception as e:
        log_debug(f"NAIVE_ERROR | {str(e)}")
        return "Error", []

def build_posthoc_evidence(claim: str, model: str) -> Tuple[str, List[Dict], Dict]:
    """Returns (evidence, papers, posthoc_metadata)"""
    try:
        # Initial retrieval
        resp = get_client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": f"Generate 2-3 search queries for: {claim}\nReturn JSON: {{\"search_queries\": [...]}}"}],
            response_format={"type": "json_object"}
        )
        queries = json.loads(resp.choices[0].message.content or "{}").get("search_queries", [])[:3]
        if not queries:
            queries = [claim]
        
        all_papers, seen_ids = [], set()
        for query in queries:
            for p in corpus_search(query, top_k=5):
                if p["doc_id"] not in seen_ids:
                    seen_ids.add(p["doc_id"])
                    all_papers.append(p)
        
        # Assess quality
        preview = "\n\n".join([f"Paper {i}: {p['title']}\n{p['abstract'][:500]}" 
                              for i, p in enumerate(all_papers[:10], 1)])
        assess_prompt = (
            f"You retrieved these papers for the claim. Assess if they provide sufficient evidence.\n\n"
            f"Claim: {claim}\n\n"
            f"Retrieved Papers:\n{preview}\n\n"
            f"Evaluate:\n"
            f"1. Confidence these papers can verify/refute the claim (0.0 = need more, 1.0 = sufficient)\n"
            f"2. What critical information is MISSING to properly assess the claim?\n"
            f"3. Should additional papers be retrieved? (yes only if confidence < 0.7)\n\n"
            f"Return JSON: {{\"confidence\": 0.0-1.0, \"need_refinement\": \"yes\"|\"no\", \"gaps\": \"what's missing\"}}"
        )
        
        resp2 = get_client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": assess_prompt}],
            response_format={"type": "json_object"}
        )
        assessment = json.loads(resp2.choices[0].message.content or "{}")
        confidence = float(assessment.get("confidence", 0.5))
        need_refinement = assessment.get("need_refinement", "no").lower() == "yes"
        
        # Store posthoc metadata for optimization
        posthoc_metadata = {
            "initial_confidence": confidence,
            "need_refinement": need_refinement,
            "refinement_triggered": False
        }
        
        threshold_config = THRESHOLDS.get(model, DEFAULT_THRESHOLDS)
        posthoc_threshold = threshold_config["posthoc_refinement"]
        
        if need_refinement and confidence < posthoc_threshold and len(all_papers) < 8:
            posthoc_metadata["refinement_triggered"] = True
            gaps = assessment.get("gaps", "")
            refine_prompt = (
                f"Initial retrieval was insufficient. Missing information: {gaps}\n\n"
                f"Claim: {claim}\n\n"
                f"Generate 1-2 NEW refined search queries to fill these specific gaps.\n"
                f"Focus on what's missing, not what you already retrieved.\n\n"
                f"Return JSON: {{\"refined_queries\": [\"query1\", \"query2\"]}}"
            )
            resp3 = get_client().chat.completions.create(
                model=model, messages=[{"role": "user", "content": refine_prompt}],
                response_format={"type": "json_object"}
            )
            refined_queries = json.loads(resp3.choices[0].message.content or "{}").get("refined_queries", [])[:2]
            
            for query in refined_queries:
                for p in corpus_search(query, top_k=5):
                    if p["doc_id"] not in seen_ids:
                        seen_ids.add(p["doc_id"])
                        all_papers.append(p)
        
        evidence = "\n\n".join([f"Paper {i}: {p['title']}\n{p['abstract']}" 
                               for i, p in enumerate(all_papers[:10], 1)])
        return evidence or "No papers found.", all_papers, posthoc_metadata
    except Exception as e:
        log_debug(f"POSTHOC_ERROR | {str(e)}")
        return "Error", [], {"initial_confidence": 0.0, "need_refinement": False, "refinement_triggered": False}

def build_overthinking_evidence(claim: str, model: str) -> Tuple[str, List[Dict], str]:
    prompt = (
        f"You're a scientific claim verifier. Before searching, analyze this claim.\n\n"
        f"Claim: {claim}\n\n"
        f"Think step-by-step:\n"
        f"1. What is this claim asserting? What are the key concepts?\n"
        f"2. What evidence would SUPPORT this? What would CONTRADICT it?\n"
        f"3. Initial confidence in finding relevant evidence (0.0-1.0)\n"
        f"4. Search strategy:\n"
        f"   - If confidence < 0.5: Generate 3-4 broad exploratory queries\n"
        f"   - If confidence 0.5-0.8: Generate 2-3 targeted queries\n"
        f"   - If confidence > 0.8: Generate 1-2 precise queries\n\n"
        f"Return JSON: {{\"reasoning\": \"your analysis\", \"initial_confidence\": 0.0-1.0, \"search_queries\": [...]}}"
    )
    try:
        resp = get_client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        queries = parsed.get("search_queries", [])[:5]
        reasoning = f"Confidence: {parsed.get('initial_confidence', 0.5):.2f}\n{parsed.get('reasoning', '')}"
        
        if not queries:
            queries = [claim]
        
        all_papers, seen_ids = [], set()
        for query in queries:
            for p in corpus_search(query, top_k=5):
                if p["doc_id"] not in seen_ids:
                    seen_ids.add(p["doc_id"])
                    all_papers.append(p)
        
        evidence = "\n\n".join([f"Paper {i}: {p['title']}\n{p['abstract']}" 
                               for i, p in enumerate(all_papers[:10], 1)])
        return evidence or "No papers found.", all_papers, reasoning
    except Exception as e:
        log_debug(f"OVERTHINKING_ERROR | {str(e)}")
        return "Error", [], "Error"

def classify_with_evidence(claim: str, evidence: str, model: str) -> Dict:
    prompt = (
        f"Verify this scientific claim using the provided evidence.\n\n"
        f"Claim: {claim}\n\n"
        f"Evidence from papers:\n{evidence}\n\n"
        f"Classify as:\n"
        f"- SUPPORT: The papers provide evidence that confirms the claim (experimental data, results, or findings that align with it)\n"
        f"- CONTRADICT: The papers provide evidence that refutes the claim (experimental data, results, or findings that oppose it)\n"
        f"- NOINFO: The papers don't contain relevant information about this specific claim\n\n"
        f"Note: If the papers discuss the relevant concepts and present data/results, classify as SUPPORT or CONTRADICT based on whether the evidence aligns or opposes the claim.\n\n"
        f"Return JSON: {{\"label\": \"SUPPORT\"|\"CONTRADICT\"|\"NOINFO\", \"confidence\": 0.0-1.0, \"justification\": \"explanation\"}}"
    )
    try:
        resp = get_client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content)
        result["label"] = normalize_label(result.get("label", "NOINFO"))
        result["confidence"] = float(result.get("confidence", 0.5))
        
        # Apply threshold (currently commented out - user will en
        # able after optimization)
        threshold_config = THRESHOLDS.get(model, DEFAULT_THRESHOLDS)
        if result["confidence"] < threshold_config["classification"] and result["label"] != "NOINFO":
            result["label"] = "NOINFO"
        
        return result
    except Exception as e:
        return {"label": "NOINFO", "confidence": 0.0, "justification": "error"}

def process_claim(row: Dict, model: str) -> Dict:
    claim = row["text"]
    gold = normalize_label(row["label_text"])
    
    naive_evidence, naive_papers = build_naive_evidence(claim, model)
    naive_result = classify_with_evidence(claim, naive_evidence, model)
    
    overthinking_evidence, overthinking_papers, overthinking_reasoning = build_overthinking_evidence(claim, model)
    overthinking_result = classify_with_evidence(claim, overthinking_evidence, model)
    
    posthoc_evidence, posthoc_papers, posthoc_metadata = build_posthoc_evidence(claim, model)
    posthoc_result = classify_with_evidence(claim, posthoc_evidence, model)
    
    return {
        "claim_id": row.get("id", ""),
        "claim": claim,
        "gold": gold,
        "naive": {
            "label": normalize_label(naive_result.get("label", "")),
            "result": naive_result,
            "doc_ids": [p["doc_id"] for p in naive_papers[:5]]
        },
        "overthinking": {
            "label": normalize_label(overthinking_result.get("label", "")),
            "result": overthinking_result,
            "doc_ids": [p["doc_id"] for p in overthinking_papers[:5]],
            "reasoning": overthinking_reasoning[:200]
        },
        "posthoc": {
            "label": normalize_label(posthoc_result.get("label", "")),
            "result": posthoc_result,
            "doc_ids": [p["doc_id"] for p in posthoc_papers[:5]],
            "metadata": posthoc_metadata
        }
    }

def process_batch_parallel(rows: List[Dict], model: str, debug_list) -> List[Dict]:
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(debug_list,)) as executor:
        future_to_row = {executor.submit(process_claim, row, model): row for row in rows}
        for i, future in enumerate(as_completed(future_to_row), 1):
            try:
                result = future.result()
                results.append(result)
                print(f"  [{i}/{len(rows)}] ✓ {result['claim_id']}: "
                      f"gold={result['gold']} | naive={result['naive']['label']} | "
                      f"overthink={result['overthinking']['label']} | posthoc={result['posthoc']['label']}")
            except Exception as e:
                print(f"  [{i}/{len(rows)}] ✗ Error: {e}")
    return results

def compute_f1_macro(predictions: List[str], golds: List[str]) -> Dict:
    counts = {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in LABELS}
    for pred, gold in zip(predictions, golds):
        for lab in LABELS:
            if pred == lab and gold == lab:
                counts[lab]["tp"] += 1
            elif pred == lab and gold != lab:
                counts[lab]["fp"] += 1
            elif pred != lab and gold == lab:
                counts[lab]["fn"] += 1
    
    f1_scores = {}
    for lab in LABELS:
        tp, fp, fn = counts[lab]["tp"], counts[lab]["fp"], counts[lab]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_scores[lab] = f1
    
    return {"macro_f1": sum(f1_scores.values()) / len(f1_scores), "per_class": f1_scores, "counts": counts}

def load_scifact_data(samples: int) -> List[Dict]:
    def extract_label(evidence):
        if not evidence:
            return "NOINFO"
        support_count = sum(1 for anns in evidence.values() for a in anns if a.get("label") == "SUPPORT")
        contradict_count = sum(1 for anns in evidence.values() for a in anns if a.get("label") == "CONTRADICT")
        if support_count > contradict_count:
            return "SUPPORT"
        elif contradict_count > support_count:
            return "CONTRADICT"
        return "NOINFO"
    
    claims_data = []
    for fold in range(1, 6):
        try:
            with open(f"data/cross_validation/fold_{fold}/claims_dev_{fold}.jsonl", "r") as f:
                for line in f:
                    item = json.loads(line)
                    label = extract_label(item.get("evidence", {}))
                    # Only include NOINFO samples
                    if label == "NOINFO":
                        claims_data.append({
                            "id": str(item["id"]),
                            "text": item["claim"],
                            "label_text": label,
                        })
        except FileNotFoundError:
            continue
    if len(claims_data) < samples:
        raise ValueError(f"Not enough NOINFO samples found. Found {len(claims_data)}, requested {samples}")
    return random.sample(claims_data, k=min(samples, len(claims_data)))

def plot_results(model_results: Dict):
    models = list(model_results.keys())
    naive_f1s = [model_results[m]["naive"]["macro_f1"] for m in models]
    overthinking_f1s = [model_results[m]["overthinking"]["macro_f1"] for m in models]
    posthoc_f1s = [model_results[m]["posthoc"]["macro_f1"] for m in models]
    
    # Bar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = range(len(models))
    width = 0.25
    ax.bar([i - width for i in x], naive_f1s, width, label="Naive", color="skyblue")
    ax.bar([i for i in x], overthinking_f1s, width, label="Overthinking", color="coral")
    ax.bar([i + width for i in x], posthoc_f1s, width, label="Post-hoc", color="lightgreen")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Retrieval Strategy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("thought_vs_naive_bar.png", dpi=150)
    print("\n📊 Bar plot saved to thought_vs_naive_bar.png")
    
    # Line graph
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = range(len(models))
    ax.plot(x, naive_f1s, marker='o', linewidth=2.5, markersize=10, label="Naive", color="#3498db")
    ax.plot(x, overthinking_f1s, marker='s', linewidth=2.5, markersize=10, label="Overthinking", color="#e74c3c")
    ax.plot(x, posthoc_f1s, marker='^', linewidth=2.5, markersize=10, label="Post-hoc", color="#2ecc71")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Retrieval Strategy Comparison: F1 by Model", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([min(min(naive_f1s), min(overthinking_f1s), min(posthoc_f1s)) - 0.05,
                 max(max(naive_f1s), max(overthinking_f1s), max(posthoc_f1s)) + 0.05])
    plt.tight_layout()
    plt.savefig("thought_vs_naive_line.png", dpi=150)
    print("📊 Line plot saved to thought_vs_naive_line.png")

def main():
    print("=" * 70)
    print("SciFact Retrieval: NAIVE vs OVERTHINKING vs POST-HOC")
    print("=" * 70)
    
    print("\nLoading corpus...")
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} documents")
    
    print(f"\nLoading {SAMPLES} samples...")
    random.seed(42)
    rows = load_scifact_data(SAMPLES)
    print(f"Loaded {len(rows)} claims")
    
    model_results = {}
    all_logs = []
    manager = Manager()
    debug_list = manager.list()
    
    for model in MODELS:
        print(f"\n{'=' * 70}")
        print(f"Model: {model}")
        print(f"{'=' * 70}")
        
        results = process_batch_parallel(rows, model, debug_list)
        
        golds = [r["gold"] for r in results]
        naive_preds = [r["naive"]["label"] for r in results]
        overthinking_preds = [r["overthinking"]["label"] for r in results]
        posthoc_preds = [r["posthoc"]["label"] for r in results]
        
        naive_metrics = compute_f1_macro(naive_preds, golds)
        overthinking_metrics = compute_f1_macro(overthinking_preds, golds)
        posthoc_metrics = compute_f1_macro(posthoc_preds, golds)
        
        model_results[model] = {
            "naive": naive_metrics,
            "overthinking": overthinking_metrics,
            "posthoc": posthoc_metrics,
            "results": results
        }
        
        # Compact logs
        for r in results:
            all_logs.append({
                "model": model,
                "claim_id": r["claim_id"],
                "claim": r["claim"],
                "gold": r["gold"],
                "naive_label": r["naive"]["label"],
                "naive_result": r["naive"]["result"],
                "naive_doc_ids": r["naive"]["doc_ids"],
                "overthinking_label": r["overthinking"]["label"],
                "overthinking_result": r["overthinking"]["result"],
                "overthinking_doc_ids": r["overthinking"]["doc_ids"],
                "posthoc_label": r["posthoc"]["label"],
                "posthoc_result": r["posthoc"]["result"],
                "posthoc_doc_ids": r["posthoc"]["doc_ids"],
                "posthoc_metadata": r["posthoc"]["metadata"]  # Include posthoc confidence for optimization
            })
        
        print(f"\n{model} Results:")
        print(f"  Naive:        F1 = {naive_metrics['macro_f1']:.3f}")
        print(f"  Overthinking: F1 = {overthinking_metrics['macro_f1']:.3f}")
        print(f"  Post-hoc:     F1 = {posthoc_metrics['macro_f1']:.3f}")
    
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for model in MODELS:
        naive_f1 = model_results[model]["naive"]["macro_f1"]
        overthink_f1 = model_results[model]["overthinking"]["macro_f1"]
        posthoc_f1 = model_results[model]["posthoc"]["macro_f1"]
        print(f"{model:<12} Naive:{naive_f1:.3f} | Overthink:{overthink_f1:.3f} | Posthoc:{posthoc_f1:.3f}")
    
    with open("thought_vs_naive_logs.json", "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"\n📝 Logs saved to thought_vs_naive_logs.json")
    
    with open("thought_vs_naive_debug.log", "w") as f:
        f.write("\n".join(list(debug_list)))
    print(f"🔍 Debug logs saved ({len(debug_list)} entries)")
    
    with open("thought_vs_naive_summary.json", "w") as f:
        summary = {model: {
            "naive_f1": model_results[model]["naive"]["macro_f1"],
            "overthinking_f1": model_results[model]["overthinking"]["macro_f1"],
            "posthoc_f1": model_results[model]["posthoc"]["macro_f1"],
            "naive_per_class": model_results[model]["naive"]["per_class"],
            "overthinking_per_class": model_results[model]["overthinking"]["per_class"],
            "posthoc_per_class": model_results[model]["posthoc"]["per_class"]
        } for model in MODELS}
        json.dump(summary, f, indent=2)
    print(f"📊 Summary saved to thought_vs_naive_summary.json")
    
    plot_results(model_results)
    print(f"\n{'=' * 70}")
    print("Complete!")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY")
    main()
