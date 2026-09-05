# eval_runner.py — Parallel Automated Test Runner for TransformerSOM Checkpoints

import subprocess
import os
import time

TEST_PROMPTS = [
    "hello how are you",
    "what is your name",
    "where are you going",
    "tell me something good",
    "who is there"
]

def run_test_suite(ckpt_path):
    if not os.path.exists(ckpt_path):
        return None
    
    cmd = ["transformer_som.exe", "--test-suite", ckpt_path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return res.stdout
    except Exception as e:
        return f"Error executing test suite: {e}"

def parse_and_score(output_text):
    if not output_text:
        return {}
    
    lines = output_text.split('\n')
    results = {}
    current_prompt = None
    
    for line in lines:
        if line.startswith("PROMPT: "):
            current_prompt = line.replace("PROMPT: ", "").strip()
        elif line.startswith("RESPONSE: ") and current_prompt:
            resp = line.replace("RESPONSE: ", "").strip()
            results[current_prompt] = resp
            current_prompt = None

    total_words = 0
    all_words = []
    non_fallback_count = 0
    
    for p, r in results.items():
        words = r.split()
        total_words += len(words)
        all_words.extend(words)
        if r != "i see" and "[no known words" not in r:
            non_fallback_count += 1
            
    unique_words = len(set(all_words))
    diversity_score = (unique_words / total_words) if total_words > 0 else 0.0
    
    return {
        "responses": results,
        "total_words": total_words,
        "unique_words": unique_words,
        "diversity_score": round(diversity_score, 3),
        "active_response_ratio": f"{non_fallback_count}/{len(TEST_PROMPTS)}"
    }

def main():
    print("--- TRANSFORMERSOM EVALUATION RUNNER STARTED ---")
    start_time = time.time()
    processed_epochs = set()
    summary_path = "eval_summary.md"
    
    with open(summary_path, "w") as f:
        f.write("# TransformerSOM (QKV Self-Attention + MiniLM 384d + OASST) Evaluation Summary\n\n")
        f.write("| Epoch | Active Responses | Total Words | Unique Words | Diversity | Response Preview |\n")
        f.write("|-------|------------------|-------------|--------------|-----------|------------------|\n")

    max_epochs = 15
    
    while len(processed_epochs) < max_epochs:
        for ep in range(1, max_epochs + 1):
            if ep in processed_epochs:
                continue
            ckpt = f"checkpoint_epoch_{ep}.bin"
            if os.path.exists(ckpt) and os.path.getmtime(ckpt) >= start_time:
                time.sleep(2)  # ensure file write is complete
                print(f"\n[EVAL RUNNER] Found newly trained {ckpt}. Running evaluation...")
                out = run_test_suite(ckpt)
                metrics = parse_and_score(out)
                
                print(f"[EVAL RUNNER Epoch {ep}] Metrics: {metrics.get('active_response_ratio')} active, {metrics.get('total_words')} words, div={metrics.get('diversity_score')}")
                
                first_resp = list(metrics.get("responses", {}).values())[0] if metrics.get("responses") else "N/A"
                preview = (first_resp[:40] + '...') if len(first_resp) > 40 else first_resp
                
                with open(summary_path, "a") as f:
                    f.write(f"| {ep} | {metrics.get('active_response_ratio')} | {metrics.get('total_words')} | {metrics.get('unique_words')} | {metrics.get('diversity_score')} | `{preview}` |\n")
                
                with open(f"eval_epoch_{ep}.txt", "w") as f:
                    f.write(f"--- Epoch {ep} Evaluation Detail ---\n\n")
                    for p, r in metrics.get("responses", {}).items():
                        f.write(f"PROMPT: {p}\nRESPONSE: {r}\n\n")
                
                processed_epochs.add(ep)
                
        time.sleep(5)
        
    print("\nAll 20 epoch evaluations complete!")

if __name__ == "__main__":
    main()
