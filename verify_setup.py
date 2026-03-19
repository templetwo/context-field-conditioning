#!/usr/bin/env python3
"""
CFC Setup Verification
Run this first to confirm Ollama + Ministral 14B environment is ready.
"""

import subprocess
import sys
import json

def check_python():
    print(f"[✓] Python {sys.version.split()[0]}")
    return True

def check_ollama():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            print(f"[✓] Ollama available — {len(lines)-1} models installed:")
            for line in lines[1:]:
                name = line.split()[0] if line.split() else "?"
                print(f"    → {name}")
            
            # Check for ministral
            ministral_found = any("ministral" in line.lower() or "mistral" in line.lower() for line in lines)
            if ministral_found:
                print("[✓] Ministral/Mistral model found")
            else:
                print("[!] No Ministral model detected — verify model name")
            return True
        else:
            print(f"[✗] Ollama returned error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("[✗] Ollama not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("[✗] Ollama timed out")
        return False

def check_ollama_api():
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print(f"[✓] Ollama API responding — {len(models)} models via API")
            for m in models:
                print(f"    → {m['name']} ({m.get('size', '?')} bytes)")
            return True
        else:
            print(f"[✗] Ollama API returned {resp.status_code}")
            return False
    except ImportError:
        print("[✗] requests not installed — run: pip install requests")
        return False
    except Exception as e:
        print(f"[✗] Ollama API not responding: {e}")
        print("    Make sure Ollama is running: ollama serve")
        return False

def check_logprobs():
    """Test whether the model returns logprobs (critical for entropy measurement)."""
    try:
        import requests
        
        # Find ministral model name
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = resp.json().get("models", [])
        model_name = None
        for m in models:
            if "ministral" in m["name"].lower() or "mistral" in m["name"].lower():
                model_name = m["name"]
                break
        
        if not model_name:
            print("[!] Cannot test logprobs — no Ministral model found")
            return False
        
        print(f"[~] Testing logprobs with {model_name}...")
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "The capital of France is",
                "options": {"temperature": 0.7, "num_predict": 10},
                "stream": False,
            },
            timeout=120,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            response_text = data.get("response", "")
            print(f"[✓] Model responds: '{response_text.strip()[:80]}'")
            
            # Check for logprobs in response
            # Ollama may include them in different fields depending on version
            has_logprobs = False
            for key in ["logprobs", "completion_probabilities", "tokens"]:
                if key in data:
                    has_logprobs = True
                    print(f"[✓] Logprobs available via '{key}' field")
                    break
            
            if not has_logprobs:
                print("[!] Logprobs not found in standard response fields")
                print("    Available response keys:", list(data.keys()))
                print("    May need to use /api/chat with stream=true for token-level data")
                print("    OR compute entropy via repeated sampling (alternative method)")
            
            return True
        else:
            print(f"[✗] Model inference failed: {resp.status_code}")
            return False
    except Exception as e:
        print(f"[✗] Logprob test failed: {e}")
        return False

def check_dependencies():
    missing = []
    for pkg in ["numpy", "scipy", "pandas", "matplotlib", "seaborn", 
                 "sklearn", "yaml", "tqdm", "jsonschema", "requests"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if not missing:
        print("[✓] All core Python dependencies available")
    else:
        print(f"[!] Missing packages: {', '.join(missing)}")
        print(f"    Run: pip install -r requirements.txt")
    
    # Check sentence-transformers separately (heavy dependency)
    try:
        __import__("sentence_transformers")
        print("[✓] sentence-transformers available (for GNS metric)")
    except ImportError:
        print("[!] sentence-transformers not installed (needed for Generative Novelty Score)")
        print("    Run: pip install sentence-transformers")
    
    return len(missing) == 0

def check_project_structure():
    import os
    expected = [
        "CLAUDE.md", "EXPERIMENT_PLAN.md", "README.md", "LICENSE",
        "honeycomb/schema.json",
        "runner/config.yaml",
        "data/raw", "data/processed",
        "analysis/results",
    ]
    missing = [p for p in expected if not os.path.exists(p)]
    if not missing:
        print("[✓] Project structure complete")
    else:
        print(f"[!] Missing: {', '.join(missing)}")
    return len(missing) == 0


if __name__ == "__main__":
    print("=" * 60)
    print("  CONTEXT FIELD CONDITIONING — Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python", check_python),
        ("Ollama CLI", check_ollama),
        ("Ollama API", check_ollama_api),
        ("Logprob Support", check_logprobs),
        ("Python Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
    ]
    
    results = {}
    for name, fn in checks:
        print(f"\n--- {name} ---")
        try:
            results[name] = fn()
        except Exception as e:
            print(f"[✗] Unexpected error: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"  Results: {passed}/{total} checks passed")
    
    if all(results.values()):
        print("  ✓ Ready for experiment. Run: python runner/experiment_runner.py --pilot")
    else:
        print("  ! Fix issues above before running experiment")
    print("=" * 60)
