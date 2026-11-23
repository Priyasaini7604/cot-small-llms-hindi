# # CoT in Small Open LLMs: Pipeline (GSM8K, English vs Hindi)
#
# **For every member:**
# 1. Turn on the GPU: Runtime > Change runtime type > **GPU**.
# 2. Add the shared folder `DL_CoT_Project` as a shortcut in your **My Drive** (in Drive: right-click the folder > Organize > Add shortcut > My Drive).
# 3. Change only `MODEL`, `MODEL_TAG`, `BS`, and `USE_4BIT` in **Cell 1**. The other settings are team-wide, so **do not change them**.
# 4. Run the cells from top to bottom. Cell 7 is a small test (nothing is saved); Cell 8 runs the full experiments.
#
# | Member | MODEL | MODEL_TAG |
# |---|---|---|
# | Priya | Qwen/Qwen2.5-0.5B-Instruct | qwen0.5b |
# | Member 1 | Qwen/Qwen2.5-1.5B-Instruct | qwen1.5b |
# | Member 2 | Qwen/Qwen2.5-3B-Instruct | qwen3b |
# | Member 3 | Qwen/Qwen2.5-7B-Instruct (USE_4BIT = True) | qwen7b |


# ===== CELL 1: CHANGE ONLY THIS PART (your own model) =====
MODEL     = "Qwen/Qwen2.5-0.5B-Instruct"   # Hugging Face model ID to evaluate
MODEL_TAG = "qwen0.5b"     # short tag used in result file names: qwen0.5b / qwen1.5b / qwen3b / qwen7b
BS        = 16             # batch size. 3B: 8-16, 7B: 4-8 (reduce it if you get a memory error)
USE_4BIT  = False          # set True for 7B (loads the model in 4-bit to save GPU memory)

# ===== Team-wide settings: DO NOT CHANGE =====
N              = 200       # number of questions per experiment
SEED           = 42        # random seed for reproducibility
MAX_NEW_TOKENS = 512       # maximum number of tokens the model may generate per answer
SC_SAMPLES     = 5         # number of samples for self-consistency
SC_TEMP        = 0.7       # sampling temperature for self-consistency
ROOT = "/content/drive/MyDrive/DL_CoT_Project/"   # shared project folder on Google Drive


# ===== CELL 2: Drive mount + paths =====
from google.colab import drive
drive.mount("/content/drive")   # mount Google Drive so Colab can read/write project files

import os, re, json, time, glob  # standard libraries: file paths, regex, JSON, timing, file patterns
from collections import Counter  # used for majority voting in self-consistency

DATA = ROOT + "data/"      # folder holding the question and few-shot JSON files
RES  = ROOT + "results/"   # folder where result files are saved
# Stop early with a clear message if the shared folder shortcut is missing
assert os.path.isdir(ROOT), "DL_CoT_Project folder not found. Add a shortcut of the shared folder to My Drive."
os.makedirs(RES, exist_ok=True)   # create the results folder if it does not exist yet
print("Files in data/:", os.listdir(DATA))   # show which data files are available


# ===== CELL 3: Model load =====
import subprocess, sys
subprocess.check_call([sys.executable, '-m'] + 'pip install -q transformers accelerate bitsandbytes'.split())
import torch
from transformers import pipeline, BitsAndBytesConfig

if USE_4BIT:
    # 4-bit quantized loading (for the large 7B model)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    pipe = pipeline("text-generation", model=MODEL, device_map="auto",
                    model_kwargs={"quantization_config": q})
else:
    # Normal loading; newer transformers versions use `dtype`,
    # older versions use `torch_dtype`, so we try both
    try:
        pipe = pipeline("text-generation", model=MODEL, device_map="auto", dtype="auto")
    except Exception:
        pipe = pipeline("text-generation", model=MODEL, device_map="auto", torch_dtype="auto")

# Left padding is required for correct batched generation in decoder-only models
pipe.tokenizer.padding_side = "left"
# Some tokenizers have no pad token; reuse the end-of-sequence token for padding
if pipe.tokenizer.pad_token is None:
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
print("Model ready:", MODEL)


# ===== CELL 4: Answer extraction =====
# Translation table that maps Hindi (Devanagari) digits to English digits
HI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# Regex for a number: optional minus sign, digits, optional decimal part
NUM = r"-?\d+(?:\.\d+)?"

def extract(text):
    """Extracts the final number (as a string) from the model's output. Returns None if none is found."""
    # Convert Hindi digits to English digits and remove thousands separators
    text = text.translate(HI_DIGITS).replace(",", "")
    # 1st choice: the number after "answer is" (the last match wins)
    m = re.findall(r"answer is[\s:*$\\(]*(" + NUM + ")", text, flags=re.I)
    if m:
        return m[-1]
    # 2nd choice: the number inside \boxed{...} (the last match wins)
    m = re.findall(r"\\boxed\{\s*\$?(" + NUM + ")", text)
    if m:
        return m[-1]
    # Fallback: the last number that appears anywhere in the text
    nums = re.findall(NUM, text)
    return nums[-1] if nums else None

def norm(p):
    """Makes '26', '26.0', '26.' all look the same (needed for majority voting)."""
    f = float(p)
    # Whole numbers become ints (no ".0"); other values stay as floats
    return str(int(f)) if f == int(f) else str(f)

def same(pred, gt):
    """Returns True if the predicted number equals the ground truth (within a tiny tolerance)."""
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except (TypeError, ValueError):
        # pred is None or not a valid number -> counts as wrong
        return False


# ===== CELL 5: Prompts (4 strategies, English + Hindi) =====
# Rule: the instruction is always in English; the question is in its own language (en or hi).
def clean_final(ans):
    """Returns only the final number from a few-shot example's answer (GSM8K '#### 72' format)."""
    if "####" in ans:
        # GSM8K marks the final answer after "####"
        return ans.split("####")[-1].strip().replace(",", "")
    return extract(ans)

def build_prompt(q, strategy, fewshot=None):
    """Builds the prompt text for question q using the given strategy."""
    if strategy == "zeroshot":
        # Zero-shot: ask for the final answer only, no reasoning steps
        return q + "\nDo not show any steps. Give only the final answer in this form: The answer is <number>."
    if strategy in ("cot", "selfcons"):
        # Chain-of-thought (also used for self-consistency): reason step by step
        return q + "\nLet's think step by step. End your response with: The answer is <number>."
    if strategy == "fewshot":
        # Few-shot: prepend solved examples (question + final answer only), then ask the real question
        shots = "".join(f"Question: {e['question']}\nThe answer is {clean_final(e['answer'])}.\n\n"
                        for e in fewshot)
        return shots + f"Question: {q}\nGive only the final answer in this form: The answer is <number>."
    raise ValueError("unknown strategy: " + strategy)


# ===== CELL 6: Experiment engine =====
def load_data(lang):
    """Loads the 200 GSM8K questions for the given language."""
    data = json.load(open(DATA + f"gsm8k_200_{lang}.json", encoding="utf-8"))
    assert len(data) == 200, f"{lang}: expected 200 entries, got {len(data)}"
    return data

def load_fewshot(lang):
    """Loads the 4 few-shot examples for the given language."""
    fs = json.load(open(DATA + f"fewshot_{lang}.json", encoding="utf-8"))
    assert len(fs) == 4, f"fewshot_{lang}: expected 4 examples, got {len(fs)}"
    return fs

def generate_batch(prompts, sample=False):
    """Generates one answer per prompt. sample=True uses random sampling; otherwise greedy decoding."""
    # Wrap each prompt as a single-turn chat message (the model's chat format)
    chats = [[{"role": "user", "content": p}] for p in prompts]
    # Sampling settings for self-consistency, greedy decoding otherwise
    gen = dict(do_sample=True, temperature=SC_TEMP) if sample else dict(do_sample=False)
    res = pipe(chats, batch_size=BS, max_new_tokens=MAX_NEW_TOKENS, **gen)
    outs = []
    for r in res:
        # Pipeline results may be wrapped in a list; unwrap if so
        r = r[0] if isinstance(r, list) else r
        # The last message in the generated chat is the model's reply
        outs.append(r["generated_text"][-1]["content"])
    return outs

def run_experiment(strategy, lang, n=N, save=True):
    """Runs one strategy on one language, computes accuracy, and optionally saves the results."""
    # Only full runs (n == N) may be saved, so small tests never overwrite real results
    if save and n != N:
        print(f"WARNING: n={n} (not N={N}), so the file will not be saved.")
        save = False
    data = load_data(lang)[:n]                                        # first n questions
    fs = load_fewshot(lang) if strategy == "fewshot" else None        # few-shot examples only when needed
    prompts = [build_prompt(ex["question"], strategy, fs) for ex in data]
    torch.manual_seed(SEED)    # fix the seed so runs are reproducible
    t0 = time.time()           # start timer

    items, correct = [], 0     # per-question records and the running correct count
    if strategy == "selfcons":
        # Self-consistency: generate SC_SAMPLES answers per question using sampling
        samples = [generate_batch(prompts, sample=True) for _ in range(SC_SAMPLES)]
        # zip(*samples) regroups the outputs so each question has all of its samples together
        for ex, outs in zip(data, zip(*samples)):
            preds = [extract(o) for o in outs]                        # extracted answer from each sample
            votes = [norm(p) for p in preds if p is not None]         # normalized valid answers
            # Majority vote across samples (None if no sample gave an answer)
            pred = Counter(votes).most_common(1)[0][0] if votes else None
            ok = same(pred, ex["answer"])
            correct += ok
            items.append({"id": ex["id"], "outputs": list(outs), "preds": preds,
                          "pred": pred, "gt": ex["answer"], "correct": bool(ok)})
    else:
        # Single-answer strategies (zeroshot / cot / fewshot): greedy decoding, one output per question
        outs = generate_batch(prompts)
        for ex, o in zip(data, outs):
            pred = extract(o)
            ok = same(pred, ex["answer"])
            correct += ok
            # format_ok records whether the model followed the "The answer is ..." format
            items.append({"id": ex["id"], "output": o, "pred": pred, "gt": ex["answer"],
                          "correct": bool(ok), "format_ok": "answer is" in o.lower()})

    acc = correct / len(data)   # accuracy as a fraction between 0 and 1
    # Bundle everything (settings included) so results can be compared across team members
    result = {"model": MODEL, "model_tag": MODEL_TAG, "strategy": strategy, "lang": lang,
              "n": len(data), "accuracy": acc, "correct": int(correct),
              "settings": {"seed": SEED, "max_new_tokens": MAX_NEW_TOKENS, "batch_size": BS,
                           "sc_samples": SC_SAMPLES, "sc_temp": SC_TEMP, "4bit": USE_4BIT},
              "seconds": round(time.time() - t0), "items": items}
    name = f"{MODEL_TAG}_{strategy}_{lang}.json"   # result file name, e.g. qwen0.5b_cot_en.json
    if save:
        json.dump(result, open(RES + name, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"{name}: {correct}/{len(data)} = {100*acc:.1f}%  ({result['seconds']} sec)"
          + ("" if save else "  [test run, not saved]"))
    return result


# ===== CELL 7: Small test (10 questions, nothing is saved) =====
# Run this first. Whatever the score is, there should just be no errors.
for strat in ["zeroshot", "cot", "fewshot"]:
    run_experiment(strat, "en", n=10, save=False)

# Read a few outputs yourself to check that the extraction is correct:
r = run_experiment("cot", "en", n=3, save=False)
for it in r["items"]:
    print("-" * 60)
    print(it["output"][-300:])   # last 300 characters of the model's output
    print("EXTRACTED:", it["pred"], "| GT:", it["gt"])   # extracted answer vs ground truth


# ===== CELL 8: FULL RUNS (200 questions, English + Hindi) =====
# Skips any result file that already exists, so if the session ends midway,
# re-run Cells 1-6 and then this cell, and it will continue from where it stopped.
# Hindi runs happen only when data/ contains gsm8k_200_hi.json and fewshot_hi.json.
ORDER = ["zeroshot", "cot", "fewshot", "selfcons"]     # quick strategies first, selfcons last (slowest)

for lang in ["en", "hi"]:
    # Skip a language if its question file is not available yet
    if not os.path.exists(DATA + f"gsm8k_200_{lang}.json"):
        print(f"[{lang}] data file is not ready yet, skipping.")
        continue
    for strat in ORDER:
        # Skip experiments that are already done (resume support)
        if os.path.exists(RES + f"{MODEL_TAG}_{strat}_{lang}.json"):
            print(f"skip (already done): {MODEL_TAG}_{strat}_{lang}.json")
            continue
        run_experiment(strat, lang)


# ===== CELL 9: Summary of your model's results =====
import pandas as pd
rows = []
# Read every saved result file that belongs to this model
for f in sorted(glob.glob(RES + f"{MODEL_TAG}_*.json")):
    r = json.load(open(f, encoding="utf-8"))
    rows.append({"strategy": r["strategy"], "lang": r["lang"], "acc%": round(100 * r["accuracy"], 1)})
if rows:
    # Table with strategies as rows and languages as columns
    print(pd.DataFrame(rows).pivot(index="strategy", columns="lang", values="acc%"))
print(len(rows), "/ 8 files ready")   # 4 strategies x 2 languages = 8 files in total
