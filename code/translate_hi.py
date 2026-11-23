# =============================================================================
# Translate GSM8K questions from English to Hindi with the Gemini API
# =============================================================================
# What this script does:
#   1. Reads the English questions (gsm8k_200_en.json) and selects the id range assigned to this team member.
#   2. Sends the questions to a Gemini model in small batches and asks for a Hindi translation (JSON).
#   3. Saves the result after every batch, so the script can be re-run and continues where it stopped.
#   4. Runs automatic checks on the translations (same numbers as the English question, no English
#      words left, question ends properly).
# Only the "question" text is translated. The "answer" field is copied from the English file unchanged.
# The API key is read from the environment variable GEMINI_API_KEY (never write the key into the code).
import json, os, re, time
from google import genai        # Google Gemini API client

# ====== Settings ======
START_ID, END_ID = 0, 66                 # range of question ids translated by this run (both ends included)
OUT_NAME = "hi_part1.json"               # output file for this part
MODELS = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.1-flash-lite",
          "gemini-flash-lite-latest", "gemini-2.5-flash-lite"]   # models are tried in this order (fallbacks if one is busy or unavailable)
BATCH = 5                                # number of questions sent per request
# ======================

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])      # API key comes from the environment
DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")                # converts Hindi digits to English digits
# Sorted list of all numbers in a text (thousands separators removed); used to compare English and Hindi questions
nums = lambda t: sorted(re.findall(r"\d+\.?\d*", t.replace(",", "")))

# Instruction sent to the model together with the questions. The rules protect the numbers and conditions
# of the math problems, because a dropped word such as "twice" or "remaining" would change the answer.
PROMPT = """Translate these math word problems from English to natural, fluent Hindi.
Rules:
- Keep ALL numbers as English digits (0-9), never Hindi digits. Keep decimals, fractions, $ and % exactly as in the original.
- Translate every word and every condition: twice, half, tripled, more than, less than, each, remaining etc. must stay in the Hindi sentence. Do not drop any part.
- The final question sentence must be present and mean the same as the English one.
- Keep units, names and meaning exactly the same. Write person names in Devanagari.
- Do NOT solve the problems. Do not leave English words in the Hindi text.
Return ONLY a JSON list like [{"id": 0, "question": "..."}], same ids, translated question text only.

"""

dead_models = set()   # models that returned 404 (not available); they are not tried again

def translate_batch(items):
    # Translate one batch of questions. Returns (translated items, model used), or (None, None) if every attempt failed.
    # The batch is sent as JSON: only the id and the English question text
    payload = json.dumps([{"id": x["id"], "question": x["question"]} for x in items],
                         ensure_ascii=False)
    want = {str(x["id"]) for x in items}       # ids that must come back, used to validate the reply
    for rnd in range(4):                       # up to 4 rounds over all models
        for model in MODELS:
            if model in dead_models:           # skip models known to be unavailable
                continue
            try:
                # Ask for JSON output and use a low temperature so the translation is stable
                r = client.models.generate_content(
                    model=model, contents=PROMPT + payload,
                    config={"response_mime_type": "application/json", "temperature": 0.2})
                out = json.loads(r.text)
                # Accept the reply only if it contains exactly the requested ids
                if {str(o["id"]) for o in out} == want:
                    return out, model
                print(f"   [{model}] returned different ids than requested")
            except Exception as e:
                msg = str(e)
                print(f"   [{model}] round {rnd+1} ->", msg[:90])
                if "404" in msg:
                    dead_models.add(model)     # model not found: never try it again
                elif "429" in msg:
                    time.sleep(20)             # rate limit reached: wait before the next model
            time.sleep(3)                      # short pause between requests
        wait = 20 * (rnd + 1)                  # waiting time grows with every round
        print(f"   all models busy, waiting {wait} sec...")
        time.sleep(wait)
    return None, None                          # all rounds failed

# Select this member's question ids from the English file
en = json.load(open("gsm8k_200_en.json", encoding="utf-8"))
subset = [x for x in en if START_ID <= int(x["id"]) <= END_ID]
print("Questions to translate:", len(subset))

# Resume support: load translations that were already saved by an earlier run
done = {}
if os.path.exists(OUT_NAME):
    for x in json.load(open(OUT_NAME, encoding="utf-8")):
        done[str(x["id"])] = x
    print("Already translated:", len(done))

def save():
    # Write all translations so far, ordered by id
    res = sorted(done.values(), key=lambda x: int(x["id"]))
    json.dump(res, open(OUT_NAME, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# Main loop: translate the remaining questions batch by batch
todo = [x for x in subset if str(x["id"]) not in done]
for i in range(0, len(todo), BATCH):
    chunk = todo[i:i + BATCH]
    out, used = translate_batch(chunk)
    if out is None:
        print("FAILED batch ids", [x["id"] for x in chunk], "- run the cell again")   # rerunning retries only the missing ids
        continue
    by_id = {str(o["id"]): o["question"] for o in out}       # translated text per id
    for item in chunk:
        k = str(item["id"])
        done[k] = {"id": item["id"],
                   "question": by_id[k].translate(DIGITS),   # convert any Hindi digits to English digits
                   "answer": item["answer"]}   # the answer is taken from the English file, never translated
    save()                                     # save after every batch so that no work is lost
    print(f"saved {len(done)} / {len(subset)}  (model: {used})")
    time.sleep(3)

# ---- Checks ----
# Automatic quality checks on the translated questions. Anything printed below needs a manual look.
print("\n===== CHECKS =====")
res = sorted(done.values(), key=lambda x: int(x["id"]))
en_by = {str(x["id"]): x for x in en}          # English questions by id, for comparison
print("Entries:", len(res), "| expected", len(subset),
      "| ids correct:", [int(x["id"]) for x in res] == [int(x["id"]) for x in subset])
for x in res:
    k = str(x["id"])
    # Check 1: the numbers in the Hindi question must be the same as in the English question
    if nums(x["question"]) != nums(en_by[k]["question"]):
        print("NUMBERS differ, id", k, nums(en_by[k]["question"]), nums(x["question"]))
    # Check 2: no English words (3 or more letters) should be left in the Hindi text
    eng = re.findall(r"[A-Za-z]{3,}", x["question"])
    if eng:
        print("English words, id", k, eng)
    # Check 3: the question should end with "?" or the Hindi full stop "।"
    if not re.search(r"[?।]\s*$", x["question"].strip()):
        print("Question does not end with ? or ।, id", k)
print("Saved:", OUT_NAME)
