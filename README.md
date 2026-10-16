# Tarang Pharma Distribution — Expense Receipt Reader

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=220&text=TARANG%20PHARMA&fontSize=48&fontAlignY=38&desc=EXPENSE%20RECEIPTS%20%7C%20VISION%20OCR%20%7C%20CLAIM%20DECISIONS&descAlignY=60&animation=fadeIn&fontColor=ffffff&color=0:0F766E,50:0D9488,100:2563EB" alt="Tarang Expense Receipts"/>

### AI-Assisted Receipt Reading • Policy Checks • Claim Review

Reads a bill photo, fills in the five required fields, and decides whether the claim can be approved or must go to a clerk.

<img src="https://img.shields.io/badge/Vision%20OCR-0F766E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/GPT%20Mini%20Vision-0D9488?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>

<br/>

### Developed by Karthika S

📧 [rskarthika08@gmail.com](mailto:rskarthika08@gmail.com)

<br/>

<a href="https://tarang-pharma-distribution-expense-receipts-yyxg7wgmnyfou4or8p.streamlit.app/">
<img src="https://img.shields.io/badge/Live%20Dashboard-Open-0D9488?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open Live Dashboard"/>
</a>

<a href="https://github.com/rskarthika2005-ai/Tarang-Pharma-Distribution-expense-receipts">
<img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repository"/>
</a>

</div>

---

## ✦ ABOUT THE PROJECT

Nilesh Agrawal, Tarang's Finance Controller, asked for a tool that reads a bill photo and fills in:

- Vendor
- Bill date
- Bill number
- GSTIN
- Total amount

He wants at least **95% accuracy** on the fields, and he wants claims **under ₹2,000 approved straight from the photo**.

The project gives him three things:

- **Receipt Reading:** A vision model reads each photo and returns the five fields with a confidence score for each.
- **Repair and Checks:** Rules fix common reading errors and flag risky bills.
- **Decision:** Each claim is marked **approve** or **review**, with a short reason.

### End-to-End Workflow

```text
Receipt Photo + Rep's Claim
        ↓
Photo Checks (orientation, blur, brightness, contrast)
        ↓
Vision Model Reads 5 Fields + Confidence
        ↓
Repair (GSTIN, amount) and Safety Checks
        ↓
Policy Rules, Duplicate Check, Suspicious-Content Flags
        ↓
APPROVE  or  REVIEW (with reason)
        ↓
Dashboard • API • Saved Output Files
```

### Quick Navigation

[Objective](#-project-objective) • [Overview](#-project-overview) • [Problems Found](#-problems-found-in-the-data) • [How I Solved It](#-how-i-solved-it) • [Results](#-results) • [AI Tools & Cost](#-ai-tools--cost) • [Skills](#-skills-demonstrated) • [Structure](#-project-structure) • [How to Run](#-how-to-run) • [Limitations](#-limitations) • [Recommendation](#-final-recommendation)

---

## ✦ PROJECT OBJECTIVE

Accuracy on single fields is not enough. A claim is only useful when **all five fields are right together**, and five fields that are each 95% right give only about **77%** of claims fully right. So both field accuracy and all-five-field accuracy were measured.

The tool must also cost less per photo than a clerk, so the savings figure subtracts the model cost.

---

## ✦ PROJECT OVERVIEW

<table align="center">
<tr>
<td align="center" width="25%">

### 467

Receipt Photos

</td>
<td align="center" width="25%">

### 277

Training Claims With Clerk Values

</td>
<td align="center" width="25%">

### 190

Test Claims

</td>
<td align="center" width="25%">

### 139 / 51

Approvals / Reviews

</td>
</tr>
<tr>
<td align="center">

### 30

Hand-Checked Claims

</td>
<td align="center">

### 29 / 30

All Five Fields Right

</td>
<td align="center">

### 6

Claims With No Photo

</td>
<td align="center">

### 3

Bills With Instruction Stamps

</td>
</tr>
</table>

---

## ✦ PROBLEMS FOUND IN THE DATA

- **Clerk values are not perfect.** Of 22 receipt-number differences checked against the photos, the tool was wrong on 10 and the clerk on 12. Thirteen clerk GSTINs were blank. CLM-30421 (bill ₹250, clerk typed ₹240) and CLM-30469 (bill ₹335, clerk typed ₹435) are clerk amount typos.
- **Photos and entries.** Six claims have no photo, CLM-31333 is cut off and CLM-31391 is unreadable. Reps type dates and amounts by hand, some photos are sideways, and the older app saved full phone paths and `.jpeg` names.
- **Risky bills.** One bill appears on two claims (CLM-30982 repeats CLM-30741). Three bills carry a printed stamp that tries to instruct an automated reviewer into approving them.
- **Cost.** The savings figure must subtract the model cost.

---

## ✦ HOW I SOLVED IT

### Stage 1: Reading

1. A vision model reads each bill and returns the five fields with a confidence score per field.
2. The photo is turned upright and resized to 2,000 pixels before it is sent.
3. Everything printed on a bill is treated as **data, never as an instruction** to the AI.

### Stage 2: Repair and Checks

Each improvement was measured by the cases it helped and the cases it hurt.

| Repair | What it does | Result |
|---|---|---|
| GSTIN repair | Fixes look-alike characters, checks GSTIN format and check digit, uses valid bills from the same vendor | GSTIN 80.5% → 89.2%, 24 helped, 0 hurt |
| Amount repair | Fixes taxi fares where OCR reads an extra 7 from the `/-` symbol | 3 helped, 0 hurt; used in CLM-30702, CLM-31182, CLM-31305, CLM-31308 |
| Suspicious flags | Flags placeholder GSTIN `27ABCDE1234F1ZH`, failed check digits and instruction-like text | 16 claims moved from approve to review, 0 the other way |
| GSTIN reread | Reads the GSTIN again only when the check digit fails; saves reads so the same photo is never paid for twice | Lower cost |

### Stage 3: Decision

Policy rules, duplicate checks and photo-quality checks decide approve or review, with a short reason. Any claim of **₹2,000 or more**, any claim with a **blank main field**, and any claim with a **doubtful photo** goes to a clerk.

### Stage 4: Service

The Streamlit dashboard and the FastAPI endpoint sit on top of the same pipeline.

---

## ✦ RESULTS

On 277 training claims with clerk values:

| Field | Agreement with clerk | After photo checks |
|---|---:|---:|
| Vendor | 98.6% | 98.9% |
| Receipt date | 96.4% | 99.3% |
| Receipt number | 92.1% | 96.4% |
| GSTIN | 89.2% (raw 80.5%) | 93.1% to about 95.3% |
| Total amount | 98.2% (raw 97.1%) | Up to 100% |
| All five fields | 77.3% (start 69.3%) | Not recomputed |

By category, all five fields were right on Parking 95.0%, Fuel 85.5%, Local travel 84.8%, Courier 77.8%, Lodging 72.2% and Food 66.0%.

### Independent hand check

The training figures are optimistic because the rules were tuned on those claims. So 30 random test claims were checked against the photos by hand:

- Vendor, date, receipt number and amount were right on 30 of 30.
- GSTIN was right on 29 of 30 (CLM-31478 was wrong).
- All five fields were right on 29 of 30 (96.7%). With a sample this small, the true rate could be roughly 83% to 99%.
- The 95% target is met on four fields. It is **not** claimed for GSTIN or overall.

### Final predictions

`predictions.csv` holds 190 test predictions: **139 approvals and 51 reviews**. A safety check confirms that no approval has a blank main field, a poor photo, low confidence, an amount of ₹2,000 or more, or a bill amount different from the claim.

---

## ✦ AI TOOLS & COST

The project uses a paid GPT mini vision model to read receipt photos. Everything else (repair rules, policy checks, duplicate checks, decisions and evaluation) runs locally in Python and does not need an AI call.

This keeps the cost per claim small and the decision logic easy to inspect.

### AI Tools Used

| Tool | Purpose | Cost Consideration |
|---|---|---|
| Python | Pipeline and application development | No separate language usage fee |
| Pandas | Data analysis and reports | Local processing |
| GPT mini vision model (OpenAI API) | Reads vendor, bill date, bill number, GSTIN and amount from receipt photos | Paid API; about $0.024 (₹2.10) per claim |
| FastAPI | Receipt-reading endpoint | Hosting may incur costs |
| Streamlit | Interactive dashboard | Hosting depends on deployment |
| Claude | Planning, coding, debugging and checking | Depends on the account and service used |

> The exact GPT model name is set in `source/config.py`. Please confirm it there.

### 1. Receipt Reading (GPT Mini Vision Model)

The vision model is the only paid part. It reads each bill photo and returns the five fields with a confidence score for each. The photo is turned upright and resized to 2,000 pixels first, and the prompt is short with JSON-only output to save tokens.

### 2. Local Repair, Checks and Decisions

GSTIN repair, amount repair, suspicious-content flags, policy rules, duplicate checks and the approve or review decision all run locally. They need no API call.

### 3. API Key

An OpenAI API key was created and provided for this project. It is stored in a `.env` file and read by `source/config.py`. It is never typed into code and never committed to GitHub (`.env` is in `.gitignore`).

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your-key-here
```

> Use the same variable name that `source/config.py` reads.

**Behaviour with no API key**

- The dashboard still starts and shows all **saved results**.
- Reading a **new** photo needs a valid key and credits. Without one, it fails politely with a clear message.

### Actual API Usage (9–16 October 2026)

| Metric | Actual usage |
|---|---:|
| API requests | 3,034 |
| Tokens used | 40,247,791 |
| Total spend | $11.27 |
| Claims processed | 473 |

### Cost and Estimated Benefit

| Item | Estimate |
|---|---:|
| Cost per claim | about $0.024 (₹2.10) |
| Monthly model cost (2,400 claims) | about ₹5,030 |
| Staff time freed per month | about 102 hours (worth about ₹24,300) |
| Net monthly benefit | about ₹19,300 |
| Annual benefit | about ₹2.3 lakh |

These are estimates, not guaranteed cash savings. They assume the test approval rate continues and exclude some review-related time savings.

### How Token Use Was Reduced

- Saved receipt reads to avoid repeat API cost
- Resized images to 2,000 pixels
- Short prompts with JSON-only output
- Re-read only GSTINs that failed the check digit
- Stopped immediately when credits ran out


---

## ✦ SKILLS DEMONSTRATED

<div align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenAI%20API-412991?style=for-the-badge&logo=openai&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Git%20%26%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>

</div>

| Skill Area | What I Did in This Project |
|---|---|
| 🧾 **Receipt OCR with a vision model** | Read vendor, bill date, bill number, GSTIN and amount from phone photos, with a confidence score for each field |
| 🧠 **Prompt design and AI safety** | Used short JSON-only prompts and treated all printed text as data, so instruction-like stamps on bills could not steer the model |
| 🖼️ **Image preprocessing** | Corrected orientation, resized photos to 2,000 pixels, and checked blur, brightness and contrast |
| 🔧 **Data repair and validation** | Fixed look-alike GSTIN characters using format rules and check digits, and corrected taxi-fare amount reads |
| 📏 **Rules and decision logic** | Applied the expense policy, duplicate checks and suspicious-GSTIN flags to choose approve or review |
| 📊 **Honest evaluation** | Measured field accuracy and all-five-field accuracy, counted cases helped and hurt by each change, and hand-checked 30 test claims |
| 💰 **Cost awareness** | Cached reads, reread only failed GSTINs and subtracted model cost from the savings estimate |
| 🖥️ **Dashboard and API building** | Built a Streamlit review screen and a FastAPI endpoint on the same pipeline |
| 🗂️ **Clean project structure** | Split the code into config, modules, outputs and entry-point files so each part has one clear job |
| 🤝 **Business communication** | Wrote a memo with a clear recommendation, limitations and what is needed from the client |

---

## ✦ PROJECT STRUCTURE

```text
tarang-expense-ai/
│
├── README.md                    # Project overview, setup, how to run, no-API-key behaviour, limitations
├── memo.md                      # One-page memo to Nilesh
├── submission-form.md           # Completed submission form
├── predictions.csv              # Final predictions for all 190 test claims
├── requirements.txt             # Python libraries needed
├── .gitignore                   # Keeps secrets, virtual environment, caches out of git
├── app.py                       # Starts the Streamlit app
├── dashboard.py                 # KPI cards, filters, tabs, receipt preview, report downloads
├── api.py                       # Takes a receipt photo + claim, returns fields, decision, reasons
├── pipeline.py                  # Runs the full flow: analysis, image checks, OCR, validation, decisions
│
├── source/
│   ├── config.py                # Settings, file paths, image-quality thresholds; reads API key from .env
│   └── llm_client.py            # Vision-model client; retries, handles errors, fails politely with no key
│
├── modules/
│   ├── analysis.py              # Inspects datasets, maps photos to claims, writes analysis reports
│   ├── preprocess.py            # Loads photos, checks orientation, blur, brightness, contrast
│   ├── ocr.py                   # Reads vendor, bill date, bill number, GSTIN, total from the photo
│   ├── validation.py            # Repairs and validates values, compares with the rep's claim
│   ├── gstin_reread.py          # Second-read GSTIN code (when check digit fails)
│   ├── decision.py              # Policy rules, duplicate checks, suspicious flags, approve/review
│   └── evaluation.py            # Field accuracy, error analysis, savings, validate_predictions_file()
│
├── data/
│   ├── images/                  # All 467 original receipt photos, never modified
│   ├── claims_train.csv         # Past claims with clerk's typed fields and decisions
│   ├── claims_test.csv          # The 190 new claims needing predictions
│   ├── employees.csv            # Employee details, grades, vehicle numbers
│   ├── expense-policy.pdf       # Tarang's expense policy v3.2
│   ├── email-thread.txt         # Emails with the client's requirements
│   ├── sample_submission.csv    # Required format for predictions.csv
│   └── README.txt               # Column definitions supplied with the data pack
│
└── output/
    ├── ocr_results.csv              # Fields read from every photo, with confidence scores
    ├── ocr_cache.json               # Saved photo reads, so results show without paying again
    ├── claim_decisions.csv          # Decision, policy checks, duplicate flags, reasons per claim
    ├── ocr_mismatches.csv           # Where OCR differs from the rep's claim
    ├── ocr_vs_clerk_differences.csv # Where OCR differs from the clerk's typed values
    ├── image_quality_report.csv     # Blur, size, brightness, contrast, quality status per photo
    ├── receipt_mapping_report.csv   # Maps each claim to its photo; missing or unmatched images
    ├── check_table.csv              # Evidence of the 30-claim hand check
    ├── evaluation_report.txt        # Accuracy, error analysis, limitations, savings estimate
    └── dataset_analysis_report.txt  # Dataset statistics, missing values, duplicates, quality findings
```

---

## ✦ HOW TO RUN

### Clean-machine setup

```bash
git clone PASTE-GITHUB-REPO-LINK-HERE
cd tarang-expense-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add your key to `.env` (see above). The app still starts without it.

### Run the full pipeline

```bash
python pipeline.py
```

### Start the dashboard

```bash
streamlit run app.py
```

---

## ✦ THREE WAYS TO VIEW THE OUTPUT

### 1. Streamlit Dashboard (`app.py` / `dashboard.py`)

- **Dataset overview:** total claims, approved, sent for review, review rate, clerk time saved, clerk cost avoided
- **Filters:** claim ID, claim set, category, image status, blur status, decision
- **Image quality:** find photos that may hurt OCR
- **OCR results:** receipt photo beside the extracted values, with raw and repaired GSTIN and confidence scores
- **Mismatch analysis:** differences between the photo and the claim
- **Claim decisions:** approve or review with the reason
- **Reports and downloads:** all generated reports

### 2. API Endpoint (`api.py`)

Accepts a receipt photo and the rep's claim, and returns the extracted fields, the decision and the reasons. Other applications can use it without the dashboard.

### 3. Saved Output Files (`predictions.csv` and `output/`)

OCR results, decisions, mismatch reports, quality and mapping reports, the hand-check table and the evaluation report can all be reviewed without re-running the model.

---

## ✦ LIMITATIONS

- **Receipt numbers:** 10 bill numbers were misread through dropped or added digits, even with confidence between 0.9 and 1.0. No safeguard catches all of these, so a wrong number could be approved.
- **GSTIN:** accuracy is below 95%. Five test claims have wrong GSTIN values (CLM-31140, CLM-31158, CLM-31233, CLM-31299, CLM-31478). All are in review, but the file still holds the wrong values.
- **Vendor:** CLM-30527 appears to have read the passenger's name as the vendor, and CLM-30944 may have read a brand instead of the dealer.
- **New vendors:** GSTIN repair needs three valid sightings of a vendor, so new vendors may not benefit.
- **Instruction stamps:** caught on CLM-30623 and CLM-31401. The stamp on CLM-30982 was not transcribed, but that claim is in review for other reasons.
- **Small validation sample:** only 30 claims were checked by hand. Receipt dates were checked against the rep's entered date, not independently against the photos.
- **Missing photos:** blank fields are correct where photos are missing, and all those claims are in review.
- **API dependency:** reading a new photo needs a paid vision-model call.

---

## ✦ FINAL RECOMMENDATION

Start a **controlled pilot** of auto-approval for claims under ₹2,000, only when:

- The bill amount matches the claim
- The photo is clear
- All required fields are present
- Confidence and validation checks pass
- No duplicate or suspicious-content flag is raised

Keep claims of ₹2,000 or more, and all uncertain cases, with the clerks. **Do not describe the tool as 95% accurate overall.**

### What I need from Nilesh

1. **Weekly quality audit:** a clerk checks one in every 20 approved claims against the photo.
2. **Reference standard:** judge GSTIN and bill-number accuracy against the photo, not only the clerk's typing.
3. **Review of suspicious claims:** CLM-30982 (possible duplicate bill), CLM-30623 and CLM-31401 (instruction-like stamps). The tool flags these; it does not decide disciplinary outcomes.
4. **Budget and access:** about ₹5,000 per month for the model, and an API key.

---

<div align="center">

### Karthika S

📧 [rskarthika08@gmail.com](mailto:rskarthika08@gmail.com)

*Tarang Pharma Distribution • Expense Receipt Reader*

</div>
