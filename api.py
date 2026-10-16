import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from pipeline import process_one_claim
from source.llm_client import NoApiKeyError

app = FastAPI(title="Tarang Expense AI")

ALLOWED_CATEGORIES = ["Fuel", "Food", "Lodging", "Local travel", "Courier", "Parking"]


@app.get("/")
def home():
    return {"message": "Tarang Expense AI is running. Use POST /process-receipt or open /docs."}


@app.post("/process-receipt")
async def process_receipt(
    image: UploadFile = File(...),
    emp_id: str = Form(...),
    category: str = Form(...),
    expense_date: str = Form(...),
    vendor: str = Form(...),
    amount_claimed: str = Form(...),
    claim_id: str = Form("NEW"),
):
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(status_code=400, detail="category must be one of: " + ", ".join(ALLOWED_CATEGORIES))

    
    suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await image.read())
        temp_path = temp.name

    claim = {"claim_id": claim_id, "emp_id": emp_id, "category": category,
             "expense_date": expense_date, "vendor": vendor, "amount_claimed": amount_claimed}

    try:
        return process_one_claim(temp_path, claim)
    except NoApiKeyError as message:
        raise HTTPException(status_code=503, detail=str(message))
    except ValueError as message:
        raise HTTPException(status_code=400, detail=str(message))
    except Exception as error:
        raise HTTPException(status_code=500, detail="Could not process the receipt: " + str(error)[:200])
    finally:
        os.remove(temp_path)