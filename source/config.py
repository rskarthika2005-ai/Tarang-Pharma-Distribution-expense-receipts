from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
OUTPUT_DIR = ROOT_DIR / "output"

TRAIN_CSV = DATA_DIR / "claims_train.csv"
TEST_CSV = DATA_DIR / "claims_test.csv"
EMPLOYEES_CSV = DATA_DIR / "employees.csv"

DATASET_REPORT = OUTPUT_DIR / "dataset_analysis_report.txt"
MAPPING_REPORT = OUTPUT_DIR / "receipt_mapping_report.csv"

IMAGE_EXTENSIONS = (".jpg", ".jpeg")


QUALITY_REPORT = OUTPUT_DIR / "image_quality_report.csv"

BLUR_THRESHOLD = 60        
DARK_THRESHOLD = 70        
BRIGHT_THRESHOLD = 235     
LOW_CONTRAST_THRESHOLD = 20  
MIN_SIDE_PIXELS = 300      


OCR_RESULTS = OUTPUT_DIR / "ocr_results.csv"
OCR_CACHE = OUTPUT_DIR / "ocr_cache.json"

DEFAULT_MODEL = "gpt-4o-mini"  
PROMPT_VERSION = "v3-gstin"     
LLM_RETRIES = 3


OCR_SKIP_BLUR = 30
OCR_SKIP_BRIGHTNESS = 70
OCR_SKIP_CONTRAST = 20



EVALUATION_REPORT = OUTPUT_DIR / "evaluation_report.txt"
VENDOR_MATCH_RATIO = 0.85   


ESCALATE_FAILED_READS = True
DEFAULT_ESCALATION_MODEL = "gpt-4o"   
DATE_WINDOW_DAYS = 0                 
AMOUNT_ROUNDING_STEP = 10             


PRICES_PER_M_TOKENS = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}


READ_EVIDENCE = True               
EVIDENCE_PROMPT_VERSION = "ev1-b"

PREDICTIONS_CSV = ROOT_DIR / "predictions.csv"
CLAIM_DECISIONS = OUTPUT_DIR / "claim_decisions.csv"
MISMATCH_REPORT = OUTPUT_DIR / "ocr_mismatches.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"


FOOD_LIMIT_PER_BILL = 2500
LODGING_LIMIT_PER_NIGHT = {"M1": 3000, "M2": 3000, "M3": 4500}  
TWO_WHEELER_MAX_LITRES = 15      
TWO_WHEELER_MAX_FUEL_RS = 1500    
MIN_CONFIDENCE = 0.6              
VENDOR_MISMATCH_BELOW = 0.5       

OCR_WORKERS = 4  
EVIDENCE_CATEGORIES = None
