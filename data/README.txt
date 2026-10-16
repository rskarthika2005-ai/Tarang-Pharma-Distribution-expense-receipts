TARANG PHARMA DISTRIBUTION - EXPENSE CLAIMS DATA PACK
=====================================================

images/               Receipt photos as uploaded through the expense app (JPEG).

claims_train.csv      Claims Finance has already processed.
claims_test.csv       The latest claims, not processed yet. Same columns without the clerk_ columns, status and clerk_note.
  claim_id            Claim number
  emp_id              Who claimed it (see employees.csv)
  submitted_on        Date the claim was submitted in the app (YYYY-MM-DD)
  expense_date        Expense date as typed by the rep
  category            Fuel | Food | Lodging | Local travel | Courier | Parking (chosen by the rep)
  vendor              Vendor as typed by the rep
  amount_claimed      Amount claimed, as typed by the rep (Rs)
  rep_comment         Rep's comment
  image_file          Photo file name as recorded by the app at upload
  clerk_vendor        Vendor as typed by the clerk from the photo            (train only)
  clerk_bill_date     Bill date as typed by the clerk (DD-MM-YYYY)            (train only)
  clerk_bill_no       Bill / receipt / invoice number as typed by the clerk   (train only)
  clerk_bill_amount   Bill amount as typed by the clerk (Rs)                  (train only)
  clerk_gstin         GSTIN as typed by the clerk, blank if none              (train only)
  status              approved | queried | rejected                           (train only)
  clerk_note          Clerk's note                                            (train only)

employees.csv         emp_id, name, designation, headquarter, manager_id, vehicle_no, grade

sample_submission.csv The format for predictions.csv: one row per claim in claims_test.csv.
  claim_id            From claims_test.csv
  vendor              Vendor name as printed on the bill
  receipt_date        Bill date, YYYY-MM-DD
  receipt_number      Bill / receipt / invoice / ticket / AWB number as printed
  gstin               GSTIN as printed, blank if the bill has none
  total_amount        Total amount on the bill (Rs), number only
  decision            approve | review
  reason              Short reason (free text)

expense-policy.pdf    Tarang field expense policy v3.2 - what can be claimed, how Finance checks, costs, volumes.
email-thread.txt      Messages already exchanged about this work.

No other documentation is available.
