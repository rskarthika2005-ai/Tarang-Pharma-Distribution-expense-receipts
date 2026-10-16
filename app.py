import streamlit as st

from dashboard import (load_data, run_analysis, show_decisions, show_image_quality, show_mismatches,
                       show_ocr_results, show_overview, show_receipt_preview, show_reports, show_summary,
                       sidebar_filters)

st.set_page_config(page_title="Tarang Expense AI", layout="wide")
st.title("Tarang Expense AI")
st.caption("Reads bill photos, checks them against the expense policy and proposes approve or review.")

if st.button("Run Analysis"):
    run_analysis()


try:
    table, quality, ocr, mismatches = load_data()
except FileNotFoundError as error:
    st.error("A report file is missing. Click Run Analysis, or run the phases in pipeline.py first.")
    st.caption(str(error))
    st.stop()
except Exception as error:
    st.error("Could not read the reports: " + str(error)[:300])
    st.stop()

shown = sidebar_filters(table)
show_summary(shown)

tabs = st.tabs(["Dataset", "Image quality", "OCR results", "Mismatches", "Decisions", "Reports"])
with tabs[0]:
    show_overview()
with tabs[1]:
    show_image_quality(quality, shown)
with tabs[2]:
    show_ocr_results(shown)
    show_receipt_preview(shown)
with tabs[3]:
    show_mismatches(mismatches, shown)
with tabs[4]:
    show_decisions(shown)
with tabs[5]:
    show_reports()