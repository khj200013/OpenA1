import json
import os
from difflib import SequenceMatcher
import streamlit as st

@st.cache_resource
def load_cases(json_path=None):
    if json_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "privacy_law5.json")
    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    return cases

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def filter_cases(user_question, cases, threshold=0.65):
    best_score = 0
    best_case = None
    for case in cases:
        score = similarity(user_question, case["case"])
        if score > best_score:
            best_score = score
            best_case = case
    if best_score >= threshold:
        return best_case
    return None
