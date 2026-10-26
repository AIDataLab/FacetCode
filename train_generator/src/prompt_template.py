import os
from src.root_dir_path import ROOT_DIR

REWRITING_PROMPT = """Rewrite the following clinical text. While keeping medical entities, terminology, and key clinical details intact, create a new version that expresses the same clinical meaning in a different way. The revised text should be distinct from the original but preserve all clinically relevant information.

Text:
{passage}

Only output the rewritten text."""

QA_GENERATION_PROMPT = """Based on the clinical passage provided, generate exactly 5 question-answer pairs covering the following six dimensions of clinical knowledge: (1) Definition, (2) Symptoms, (3) Diagnosis, (4) Treatment, (5) Complications, (6) Summary. Each pair should be grounded solely in the information provided in the passage.

Requirements:
- Each pair must begin with [DIM: <dimension_name>]
- Answers must be 2-4 sentences (40-80 words)
- Questions must be specific enough to distinguish this condition from related conditions
- Do not speculate or hallucinate information not present in the passage
- Use precise clinical terminology consistent with standard medical documentation

Passage:
{passage}

Output format:
[DIM: Definition] Q1: ... A1: ...
[DIM: Symptoms] Q2: ... A2: ...
[DIM: Diagnosis] Q3: ... A3: ...
[DIM: Treatment] Q4: ... A4: ...
[DIM: Complications] Q5: ... A5: ..."""

FACET_PROFILE_PROMPT = """You are a clinical informaticist. Based on the structured evidence provided below, generate a multi-faceted clinical profile for the ICD code.

Structured Evidence:
{evidence}

Requirements:
- Generate descriptions for exactly 6 clinical dimensions: Definition, Symptoms, Diagnosis, Treatment, Complications, Summary
- Each dimension description should be 2-4 sentences
- Use precise medical terminology
- Preserve quantitative thresholds and diagnostic criteria verbatim where clinically significant

Output format:
Definition: <description>
Symptoms: <description>
Diagnosis: <description>
Treatment: <description>
Complications: <description>
Summary: <description>"""


def fix_json_qa(output, model_name=None):
    if model_name and "3.2-1b-instruct" in model_name.lower():
        if "[" in output:
            output = output[output.find("["):]
        if output.endswith(","):
            output = output[:-1]
        if not output.endswith("]"):
            output += "]"
    elif model_name and "3-8b-instruct" in model_name.lower():
        if "[" in output:
            output = output[output.find("["):]
        if "]" in output:
            output = output[:output.find("]")+1]
    return output


def fix_qa(qa):
    if isinstance(qa, list):
        if len(qa) >= 3:
            qa = qa[:3]
            for data in qa:
                if "question" not in data or "answer" not in data or "full_answer" not in data:
                    return False, qa
                if isinstance(data["answer"], list):
                    data["answer"] = ", ".join(data["answer"])
                if isinstance(data["answer"], int):
                    data["answer"] = str(data["answer"])
                if data["answer"] is None:
                    data["answer"] = "Unknown"
            return True, qa
    return False, qa