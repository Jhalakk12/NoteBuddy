import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from week4.evaluator import SE_CATEGORIES, CATEGORY_QUESTIONS, aggregate_category_model, evaluate_category_verdicts, render_markdown

mapping = {
    'repo-01': ('exp-01', 'Explanation'),
    'course-06': ('exp-02', 'Explanation'),
    'repo-04': ('exp-03', 'Explanation'),
    'course-10': ('ret-01', 'Code Retrieval'),
    'course-11': ('ret-02', 'Code Retrieval'),
    'course-12': ('ret-03', 'Code Retrieval'),
    'repo-02': ('dep-01', 'Dependency Understanding'),
    'repo-03': ('dep-02', 'Dependency Understanding'),
    'course-09': ('dep-03', 'Dependency Understanding'),
    'course-13': ('bug-01', 'Bug Analysis'),
    'course-14': ('bug-02', 'Bug Analysis'),
    'course-16': ('bug-03', 'Bug Analysis'),
    'code-01': ('code-01', 'Code Generation'),
    'code-02': ('code-02', 'Code Generation'),
    'code-03': ('code-03', 'Code Generation'),
    'course-05': ('ref-01', 'Refactoring'),
    'course-07': ('ref-02', 'Refactoring'),
    'course-08': ('ref-03', 'Refactoring'),
    'course-01': ('rag-01', 'RAG based Question'),
    'course-02': ('rag-02', 'RAG based Question'),
    'course-03': ('rag-03', 'RAG based Question'),
    'course-04': ('rag-04', 'RAG based Question'),
    'course-15': ('rag-05', 'RAG based Question'),
    'course-17': ('rag-06', 'RAG based Question'),
    'course-18': ('rag-07', 'RAG based Question'),
}

with open('data/evaluations/latest.json', 'r', encoding='utf-8') as f:
    latest = json.load(f)

# Load evaluation dataset to get questions text
with open('week4/evaluation_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)
dataset_by_id = {d['id']: d for d in dataset}

# Update each result row with category and new question_id and question text
for r in latest['results']:
    old_qid = r['question_id']
    if old_qid in mapping:
        new_qid, cat = mapping[old_qid]
        r['question_id'] = new_qid
        r['category'] = cat
        if new_qid in dataset_by_id:
            r['question'] = dataset_by_id[new_qid]['question']
            r['domain'] = dataset_by_id[new_qid]['domain']

models = latest['models']
cat_aggregates = {
    cat: {m: aggregate_category_model(cat, m, latest['results']) for m in models}
    for cat in SE_CATEGORIES
}
verdicts = evaluate_category_verdicts(cat_aggregates)

latest['categories'] = SE_CATEGORIES
latest['category_aggregates'] = cat_aggregates
latest['category_verdicts'] = verdicts

# Also update status.json to completed
eval_dir = Path("data/evaluations")
with open(eval_dir / "latest.json", "w", encoding="utf-8") as f:
    json.dump(latest, f, indent=2)

markdown_report = render_markdown(latest)
(eval_dir / "WEEK4_EVALUATION_REPORT.md").write_text(markdown_report, encoding="utf-8")

print(f"Successfully updated latest.json and WEEK4_EVALUATION_REPORT.md with {len(verdicts)} category verdicts.")
