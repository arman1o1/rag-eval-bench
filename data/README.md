# Data Directory

This directory holds the corpus documents and ground-truth evaluation data for the RAG Eval Bench pipeline.

## Directory Structure

```
data/
├── README.md                  ← You are here
├── corpus/                    ← Contract documents (PDF or TXT)
│   ├── contract_001.txt
│   ├── contract_002.txt
│   └── ...
└── ground_truth/              ← Ground-truth Q&A pairs
    └── ground_truth_qa.json
```

## Downloading the CUAD Dataset

The project uses the [CUAD (Contract Understanding Atticus Dataset)](https://www.atticusprojectai.org/cuad) — 510 commercial legal contracts with 13,000+ expert annotations across 41 clause types.

### Automatic Download

```bash
python scripts/download_cuad.py --output-dir data/corpus
```

Options:
- `--output-dir` — Target directory for contract files (default: `data/corpus`)
- `--max-contracts` — Limit number of contracts to download

### Manual Download

1. Visit [HuggingFace: theatticusproject/cuad-qa](https://huggingface.co/datasets/theatticusproject/cuad-qa)
2. Download the dataset files
3. Extract contract texts into `data/corpus/` as `.txt` files

## Generating Ground-Truth Q&A Pairs

```bash
python scripts/generate_qa.py \
    --corpus-dir data/corpus \
    --output-dir data/ground_truth \
    --num-pairs 50
```

Options:
- `--corpus-dir` — Directory containing contract documents
- `--output-dir` — Output directory for Q&A JSON files
- `--num-pairs` — Number of Q&A pairs to generate (default: 50)
- `--use-llm` — Also generate additional pairs using GPT-4o
- `--llm-pairs` — Number of LLM-generated pairs (default: 20)

## Ground-Truth JSON Schema

Each file in `ground_truth/` is a JSON array of evaluation items:

```json
[
  {
    "question": "Who are the parties involved in this contract?",
    "ground_truth_answer": "The parties are Acme Corp and Widget Inc.",
    "source_doc": "contract_001.txt",
    "source_page": 1,
    "source_section": "Parties",
    "context_texts": [
      "This Agreement is entered into between Acme Corp (\"Company\") and Widget Inc (\"Vendor\")..."
    ]
  }
]
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | ✅ | The evaluation question |
| `ground_truth_answer` | string | ✅ | Expected correct answer |
| `source_doc` | string | ❌ | Filename of the source document |
| `source_page` | integer | ❌ | Page number in source (0 if N/A) |
| `source_section` | string | ❌ | Section/clause type in source |
| `context_texts` | list[string] | ❌ | Relevant text passages from the source |

## Using Custom Data

You can use your own documents instead of CUAD:

1. Place documents (`.pdf`, `.txt`, `.docx`) in `data/corpus/`
2. Create a ground-truth JSON file following the schema above in `data/ground_truth/`
3. Run the pipeline as usual

> [!NOTE]
> The pipeline's document loader currently supports PDF (via PyMuPDF) and plain text files.
> For other formats, convert to PDF or TXT first.
