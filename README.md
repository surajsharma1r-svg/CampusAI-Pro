# CampusAI Pro 🎓

**A Retrieval-Augmented Generation (RAG) assistant for answering college policy questions using local AI models.**

CampusAI Pro demonstrates how a college helpdesk assistant can retrieve relevant policy information, filter results by category, rewrite user queries, and generate context-grounded answers using a locally running language model.

## 🚀 Features

- **RAG pipeline:** Retrieves policy evidence before generating answers.
- **Semantic retrieval:** Uses Hugging Face sentence-transformer embeddings.
- **Vector database:** Stores policy chunks in ChromaDB.
- **Query rewriting:** Converts supported questions into clearer retrieval queries.
- **Category-based routing:** Identifies attendance, assignment, or library questions.
- **Metadata filtering:** Restricts retrieval to the selected policy category.
- **Similarity threshold:** Rejects retrieved results that exceed the configured distance threshold.
- **Fallback handling:** Responds when a question is unsupported or relevant evidence is unavailable.
- **Automated evaluation:** Includes checks for routing, answer content, retrieval, evidence, thresholds, and query rewriting.

## 🛠️ Tech Stack

- Python
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- `sentence-transformers/all-MiniLM-L6-v2`
- Ollama
- `llama3.2:3b` local language model

## 🧠 Architecture

1. Load college policy documents.
2. Separate policy sections and attach category metadata.
3. Split policies into chunks.
4. Generate embeddings for policy chunks.
5. Store the chunks in ChromaDB.
6. Route the user's question to a policy category.
7. Rewrite supported queries to improve retrieval.
8. Retrieve relevant chunks using semantic search and metadata filtering.
9. Apply the similarity-distance threshold.
10. Pass retrieved context to the local language model.
11. Return a grounded answer or fallback response.

## 📚 Knowledge Base

The current demonstration dataset covers three policy categories.

| Category | Example policy |
|---|---|
| Attendance | At least 75% attendance is required for semester examinations. |
| Assignments | Assignments must be submitted before the announced deadline. |
| Library | Students can borrow up to 3 books for 14 days. |

The dataset is intentionally small. Results demonstrate the pipeline on these sample policies and should not be interpreted as a comprehensive evaluation of a real college helpdesk.

## 💬 Example

**Question:** How many days can I keep library books?

**Retrieval goal:** Find the library policy and its borrowing duration.

**Expected policy evidence:** Students can borrow up to 3 books at a time for 14 days.

**Expected answer:** A student can borrow up to 3 books at a time for 14 days.

*Actual output may vary with the current code, dataset, and local model setup.*

## 🧪 Evaluation

`evaluation.py` contains checks covering:

- Question category routing
- Expected answer content
- Category-filtered retrieval
- Evidence presence
- Unfiltered retrieval
- Similarity-threshold behaviour
- Query rewriting
- Rewritten-query retrieval
- End-to-end query rewriting and answer content

Evaluation results are specific to the current sample dataset and test cases. They do not establish production-level accuracy or guarantee that hallucinations are eliminated.

## ⚙️ Setup

### Prerequisites

- Python 3.14 or another version compatible with the pinned dependencies
- Git
- Ollama installed
- The `llama3.2:3b` model available locally

Download the model:

```bash
ollama pull llama3.2:3b
```

Ensure Ollama is running before testing the language-model response path.

### 1. Clone the repository

```bash
git clone https://github.com/surajsharma1r-svg/CampusAI-Pro.git
cd CampusAI-Pro
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the local model

```bash
ollama list
```

Confirm that `llama3.2:3b` is listed.

### 5. Run the current evaluation script

```bash
python evaluation.py
```

**Implementation note:** The current scripts contain top-level initialization and evaluation code. Importing functions from `main.py` can execute that module's top-level statements again, including Chroma collection initialization and demonstration tests. Review this behaviour before using the project as a production service.

## 🔮 Future Improvements

- Separate application logic from evaluation scripts.
- Add a clean application entry point and reusable test functions.
- Avoid resetting the Chroma collection on every application startup.
- Expand the knowledge base with verified college policies.
- Add ambiguous-query and unsupported-question test cases.
- Evaluate retrieval quality and answer faithfulness on a larger dataset.
- Add a user-friendly web interface and conversation context.

## 👨‍💻 Author

**Suraj Sharma**

Aspiring AI / ML and Generative AI Engineer

---

*Built as a practical project exploring RAG, semantic retrieval, vector databases, metadata filtering, query rewriting, and evaluation.*
