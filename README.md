# CampusAI Pro 🎓

**An AI-powered college policy assistant built with Retrieval-Augmented Generation (RAG).**

CampusAI Pro helps students find answers to college policy questions using a local knowledge base, semantic search, metadata filtering, and query rewriting.

## 🚀 Features

- **Retrieval-Augmented Generation (RAG):** Retrieves relevant college policy information before generating an answer.
- **Semantic Search:** Uses sentence-transformer embeddings to find relevant policy content.
- **Chroma Vector Database:** Stores and retrieves embedded policy chunks.
- **Query Rewriting:** Converts supported natural-language questions into clearer retrieval queries.
- **Category-Based Routing:** Routes questions to attendance, assignment, or library policies.
- **Metadata Filtering:** Restricts retrieval to the detected policy category.
- **Similarity Threshold:** Filters out retrieved chunks that do not meet the configured distance threshold.
- **Fallback Handling:** Returns a fallback response when a question is unsupported or relevant evidence is unavailable.
- **Automated Evaluation:** Tests routing, retrieval, evidence, threshold filtering, and query rewriting.

## 🛠️ Tech Stack

- Python
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- `sentence-transformers/all-MiniLM-L6-v2`
- Ollama with a locally running `llama3.2:3b` model

## 🧠 How It Works

1. A student submits a question.
2. The question router identifies the relevant policy category.
3. Query rewriting improves supported questions for retrieval.
4. The embedding model represents the query as a vector.
5. Chroma searches the relevant category for policy evidence.
6. Similarity-threshold filtering removes unsuitable results.
7. The retrieved evidence is passed to the language model.
8. The assistant generates an answer grounded in the retrieved policy, or returns a fallback when appropriate.

## 📚 Current Knowledge Base

The current demonstration dataset contains three policy categories:

| Category | Example policy |
|---|---|
| Attendance | Minimum 75% attendance for semester examinations |
| Assignments | Submit assignments before the announced deadline |
| Library | Borrow up to 3 books for 14 days |

The current dataset is intentionally small and is intended for demonstrating and evaluating the RAG pipeline.

## 💬 Example

**Question:** How many days can I keep library books?

**Rewritten query:** How many books can a student borrow, and for how long?

**Retrieved evidence:** Students can borrow up to 3 books at a time for 14 days.

**Answer:** A student can borrow up to 3 books at a time for 14 days.

## 🧪 Evaluation

The project includes automated checks for:

- Question category routing
- Expected answer content
- Category-filtered retrieval
- Evidence presence
- Similarity-threshold behaviour
- Query rewriting
- Rewritten-query retrieval
- End-to-end query rewriting and answer generation

Evaluation results depend on the current dataset, test cases, and local model configuration.

## ⚙️ Setup

### Prerequisites

- Python compatible with the installed project dependencies
- Ollama installed and running
- The `llama3.2:3b` model downloaded locally

Pull the model if it is not already available:

```bash
ollama pull llama3.2:3b
```

### Installation

Clone the repository and enter its directory:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "CampusAI Pro"
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project's dependencies:

```bash
pip install -r requirements.txt
```

Ensure Ollama is running and the required model is available before running the application or evaluations.

> Note: Confirm the actual entry point and dependency file in the repository before publishing these commands.

## 🔮 Future Improvements

- Expand the knowledge base with more verified college policies.
- Add more ambiguous-query and unsupported-question evaluations.
- Improve conversation context and follow-up question handling.
- Add a user-friendly web interface.
- Measure retrieval quality and answer faithfulness on a larger test dataset.

## 👨‍💻 Author

**Suraj Sharma**

Aspiring AI / ML and Generative AI Engineer

---

*Built as a practical learning project exploring RAG, semantic retrieval, vector databases, query rewriting, and evaluation.*
