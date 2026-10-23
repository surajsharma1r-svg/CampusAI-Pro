
from pathlib import Path
import re

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate



# ==========================================
# 1. LOAD COLLEGE POLICY
# ==========================================

file_path = Path("data/college_policy.txt")

loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()

print("Total source documents:", len(documents))
raw_text = documents[0].page_content

print("\nOriginal College Policy:")
print(raw_text)


# ==========================================
# 2. SEPARATE POLICIES + ADD METADATA
# ==========================================

policy_pattern = (
    r"(?m)(?=^\s*\d+\.\s*"
    r"(?:Attendance|Assignment|Library)\s+Policy:)"
)

policy_sections = re.split(policy_pattern, raw_text)

category_map = {
    "attendance": "attendance",
    "assignment": "assignment",
    "library": "library",
}

policy_documents = []

for section in policy_sections:
    section = section.strip()

    if not section:
        continue

    category = None

    for policy_name, category_value in category_map.items():
        if re.search(
            rf"(?i)\b{policy_name}\s+Policy:",
            section
        ):
            category = category_value
            break

    # Skip the general document title
    if category is None:
        continue

    policy_documents.append(
        Document(
            page_content=section,
            metadata={
                "category": category,
                "source": "college_policy.txt"
            }
        )
    )

print("\n===== POLICY DOCUMENTS =====")

for doc in policy_documents:
    print("\nCategory:", doc.metadata["category"])
    print("Content:", doc.page_content)


# ==========================================
# 3. SPLIT EACH POLICY INTO CHUNKS
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

chunks = splitter.split_documents(policy_documents)

print("\nTotal Policy Chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}")
    print("Metadata:", chunk.metadata)
    print(chunk.page_content)


# ==========================================
# 4. GENERATE EMBEDDINGS
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectors = embeddings.embed_documents(
    [chunk.page_content for chunk in chunks]
)

print("\nTotal Embeddings:", len(vectors))
print("Vector Dimension:", len(vectors[0]))
print("First 5 Values:", vectors[0][:5])


# ==========================================
# 5. CHROMA DATABASE WITH METADATA
# ==========================================

# Development only: reset old collection
vector_store = Chroma(
    collection_name="college_policies",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

vector_store.delete_collection()

# Create a clean collection
vector_store = Chroma(
    collection_name="college_policies",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

document_ids = [
    f"college-policy-{chunk.metadata['category']}-{i}"
    for i, chunk in enumerate(chunks)
]

vector_store.add_documents(
    documents=chunks,
    ids=document_ids
)

print("\nChroma Database Ready!")
print("Stored Chunks:", vector_store._collection.count())


# ==========================================
# 6. LLM + GROUNDED ANSWER PROMPT
# ==========================================

SIMILARITY_THRESHOLD = 0.9

FALLBACK_ANSWER = (
    "I don't have that information in the college policy."
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


prompt = ChatPromptTemplate.from_template("""
You are CampusAI Pro, a college policy assistant.

Instructions:
1. Answer using ONLY the provided context.
2. Include ALL relevant facts from the context that answer the question.
3. Do not omit important numbers, limits, durations, or deadlines.
4. Do not guess or add information from outside the context.
5. If the answer is not present in the context, say:
"I don't have that information in the college policy."
6. Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
""")

chain = prompt | llm



# ==========================================
# 7. METADATA FILTERING + THRESHOLD
# ==========================================

def answer_question(question, category):

    print(f"\n===== QUESTION: {question} =====")
    print("Metadata category:", category)

    # Search only in the selected category
    scored_results = vector_store.similarity_search_with_score(
        question,
        k=2,
        filter={"category": category}
    )

    print("\n===== FILTERED RETRIEVAL =====")

    for i, (doc, distance) in enumerate(
        scored_results, start=1
    ):
        print(f"\nResult {i} | Distance: {distance:.4f}")
        print("Metadata:", doc.metadata)
        print(doc.page_content)

    # Apply the existing distance threshold
    relevant_docs = [
        doc
        for doc, distance in scored_results
        if distance <= SIMILARITY_THRESHOLD
    ]

    print("\nThreshold:", SIMILARITY_THRESHOLD)
    print("Accepted chunks:", len(relevant_docs))

    # Do not call the LLM if no chunks pass
    if not relevant_docs:
        print("\n===== CAMPUSAI PRO ANSWER =====")
        print(FALLBACK_ANSWER)
        return

    context = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    response = chain.invoke({
        "context": context,
        "question": question
    })

    print("\n===== CAMPUSAI PRO ANSWER =====")
    print(response.content)


# ======================
# Automatic Query Routing 
# Automatically routing labels will help to gusses the automatics with examples 

# ==============================
# AUTOMATIC QUESTION ROUTING
# ==============================

def route_question(question):
    q = question.lower().strip()

    # Attendance-related questions
    if any(word in q for word in [
        "attendance",
        "present",
        "absent",
        "semester exam",
        "exam eligibility"
    ]):
        return "attendance"

    # Assignment-related questions
    if any(word in q for word in [
        "assignment",
        "assigment",
        "deadline",
        "submit",
        "submission",
        "due date",
        "when is it due",
        "when is the deadline"
    ]):
        return "assignment"

    # Library-related questions
    if any(word in q for word in [
        "library",
        "book",
        "books",
        "borrow"
    ]):
        return "library"

    # Unknown or ambiguous question
    return None



# ==============================
# QUERY REWRITING
# ==============================

def rewrite_query(question):
    question = question.strip()
    normalized_question = question.lower().strip(" ?.!")

    # Short library queries
    if normalized_question in [
        "library books",
        "books",
        "library",
        "books library"
    ]:
        return "What are the college library book borrowing rules?"

    # Library quantity and duration questions
    library_keywords = [
        "books",
        "book",
        "library",
        "borrow"
    ]

    if any(word in normalized_question for word in library_keywords):
        duration_keywords = [
            "how many books",
            "how many days",
            "how long",
            "kitne time",
            "kitne din",
            "borrow",
            "reh sakti",
            "keep library books"
        ]

        if any(word in normalized_question for word in duration_keywords):
            return "How many books can a student borrow, and for how long?"

        if "book" in normalized_question or "library" in normalized_question:
            return "What are the college library book borrowing rules?"

    # Other questions remain unchanged
    return question


# ==============================
# MAIN RAG QUESTION-ANSWER PIPELINE
# ==============================

def answer_question(question):
    print(f"\n===== QUESTION: {question} =====")

    # Step 1: Detect the question category
    category = route_question(question)

    if category is None:
        print("Detected category: Unknown")
        print("\n===== CAMPUSAI PRO ANSWER =====")
        print(FALLBACK_ANSWER)
        return FALLBACK_ANSWER

    print("Auto-detected category:", category)

    # Step 2: Rewrite the question
    rewritten_question = rewrite_query(question)

    print("Original Question:", question)
    print("Rewritten Question:", rewritten_question)

    # Step 3: Retrieve documents from the detected category
    scored_results = vector_store.similarity_search_with_score(
        rewritten_question,
        k=2,
        filter={"category": category}
    )

    print("\n===== FILTERED RETRIEVAL =====")

    for i, (doc, distance) in enumerate(scored_results, start=1):
        print(f"\nResult {i} | Distance: {distance:.4f}")
        print("Metadata:", doc.metadata)
        print(doc.page_content)

    # Step 4: Apply the similarity threshold
    relevant_docs = [
        doc
        for doc, distance in scored_results
        if distance <= SIMILARITY_THRESHOLD
    ]

    print("\nThreshold:", SIMILARITY_THRESHOLD)
    print("Accepted chunks:", len(relevant_docs))

    if not relevant_docs:
        print("\n===== CAMPUSAI PRO ANSWER =====")
        print(FALLBACK_ANSWER)
        return FALLBACK_ANSWER

    # Step 5: Build context from accepted documents
    context = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    print("\n===== CONTEXT SENT TO LLM =====")
    print(context)

    print("\n===== QUESTION SENT TO LLM =====")
    print(rewritten_question)

    # Step 6: Generate the answer using the existing chain
    response = chain.invoke({
        "context": context,
        "question": rewritten_question
    })

    print("\n===== CAMPUSAI PRO ANSWER =====")
    print(response.content)

    return response.content


# ==============================
# ORIGINAL VS REWRITTEN RETRIEVAL
# ==============================

original_question = "How many days can I keep library books?"
rewritten_question = rewrite_query(original_question)

print("\n===== ORIGINAL QUERY RETRIEVAL =====")
print("Original query:", original_question)

original_results = vector_store.similarity_search_with_score(
    original_question,
    k=2
)

for doc, distance in original_results:
    print("Category:", doc.metadata.get("category"))
    print("Distance:", round(distance, 4))
    print("Content:", doc.page_content)
    print("-" * 40)


print("\n===== REWRITTEN QUERY RETRIEVAL =====")
print("Rewritten query:", rewritten_question)

rewritten_results = vector_store.similarity_search_with_score(
    rewritten_question,
    k=2
)

for doc, distance in rewritten_results:
    print("Category:", doc.metadata.get("category"))
    print("Distance:", round(distance, 4))
    print("Content:", doc.page_content)
    print("-" * 40)



# ==============================
# QUERY REWRITING DISTANCE EVALUATION
# ==============================

from main import vector_store, rewrite_query

print("\n===== QUERY REWRITING DISTANCE EVALUATION =====")

original_question = "How many days can I keep library books?"
rewritten_question = rewrite_query(original_question)

original_results = vector_store.similarity_search_with_score(
    original_question,
    k=1,
    filter={"category": "library"}
)

rewritten_results = vector_store.similarity_search_with_score(
    rewritten_question,
    k=1,
    filter={"category": "library"}
)

original_distance = original_results[0][1]
rewritten_distance = rewritten_results[0][1]

print("Original question:", original_question)
print("Rewritten question:", rewritten_question)
print(f"Original distance: {original_distance:.4f}")
print(f"Rewritten distance: {rewritten_distance:.4f}")

if rewritten_distance < original_distance:
    print("Result: PASS - Rewriting improved the distance score")
else:
    print("Result: CHECK - Distance did not improve")


from main import answer_question

print("\n===== QUERY REWRITING FINAL TEST =====")

question = "How many days can I keep library books?"
answer = answer_question(question)

answer_text = str(answer).lower()

has_book_count = "3 books" in answer_text
has_duration = "14 days" in answer_text

print("Contains 3 books:", has_book_count)
print("Contains 14 days:", has_duration)

if has_book_count and has_duration:
    print("FINAL RESULT: PASS")
else:
    print("FINAL RESULT: FAIL")

    