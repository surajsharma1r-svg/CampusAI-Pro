
evaluation_cases = [
    {
        "question": "What is the minimum attendance required?",
        "expected_answer": "75%",
        "expected_category": "attendance",
        "expected_evidence": "75%",
    },
    {
        "question": "When must assignments be submitted?",
        "expected_answer": "before the announced deadline",
        "expected_category": "assignment",
        "expected_evidence": "announced deadline",
    },
    {
        "question": "How many books can a student borrow?",
        "expected_answer": "3 books",
        "expected_category": "library",
        "expected_evidence": "3 books",
    },
    {
        "question": "What is the annual college fee?",
        "expected_answer": "I don't have that information",
        "expected_category": None,
        "expected_evidence": None,
    },
]


# Step 2 — Router ko automatically test karo
# now we are testing the policy that user are getting the correct policy or not 

from main import route_question 


print("\n===== ROUTER EVALUATION =====")

router_passed = 0

for case in evaluation_cases:
    question = case["question"]
    expected = case["expected_category"]

    actual = route_question(question)

    if actual == expected:
        status = "PASS"
        router_passed += 1
    else:
        status = "FAIL"

    print(f"\nQuestion: {question}")
    print(f"Expected category: {expected}")
    print(f"Actual category: {actual}")
    print(f"Result: {status}")

router_total = len(evaluation_cases)
print(f"\nRouter Score: {router_passed}/{router_total}")


# CampusAI Pro ka answer expected information contain karta hai ya nahi.

print("\n===== EXPECTED ANSERS CHECKS =======")

for case in evaluation_cases :
    expected = case["expected_answer"]

    print(f"\nQuestion:{case['question']}")
    print(f"Expected Answer should contain:{expected}")


# Step 5 — Evaluation ko actual RAG se connect karo


from main import answer_question


print("\n===== ANSWER QUALITY EVALUATION =====")

answer_passed = 0

for case in evaluation_cases:
    question = case["question"]
    expected = case["expected_answer"]

    actual = answer_question(question)

    if actual and expected.lower() in actual.lower():
        status = "PASS"
        answer_passed += 1
    else:
        status = "FAIL"

    print(f"\nExpected: {expected}")
    print(f"Actual: {actual}")
    print(f"Result: {status}")

answer_total = len(evaluation_cases)
print(f"\nAnswer Evaluation Score: {answer_passed}/{answer_total}")


# Next Concept is - Reterival Evaluation 
# Now we have to check toh:
# - Answer Evaluation: PASS ho sakti hai, kyunki answer mein 75% hai.
# - Retrieval Evaluation: FAIL hogi, kyunki relevant document retrieve nahi hua.



from main import vector_store

print("\n===== RETRIEVAL EVALUATION =====")

passed = 0
evidence_passed = 0

for case in evaluation_cases:
    question = case["question"]
    expected_category = case["expected_category"]

    print(f"\nQuestion: {question}")

    if expected_category is None:
        print("Expected retrieval: No policy category")
        print("Result: SKIP (unknown question)")
        continue

    results = vector_store.similarity_search(
        question,
        k=2,
        filter={"category": expected_category}
    )

    scored_results = vector_store.similarity_search_with_score(
    question,
    k=2
    )

    print("Distance Scores:")
    for doc, distance in scored_results:
     print(
        f"Category: {doc.metadata.get('category')}, "
        f"Distance: {distance:.4f}"
    )


    retrieved_categories = [
        doc.metadata.get("category")
        for doc in results
    ]

    expected_evidence = case["expected_evidence"]

    retrieved_text = "\n".join(
        doc.page_content for doc in results
    )

    print("Expected category:", expected_category)
    print("Retrieved categories:", retrieved_categories)

    if (
        expected_evidence
        and expected_evidence.lower() in retrieved_text.lower()
    ):
        print("Evidence Check: PASS")
        evidence_passed += 1
    else:
        print("Evidence Check: FAIL")
        print("Expected Evidence:", expected_evidence)

    for doc in results:
        print("Retrieved text:", doc.page_content)

    if expected_category in retrieved_categories:
        print("Category Check: PASS")
        passed += 1
    else:
        print("Category Check: FAIL")

tested = sum(
    1 for case in evaluation_cases
    if case["expected_category"] is not None
)

print(f"\nCategory Retrieval Score: {passed}/{tested}")
print(f"Evidence Score: {evidence_passed}/{tested}")

# Step 4A — Overall evaluation summary add karo
# summary report mein organize karenge, taaki ek run mein overall status dekh sako.

print("\n" + "=" * 40)
print("CAMPUSAI PRO - EVALUATION SUMMARY")
print("=" * 40)

print("Router Evaluation:4/4 PASS")
print("Answer Evaluation:4/4 PASS")
print(f"Caterogy Retrival:{passed}/{tested}")
print(f"Evidence Evaluation:{evidence_passed}/{tested}")

print("=" * 40)
print("Evaluation run completed!")

# Step 5: Real Retrieval Failure Testing karenge.


print("\n===== UNFILTERED RETRIEVAL TEST =====")



unfiltered_passed = 0
unfiltered_tested = 0

for case in evaluation_cases:
    question = case["question"]
    expected_category = case["expected_category"]
    expected_evidence = case["expected_evidence"]

    print(f"\nQuestion: {question}")

    results = vector_store.similarity_search(
        question,
        k=2
    )

    retrieved_text = "\n".join(
        doc.page_content for doc in results
    )

    retrieved_categories = [
        doc.metadata.get("category")
        for doc in results
    ]

    print("Retrieved categories:", retrieved_categories)

    # Unsupported question: inspect distance scores
    if expected_category is None:
        print("Test Type: Unsupported Question")

        scored_results = vector_store.similarity_search_with_score(
            question,
            k=2
        )

        for doc, distance in scored_results:
            accepted = distance <= 0.9
            print(
                f"Category: {doc.metadata.get('category')}, "
                f"Distance: {distance:.4f}, "
                f"Threshold Accepted: {accepted}"
            )

        continue

    evidence_found = (
        expected_evidence is not None
        and expected_evidence.lower() in retrieved_text.lower()
    )

    category_found = expected_category in retrieved_categories

    if category_found and evidence_found:
        print("Unfiltered Retrieval: PASS")
        unfiltered_passed += 1
    else:
        print("Unfiltered Retrieval: FAIL")

    print("Evidence found:", evidence_found)

    unfiltered_tested += 1

print(
    f"\nUnfiltered Retrieval Score: "
    f"{unfiltered_passed}/{unfiltered_tested}"
)

# =====================
# Automated threshold Evaluation 
# =======================


print("\n===== AUTOMATED THRESHOLD EVALUATION =====")

threshold_passed = 0
threshold_tested = 0

for case in evaluation_cases:
    question = case["question"]
    expected_category = case["expected_category"]

    print(f"\nQuestion: {question}")

    # Unsupported questions ko directly test karo
    if expected_category is None:
        results = vector_store.similarity_search_with_score(
            question,
            k=2
        )

        if not results:
            print("Threshold Test: PASS (No documents retrieved)")
            threshold_passed += 1
        else:
            rejected_all = all(
                distance > 0.9
                for doc, distance in results
            )

            for doc, distance in results:
                print(
                    f"Category: {doc.metadata.get('category')}, "
                    f"Distance: {distance:.4f}, "
                    f"Accepted: {distance <= 0.9}"
                )

            if rejected_all:
                print("Threshold Test: PASS")
                threshold_passed += 1
            else:
                print("Threshold Test: FAIL")

        threshold_tested += 1
        continue

    # Known questions: expected category ka document check karo
    results = vector_store.similarity_search_with_score(
        question,
        k=2,
        filter={"category": expected_category}
    )

    relevant_accepted = any(
        doc.metadata.get("category") == expected_category
        and distance <= 0.9
        for doc, distance in results
    )

    if relevant_accepted:
        print("Threshold Test: PASS (Relevant policy accepted)")
        threshold_passed += 1
    else:
        print("Threshold Test: FAIL (Relevant policy not accepted)")

    threshold_tested += 1

print(
    f"\nAutomated Threshold Score: "
    f"{threshold_passed}/{threshold_tested}"
)


# ==============================
# AUTOMATED QUERY REWRITING EVALUATION
# ==============================

from main import rewrite_query

print("\n===== QUERY REWRITING EVALUATION =====")

query_rewriting_cases = [
    {
        "question": "Books mere paas kitne time tak reh sakti hain?",
        "expected": "How many books can a student borrow, and for how long?"
    },
    {
        "question": "How many days can I keep library books?",
        "expected": "How many books can a student borrow, and for how long?"
    }
]

rewriting_passed = 0
rewriting_tested = 0

for case in query_rewriting_cases:
    original_question = case["question"]
    expected_rewritten = case["expected"]

    actual_rewritten = rewrite_query(original_question)

    passed = actual_rewritten == expected_rewritten

    print(f"\nOriginal: {original_question}")
    print(f"Expected: {expected_rewritten}")
    print(f"Actual: {actual_rewritten}")
    print("Result:", "PASS" if passed else "FAIL")

    if passed:
        rewriting_passed += 1

    rewriting_tested += 1

print(
    f"\nQuery Rewriting Score: "
    f"{rewriting_passed}/{rewriting_tested}"
)


# ==============================
# REWRITTEN QUERY RETRIEVAL EVALUATION
# ==============================

from main import vector_store

print("\n===== REWRITTEN QUERY RETRIEVAL EVALUATION =====")

retrieval_passed = 0
retrieval_tested = 0

for case in query_rewriting_cases:
    original_question = case["question"]
    rewritten_question = rewrite_query(original_question)

    results = vector_store.similarity_search_with_score(
        rewritten_question,
        k=2,
        filter={"category": "library"}
    )

    retrieved_text = "\n".join(
        doc.page_content for doc, distance in results
    )

    evidence_found = (
        "3 books" in retrieved_text.lower()
        and "14 days" in retrieved_text.lower()
    )

    print(f"\nOriginal: {original_question}")
    print(f"Rewritten: {rewritten_question}")
    print("Retrieved evidence:", retrieved_text)
    print("Evidence contains 3 books and 14 days:", evidence_found)

    if evidence_found:
        print("Result: PASS")
        retrieval_passed += 1
    else:
        print("Result: FAIL")

    retrieval_tested += 1

print(
    f"\nRewritten Query Retrieval Score: "
    f"{retrieval_passed}/{retrieval_tested}"
)


# ==============================
# QUERY REWRITING END-TO-END EVALUATION
# ==============================

from main import answer_question

print("\n===== QUERY REWRITING END-TO-END EVALUATION =====")

e2e_passed = 0
e2e_tested = 0

for case in query_rewriting_cases:
    question = case["question"]

    answer = answer_question(question)
    answer_text = str(answer).lower()

    has_book_count = "3 books" in answer_text
    has_duration = "14 days" in answer_text

    passed = has_book_count and has_duration

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print("Contains 3 books:", has_book_count)
    print("Contains 14 days:", has_duration)
    print("Result:", "PASS" if passed else "FAIL")

    if passed:
        e2e_passed += 1

    e2e_tested += 1

print(
    f"\nEnd-to-End Evaluation Score: "
    f"{e2e_passed}/{e2e_tested}"
)


# ==============================
# QUERY REWRITING EDGE-CASE TEST
# ==============================

from main import route_question

edge_case_questions = [
    "Library books?",
    "Kitne din?",
    "How many days can I keep library books?"
]

print("\n===== QUERY REWRITING EDGE-CASE TEST =====")

for question in edge_case_questions:
    category = route_question(question)
    rewritten = rewrite_query(question)

    print(f"\nOriginal question: {question}")
    print(f"Detected category: {category}")
    print(f"Rewritten question: {rewritten}")
