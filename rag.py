# from pathlib import Path

# import chromadb
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer


# BASE_DIR = Path(__file__).resolve().parent

# print("Loading embedding model...")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# client = chromadb.Client(
#     Settings(
#         chroma_api_impl="chromadb.api.segment.SegmentAPI",
#         anonymized_telemetry=False,
#     )
# )

# collection = client.get_or_create_collection(
#     name="knowledge_base",
#     embedding_function=None,
# )


# def load_knowledge():
#     file_path = BASE_DIR / "knowledge_base.txt"

#     with file_path.open("r", encoding="utf-8") as file:
#         documents = [
#             line.strip()
#             for line in file
#             if line.strip()
#         ]  

#     if not documents:
#         raise ValueError("knowledge_base.txt is empty.")

#     if collection.count() == 0:
#         embeddings = model.encode(documents).tolist()

#         collection.add(
#             ids=[str(i) for i in range(len(documents))],
#             documents=documents,
#             embeddings=embeddings,
#         )

#     print(f"Knowledge loaded: {collection.count()} chunks")


# def retrieve(question, top_k=3):
#     count = collection.count()

#     if count == 0:
#         return ""

#     question_embedding = model.encode([question]).tolist()

#     results = collection.query(
#         query_embeddings=question_embedding,
#         n_results=min(top_k, count),
#         include=["documents"],
#     )

#     return "\n\n".join(results["documents"][0])


# if __name__ == "__main__":
#     load_knowledge()

#     question = "What is RAG?"

#     print(f"\nQuestion: {question}")
#     print("\nRetrieved information:")
#     print(retrieve(question))

from pathlib import Path

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
document_vectors = []


def load_knowledge():
    global documents, document_vectors

    file_path = BASE_DIR / "knowledge_base.txt"

    with file_path.open("r", encoding="utf-8") as file:
        documents = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if not documents:
        raise ValueError("knowledge_base.txt is empty.")

    print("Creating embeddings...")

    document_vectors = model.encode(
        documents,
        normalize_embeddings=True,
    ).tolist()

    print(f"Knowledge loaded: {len(documents)} chunks")


def retrieve(question, top_k=3):
    if not document_vectors:
        raise ValueError("Call load_knowledge() first.")

    if top_k < 1:
        return ""

    question_vector = model.encode(
        [question],
        normalize_embeddings=True,
    )[0].tolist()

    scores = [
        sum(a * b for a, b in zip(question_vector, vector))
        for vector in document_vectors
    ]

    ranked_indices = sorted(
        range(len(documents)),
        key=lambda index: scores[index],
        reverse=True,
    )

    selected_indices = ranked_indices[:top_k]

    return "\n\n".join(
        documents[index]
        for index in selected_indices
    )


if __name__ == "__main__":
    load_knowledge()

    question = "What is RAG?"

    print(f"\nQuestion: {question}")
    print("\nRetrieved information:")
    print(retrieve(question))