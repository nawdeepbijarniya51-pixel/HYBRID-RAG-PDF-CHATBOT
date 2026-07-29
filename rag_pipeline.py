"""
Core RAG pipeline: PDF ingestion, hybrid retrieval, query routing (NoChange /
Rephrase / MultiQuery), and answer generation — with real staged progress
reporting for both ingestion (parsing/chunking/embedding/storing) and
answering (routing/branching/retrieving/reranking/answering), plus real
retrieved-source metadata for the UI's sources panel.
"""

import os
import math
import hashlib
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import trim_messages

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://1cc7e4ad-58ad-457a-bba4-c1b93e39aaf7.us-west-2-0.aws.cloud.qdrant.io",
)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

REQUIRED_ENV_VARS = ["MISTRAL_API_KEY", "COHERE_API_KEY", "QDRANT_API_KEY"]

# Cohere sunset rerank-multilingual-v2.0 on 2025-12-01; rerank-v3.5 is itself
# being phased out (deprecated 2026-07-01, fully redirected 2026-08-01), so we
# go straight to the current generation.
COHERE_RERANK_MODEL = "rerank-v4.0-fast"

ProgressFn = Optional[Callable[[str, int, str], None]]


def check_env_vars() -> List[str]:
    return [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]


# =====MODELS / EMBEDDINGS (created once, reused across sessions)=====
model = ChatMistralAI(model="mistral-large-latest", temperature=0.3)
router = ChatMistralAI(model="ministral-8b-latest", temperature=0.2)
conditional_model = ChatMistralAI(model="mistral-small-latest", temperature=0.2)

dense_embeddings = MistralAIEmbeddings(model="mistral-embed")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def get_model_config() -> dict:
    """Real model identifiers actually in use — for the UI's 'Model configuration' panel."""
    return {
        "embedding_model": getattr(dense_embeddings, "model", "mistral-embed"),
        "llm_model": getattr(model, "model", "mistral-large-latest"),
        "router_model": getattr(router, "model", "ministral-8b-latest"),
        "rerank_model": COHERE_RERANK_MODEL,
    }


# =====PER-PDF VECTOR STORE=====

def pdf_hash(pdf_path: str) -> str:
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def get_or_create_vector_store(
    pdf_path: str, collection_name: str, on_progress: ProgressFn = None
) -> Tuple[QdrantVectorStore, int, int]:
    """Returns (vector_store, num_pages, num_chunks). Always parses + splits
    (cheap, CPU-only) so the caller gets real page/chunk counts either way;
    only embeds + upserts into Qdrant if the collection doesn't exist yet.
    Reports real, incremental progress via on_progress(stage, pct, message)."""

    def report(stage: str, pct: int, msg: str):
        if on_progress:
            on_progress(stage, pct, msg)

    report("parsing", 5, "Reading PDF pages")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    num_pages = len(docs)

    report("chunking", 15, f"Splitting {num_pages} page(s) into chunks")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitted_docs = splitter.split_documents(docs)
    num_chunks = len(splitted_docs)
    report("chunking", 25, f"Created {num_chunks} chunks")

    if qdrant_client.collection_exists(collection_name):
        report("storing", 90, "Reusing existing vector index for this document")
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        report("done", 100, "Index ready")
        return vector_store, num_pages, num_chunks

    # Embed + upsert in ~10 batches so progress reflects real work done,
    # not a fixed timer.
    batch_size = max(1, math.ceil(num_chunks / 10)) if num_chunks else 1
    vector_store: Optional[QdrantVectorStore] = None

    for i in range(0, num_chunks, batch_size):
        batch = splitted_docs[i : i + batch_size]
        if vector_store is None:
            try:
                vector_store = QdrantVectorStore.from_documents(
                    batch,
                    embedding=dense_embeddings,
                    sparse_embedding=sparse_embeddings,
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    collection_name=collection_name,
                    retrieval_mode=RetrievalMode.HYBRID,
                )
            except Exception:
                # Narrow race: another request created this exact (content-identical)
                # collection between our exists-check and here.
                if qdrant_client.collection_exists(collection_name):
                    vector_store = QdrantVectorStore(
                        client=qdrant_client,
                        collection_name=collection_name,
                        embedding=dense_embeddings,
                        sparse_embedding=sparse_embeddings,
                        retrieval_mode=RetrievalMode.HYBRID,
                    )
                else:
                    raise
        else:
            vector_store.add_documents(batch)

        done = min(i + batch_size, num_chunks)
        pct = 25 + int(65 * done / max(num_chunks, 1))
        report("embedding", pct, f"Embedded {done}/{num_chunks} chunks")

    report("storing", 95, "Finalizing vector index")
    report("done", 100, "Index ready")
    return vector_store, num_pages, num_chunks


@dataclass
class RetrieverBundle:
    """Base retriever and reranker kept separate (not just the combined
    ContextualCompressionRetriever) so answer_query can report real
    'retrieving' vs 'reranking' stages instead of one opaque call."""
    base_retriever: object
    compressor: object
    combined: ContextualCompressionRetriever


def build_retriever(vector_store: QdrantVectorStore) -> RetrieverBundle:
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    compressor = CohereRerank(model=COHERE_RERANK_MODEL, top_n=5)
    combined = ContextualCompressionRetriever(
        base_retriever=base_retriever, base_compressor=compressor
    )
    return RetrieverBundle(base_retriever=base_retriever, compressor=compressor, combined=combined)


# ==========ROUTING CHAIN=============

class RouteOutput(BaseModel):
    label: Literal["NoChange", "Rephrase", "MultiQuery"] = Field(
        description="The routing decision for the query"
    )


routing_parser = PydanticOutputParser(pydantic_object=RouteOutput)

routing_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a query routing assistant for a PDF-based Retrieval-Augmented Generation (RAG) chatbot.
Your ONLY job is to analyze the user's question and classify it into exactly one of three categories:

1. "NoChange" — The query is already clear, specific, and well-formed.
2. "Rephrase" — The query is vague, too short, poorly worded, or missing important keywords.
3. "MultiQuery" — The query contains multiple sub-questions or compares two or more things.

Rules:
- Always respond with ONLY a valid JSON object, no explanation, no extra text.
- The JSON must follow this exact schema:
  {{"label": "<NoChange|Rephrase|MultiQuery>"}}
- Do not answer the user's question. Do not add commentary. Only output the JSON.

Examples:

User: what is the termination clause
Response: {{"label": "NoChange"}}

User: tell me about that thing in section 2
Response: {{"label": "Rephrase"}}

User: compare the warranty terms and the refund policy, and also tell me who is eligible for support
Response: {{"label": "MultiQuery"}}

these rules should be followed:

{format_instructions}
""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{users_query}"),
    ]
).partial(format_instructions=routing_parser.get_format_instructions())

routing_chain = routing_prompt | router | routing_parser


# =======CONDITIONAL CHAINS=========

class ConditionalChainOutput(BaseModel):
    Query: List[str] = Field(description="List of queries to be processed by the RAG chain")


conditional_parser = PydanticOutputParser(pydantic_object=ConditionalChainOutput)

rephrase_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a query rewriting assistant for a PDF-based Retrieval-Augmented Generation (RAG) system.
Rewrite the user's query into a single clearer, more specific, keyword-rich version.
Do NOT answer the query. Only rewrite it. Keep it to one sentence.

Output ONLY a valid JSON object matching this schema, and nothing else:
{format_instructions}

The "Query" list must contain EXACTLY ONE string.

Examples:

Query: refund policy?
Output: {{"Query": ["What is the refund policy described in the document?"]}}

Query: tell me about that thing in section 2
Output: {{"Query": ["What are the key points discussed in Section 2 of the document?"]}}
""",
        ),
        ("human", "{users_query}"),
    ]
).partial(format_instructions=conditional_parser.get_format_instructions())

rephrase_chain = rephrase_prompt | conditional_model | conditional_parser

multiquery_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert at generating search queries for Retrieval-Augmented Generation (RAG).
Generate 2-4 diverse search queries that preserve the original intent using different wording,
synonyms, and levels of specificity. Do NOT answer the question. Do NOT explain your reasoning.

Output ONLY a valid JSON object matching this schema, and nothing else:
{format_instructions}

The JSON key must be "Query" (a list of strings).

Example:

Question: Compare the warranty terms and the refund policy, and also tell me who is eligible for support.
Output: {{"Query": ["What are the warranty terms described in the document?", "What is the refund policy described in the document?", "Who is eligible for support according to the document?"]}}
""",
        ),
        ("human", "User Question:\n{users_query}"),
    ]
).partial(format_instructions=conditional_parser.get_format_instructions())

multiquery_chain = multiquery_prompt | conditional_model | conditional_parser


# ==========RAG ANSWER CHAIN=============

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a precise, factual assistant that answers questions using ONLY the provided context.

RULES (follow strictly):
1. Answer using ONLY information found in the "Context" section below. Do not use outside knowledge, assumptions, or prior training data.
2. If the context does not contain enough information to answer the question, respond exactly with:
   "I don't have enough information in the provided context to answer this question."
   Do NOT guess, infer beyond what's stated, or fabricate details.
3. If the context is partially relevant, answer only the part you can support, and explicitly state what is missing.
4. Do not mention "the context" explicitly in your final answer. Just answer naturally.
5. Never invent sources, numbers, names, dates, or citations that aren't explicitly present in the context.
6. If multiple context chunks conflict, point out the discrepancy instead of picking one arbitrarily.

FORMATTING RULES:
- Use Markdown formatting.
- Start with a short, direct answer (1-2 sentences).
- Follow with supporting details as bullet points or a short numbered list if applicable.
- Use **bold** for key terms, numbers, or entities.
- Use a table if the answer involves comparing multiple items or structured data.
- Keep the tone neutral, clear, and concise.
- If code is part of the answer, use proper code blocks with language tags.

Context:
{context}
""",
        ),
        ("human", "{question}"),
    ]
)

rag_chain = rag_prompt | model


# ==========HELPERS=============

trimmer = trim_messages(
    max_tokens=7000,
    strategy="last",
    token_counter=lambda messages: sum(len(m.content.split()) for m in messages),
    include_system=True,
    allow_partial=False,
    start_on="human",
)


def answer_query(
    user_input: str,
    retriever_bundle: RetrieverBundle,
    chat_history: list,
    on_stage: ProgressFn = None,
) -> Tuple[str, List[dict]]:
    """Runs routing -> (rephrase/multiquery/nochange) -> per-query retrieval ->
    rerank -> answer generation, reporting real stages as it goes. Returns
    (final_markdown_answer, sources) where sources is a list of
    {"page": int|None, "score": float|None, "snippet": str}."""

    def report(stage: str, pct: int, msg: str):
        if on_stage:
            on_stage(stage, pct, msg)

    report("routing", 8, "Classifying your question")
    trimmed_history = trimmer.invoke(chat_history)
    routing_result = routing_chain.invoke(
        {"users_query": user_input, "chat_history": trimmed_history}
    )

    report("branching", 20, f"Routed as {routing_result.label}")
    if routing_result.label == "NoChange":
        branch_result = ConditionalChainOutput(Query=[user_input])
    elif routing_result.label == "Rephrase":
        branch_result = rephrase_chain.invoke({"users_query": user_input})
    else:
        branch_result = multiquery_chain.invoke({"users_query": user_input})

    queries_to_process = branch_result.Query
    n = len(queries_to_process)
    report("branching", 30, f"Prepared {n} search quer{'y' if n == 1 else 'ies'}")

    collected_answers: List[str] = []
    collected_sources: List[dict] = []

    for idx, query in enumerate(queries_to_process):
        base_pct = 35 + int(50 * idx / max(n, 1))

        report("retrieving", base_pct, f"Searching document for: {query[:70]}")
        base_docs = retriever_bundle.base_retriever.invoke(query)

        report("reranking", min(base_pct + 8, 95), f"Reranking {len(base_docs)} candidate passages")
        reranked_docs = retriever_bundle.compressor.compress_documents(base_docs, query)

        report("answering", min(base_pct + 14, 98), "Generating answer from retrieved context")
        context = "\n\n".join(d.page_content for d in reranked_docs)
        answer = rag_chain.invoke({"context": context, "question": query})
        collected_answers.append(answer.content)

        for d in reranked_docs:
            raw_page = d.metadata.get("page")
            page = raw_page + 1 if isinstance(raw_page, int) else raw_page
            collected_sources.append(
                {
                    "page": page,
                    "score": d.metadata.get("relevance_score"),
                    "snippet": d.page_content[:220],
                }
            )

    report("done", 100, "Answer ready")
    return "\n\n---\n\n".join(collected_answers), collected_sources
