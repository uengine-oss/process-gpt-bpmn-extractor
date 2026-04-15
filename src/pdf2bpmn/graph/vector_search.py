"""Vector embedding and similarity search for entity matching."""
from typing import Optional
import numpy as np
import httpx

from ..config import Config
from .neo4j_client import Neo4jClient


class VectorSearch:
    """Vector-based similarity search for entity matching and deduplication."""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.merge_threshold = Config.SIMILARITY_MERGE_THRESHOLD
        self.review_threshold = Config.SIMILARITY_REVIEW_THRESHOLD
    
    def _embedding_endpoint(self) -> str:
        base_url = (Config.EMBEDDING_BASE_URL or "").rstrip("/")
        if not base_url:
            raise ValueError("EMBEDDING_BASE_URL 또는 LLM_BASE_URL이 설정되지 않았습니다.")
        return f"{base_url}/embeddings"
    
    def _request_embeddings(self, inputs: list[str]) -> list[list[float]]:
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        headers = {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": Config.EMBEDDING_MODEL,
            "input": inputs,
        }
        
        with httpx.Client(timeout=Config.EMBEDDING_TIMEOUT_SEC) as client:
            response = client.post(
                self._embedding_endpoint(),
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        
        data = body.get("data")
        if isinstance(data, list):
            embeddings = [
                item.get("embedding")
                for item in data
                if isinstance(item, dict) and isinstance(item.get("embedding"), list)
            ]
            if len(embeddings) == len(inputs):
                return embeddings
        
        embeddings = body.get("embeddings")
        if isinstance(embeddings, list) and len(embeddings) == len(inputs):
            return embeddings
        
        raise ValueError(
            "No embedding data received. "
            f"model={Config.EMBEDDING_MODEL}, response_keys={list(body.keys())}, "
            f"response_preview={str(body)[:500]}"
        )
    
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text."""
        return self._request_embeddings([text])[0]
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        return self._request_embeddings(texts)
    
    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def find_similar_chunks(
        self, 
        embedding: list[float], 
        top_k: int = 5
    ) -> list[dict]:
        """Find similar reference chunks using vector index."""
        query = """
        CALL db.index.vector.queryNodes('chunk_embedding_idx', $top_k, $embedding)
        YIELD node, score
        RETURN node {.*, similarity: score} as chunk
        """
        with self.neo4j.session() as session:
            try:
                result = session.run(query, {
                    "embedding": embedding,
                    "top_k": top_k
                })
                return [record["chunk"] for record in result]
            except Exception as e:
                print(f"Vector search warning: {e}")
                return []
    
    def find_similar_entity(
        self, 
        entity_type: str, 
        name: str, 
        description: str = ""
    ) -> tuple[Optional[dict], float, str]:
        """
        Find similar existing entity in graph.
        
        Returns:
            tuple: (matching_entity, similarity_score, action)
            action: "merge" | "review" | "create" | "conflict"
        """
        # Create combined text for embedding
        search_text = f"{name}. {description}" if description else name
        
        # First try full-text search (faster)
        text_matches = self.neo4j.search_similar_by_name(
            entity_type, name, limit=10
        )
        
        if not text_matches:
            return None, 0.0, "create"
        
        # Generate embedding for semantic comparison
        query_embedding = self.embed_text(search_text)
        
        best_match = None
        best_score = 0.0
        
        for match in text_matches:
            # Generate or retrieve embedding for candidate
            match_text = f"{match.get('name', '')}. {match.get('description', '')}"
            match_embedding = self.embed_text(match_text)
            
            similarity = self.cosine_similarity(query_embedding, match_embedding)
            
            if similarity > best_score:
                best_score = similarity
                best_match = match
        
        # Determine action based on similarity threshold
        if best_score >= self.merge_threshold:
            return best_match, best_score, "merge"
        elif best_score >= self.review_threshold:
            return best_match, best_score, "review"
        else:
            return None, best_score, "create"
    
    def find_or_create_entity(
        self,
        entity_type: str,
        entity_data: dict,
        auto_merge: bool = True
    ) -> tuple[str, str]:
        """
        Find existing entity or prepare for creation.
        
        Returns:
            tuple: (entity_id, action_taken)
            action_taken: "merged" | "created" | "pending_review"
        """
        name = entity_data.get("name", "")
        description = entity_data.get("description", "")
        
        match, score, action = self.find_similar_entity(
            entity_type, name, description
        )
        
        if action == "merge" and auto_merge:
            # Return existing entity ID
            id_field = f"{entity_type.lower()}_id"
            if entity_type == "Process":
                id_field = "proc_id"
            return match.get(id_field, match.get("id")), "merged"
        
        elif action == "review":
            # Needs human review
            return None, "pending_review"
        
        else:
            # Create new entity
            return None, "create"
    
    def update_chunk_embedding(self, chunk_id: str, embedding: list[float]):
        """Update embedding for a reference chunk."""
        query = """
        MATCH (c:ReferenceChunk {chunk_id: $chunk_id})
        SET c.embedding = $embedding
        RETURN c.chunk_id
        """
        with self.neo4j.session() as session:
            session.run(query, {
                "chunk_id": chunk_id,
                "embedding": embedding
            })
    
    def batch_embed_chunks(self, chunks: list) -> list:
        """Embed multiple chunks and update in Neo4j."""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            self.update_chunk_embedding(chunk.chunk_id, embedding)
        
        return chunks
    
    def semantic_search_entities(
        self, 
        query: str, 
        entity_types: list[str] = None,
        top_k: int = 10
    ) -> list[dict]:
        """
        Semantic search across multiple entity types.
        """
        if entity_types is None:
            entity_types = ["Process", "Task", "Role", "Skill"]
        
        query_embedding = self.embed_text(query)
        results = []
        
        # Search in reference chunks first
        similar_chunks = self.find_similar_chunks(query_embedding, top_k * 2)
        
        # Get entities linked to these chunks
        for chunk in similar_chunks:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            
            # Find entities supported by this chunk
            entity_query = """
            MATCH (e)-[:SUPPORTED_BY]->(c:ReferenceChunk {chunk_id: $chunk_id})
            RETURN labels(e)[0] as entity_type, e {.*} as entity
            """
            with self.neo4j.session() as session:
                result = session.run(entity_query, {"chunk_id": chunk_id})
                for record in result:
                    if record["entity_type"] in entity_types:
                        results.append({
                            "entity_type": record["entity_type"],
                            "entity": record["entity"],
                            "chunk_similarity": chunk.get("similarity", 0)
                        })
        
        # Deduplicate and sort by similarity
        seen = set()
        unique_results = []
        for r in sorted(results, key=lambda x: x["chunk_similarity"], reverse=True):
            entity_id = str(r["entity"])
            if entity_id not in seen:
                seen.add(entity_id)
                unique_results.append(r)
        
        return unique_results[:top_k]




