# Copyright 2024
# Directory: yt-rag/app/services/chat.py

"""
Chat completion service for generating RAG responses.
Now using Groq (ChatGroq) as the LLM provider.
"""

import logging
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatService:
    """Service for chat completions using Groq."""

    def __init__(self):
        """Initialize Groq chat client."""
        self.client = ChatGroq(
            model=settings.groq_chat_model,  # misal: "deepseek-r1-distill-llama-70b"
            temperature=settings.temperature,
            api_key=settings.groq_api_key
        )
        self.model = settings.groq_chat_model
        logger.info(f"Initialized chat service with Groq model: {self.model}")

    async def generate_answer(self, query: str, context_blocks: List[Dict[str, Any]]) -> str:
        """
        Generate RAG answer using context blocks.

        Args:
            query: User's question
            context_blocks: Retrieved chunks with metadata

        Returns:
            Generated answer with citations
        """
        # Build context string with citations
        context_parts = []
        for block in context_blocks:
            chunk_id = block.get('chunk_id', 'unknown')
            text = block.get('text', '')
            context_parts.append(f"[{chunk_id}] {text}")

        context = "\n\n".join(context_parts)

        system_prompt = """You are a helpful AI assistant for customer support that answers questions based on provided context.

                            IMPORTANT RULES:
                            1. For questions about policies, returns, shipping, sizing, or support: Answer ONLY using the provided context and include citations
                            2. For general greetings or casual conversation: You can respond naturally and friendly
                            3. For questions outside your knowledge base: Politely redirect to relevant policies or suggest contacting support
                            4. Always include citations [chunk_id] when using context information
                            5. Be concise but comprehensive
                            6. Maintain a helpful, professional tone"""

        user_prompt = f"""Context:
                        {context}

                        Question: {query}

                        Please provide an answer based on the context above, including appropriate citations."""

        try:
            response = self.client.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            # Ambil teks hasil model
            answer = response.content if hasattr(response, "content") else str(response)

            logger.info("Generated answer using Groq")
            return answer or "I couldn't generate an answer."

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"I encountered an error while processing your question: {str(e)}"


# Global service instance
chat_service = ChatService()