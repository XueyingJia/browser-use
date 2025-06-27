from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
import os
import json
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.messages.utils import convert_to_openai_messages
from browser_use.agent.message_manager.views import MessageMetadata
from browser_use.agent.memory.service import Memory

logger = logging.getLogger(__name__)

class SelectiveMemory(Memory):
    """Memory system that only stores successful trajectories and retrieves relevant memories for new tasks using Chroma."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory_messages = []
        self.disable_periodic_memory = True  # disable default periodic memory creation
        
        # create storage directories
        self.storage_dir = os.path.expanduser("./browser_use_memories")
        self.chroma_dir = os.path.join(self.storage_dir, "chroma_db")
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.chroma_dir, exist_ok=True)
        
        # initialize embedding model
        self.embedding_model = self._initialize_embeddings()
        
        # init or load existing Chroma database
        self.vector_db = self._initialize_vector_db()
    
    def _initialize_embeddings(self) -> Embeddings:
        """Initialize embedding model"""
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="neulab/text-embedding-3-small",
                openai_api_key=os.environ.get("LITELLM_API_KEY"),
                openai_api_base="https://cmu.litellm.ai"
            )
        except Exception as e:
            logger.warning(f"Failed to load embedding models: {e}. Falling back to default.")
            
    
    def _initialize_vector_db(self) -> Chroma:
        """Initialize or load existing Chroma database"""
        try:
            # if the directory exists, try to load the existing Chroma instance
            # else create a new one
            return Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embedding_model,
                collection_name="browser_memories"
            )
        except Exception as e:
            logger.warning(f"Error initializing Chroma: {e}")
            # if the directory doesn't exist or loading fails, create a new instance
            return Chroma(
                embedding_function=self.embedding_model,
                collection_name="browser_memories"
            )
    
    def create_procedural_memory(self, current_step: int) -> None:
        """Override default periodic memory creation"""
        if self.disable_periodic_memory:
            # skip the default periodic memory creation
            return None
        else:
            # if you want to keep the original functionality, call the parent method
            return super().create_procedural_memory(current_step)
    
    def collect_message(self, message: BaseMessage) -> None:
        """Collect message for the trajectory"""
        self.trajectory_messages.append(message)
    
    def clear_trajectory(self) -> None:
        """Clear stored trajectory"""
        self.trajectory_messages = []
    
    def store_successful_trajectory(self, success: bool, task: str) -> Optional[str]:
        """Store trajectory only if task was successful"""
        if not success or not self.trajectory_messages:
            logger.info("Not storing trajectory: Task unsuccessful or no messages")
            self.clear_trajectory()
            return None
        
        logger.info(f"Task successful - storing trajectory of {len(self.trajectory_messages)} messages in Chroma")
        
        try:
            task_description = task
            
            # convert to OpenAI messages format
            parsed_messages = convert_to_openai_messages(self.trajectory_messages)
            
            # reconstruct messages to ensure each has meaningful content
            reconstructed_messages = []
            for msg in parsed_messages:
                if isinstance(msg, dict):
                    content = ""
                    
                    # if we have tool_calls, abstract function calls and arguments
                    if msg.get('tool_calls'):
                        for tool_call in msg['tool_calls']:
                            if tool_call.get('function'):
                                func_name = tool_call['function'].get('name', '')
                                func_args = tool_call['function'].get('arguments', '{}')
                                content += f"Called {func_name} with arguments: {func_args}\n"
                    
                    # if the content is empty but we have nonempty extracted content above, use it
                    if not msg.get('content') and content:
                        msg['content'] = content
                        
                    # if still no content, add a placeholder
                    if not msg.get('content'):
                        msg['content'] = f"Message of role: {msg.get('role', 'unknown')}"
                        
                    reconstructed_messages.append(msg)
            
            logger.info(f"Reconstructed {len(reconstructed_messages)} messages with content")
            
            # create a summary of the trajectory for indexing
            trajectory_summary = f"Task: {task_description}\n\nSteps:"
            for i, msg in enumerate(reconstructed_messages):
                if msg.get('content'):
                    trajectory_summary += f"\nStep {i+1}: {msg['content']}"
                    
            logger.info(f"Created trajectory summary of length {len(trajectory_summary)}")
            
            # create unique ID and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_success.json"
            memory_id = f"memory_{timestamp}"
            
            # save as a JSON file
            memory_data = {
                "task": task_description,
                "timestamp": datetime.now().isoformat(),
                "summary": trajectory_summary,
                "messages": reconstructed_messages,
                "id": memory_id
            }
            
            file_path = os.path.join(self.storage_dir, filename)
            with open(file_path, 'w') as f:
                json.dump(memory_data, f, indent=2)
                
            logger.info(f"Successfully stored memory to {file_path}")
            
            # try to add new item to the vector database
            try:
                # add the trajectory info to the vector database
                self.vector_db.add_texts(
                    texts=[trajectory_summary],
                    metadatas=[{
                        'task': task_description,
                        'steps': len(self.trajectory_messages),
                        'file_path': file_path,
                        'id': memory_id
                    }],
                    ids=[memory_id]
                )
                
                # try to persist the Chroma database
                try:
                    if hasattr(self.vector_db, '_collection') and hasattr(self.vector_db._collection, 'persist'):
                        self.vector_db._collection.persist()
                        logger.info("Successfully persisted Chroma using _collection.persist()")
                    elif hasattr(self.vector_db, 'client') and hasattr(self.vector_db.client, 'persist'):
                        self.vector_db.client.persist()
                        logger.info("Successfully persisted Chroma using client.persist()")
                    else:
                        logger.warning("Could not find persist method on Chroma instance")
                except Exception as persist_err:
                    logger.warning(f"Error persisting Chroma database: {persist_err}")
                    
                logger.info("Successfully stored in Chroma vector database")
            except Exception as db_err:
                logger.warning(f"Failed to store in Chroma (falling back to file storage only): {db_err}")
                
            return f"Stored successful trajectory for task: {task_description[:50]}..."
                
        except Exception as e:
            logger.error(f"Error storing successful trajectory: {e}", exc_info=True)
            return None
        finally:
            self.clear_trajectory()
    
    def retrieve_relevant_memories(self, task_query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to the current task using vector similarity"""
        if not task_query:
            return []
        
        memories = []
        
        # retrieve memories from Chroma vector database
        try:
            logger.info(f"Retrieving memories from Chroma for query: {task_query[:50]}...")
            
            # using similarity_search_with_score to get both documents and their scores
            results = self.vector_db.similarity_search_with_score(
                query=task_query,
                k=limit
            )
            
            if results:
                logger.info(f"Found {len(results)} memories in vector database")
                
                # convert results to standard memory format
                for doc, score in results:
                    # convert distance score to similarity (0 distance score is the most similar one, then the similarity as 1.0 is most similar)
                    similarity = 1.0 - min(score, 1.0)
                    
                    memories.append({
                        "content": doc.page_content,
                        "metadata": {
                            **doc.metadata,
                            "similarity": similarity
                        }
                    })
            else:
                logger.info("No memories found in vector database")
                
        except Exception as db_err:
            logger.warning(f"Error retrieving from Chroma, falling back to file-based retrieval: {db_err}")
        
        # ensure results are sorted by similarity and limited to the specified number
        memories.sort(key=lambda x: x["metadata"].get("similarity", 0), reverse=True)
        return memories[:limit]
    
    def format_memory_for_context(self, memory: Dict[str, Any]) -> str:
        """Format memory for inclusion in context"""
        content = memory.get('content', '')
        metadata = memory.get('metadata', {})
        
        task = metadata.get('task', 'previous task')
        steps = metadata.get('steps', 'unknown')
        similarity = metadata.get('similarity', 0)
        similarity_percent = f"{similarity:.1%}" if isinstance(similarity, float) else 'unknown'
        
        formatted_memory = f"""PREVIOUS SUCCESSFUL EXPERIENCE (Relevance: {similarity_percent}):
Task: {task}
Steps taken: {steps}
Approach: {content}
"""
        return formatted_memory
    
    def inject_memories_to_context(self, task_query: str) -> None:
        """Retrieve and inject relevant memories into context"""
        memories = self.retrieve_relevant_memories(task_query)
        
        if not memories:
            logger.info("No relevant memories found for the current task")
            return

        # Format each memory for context
        memory_texts = [self.format_memory_for_context(m) for m in memories]
        combined_memory = "\n\n".join(memory_texts)

        logger.info(f"Found {len(memories)} relevant memories for task: {task_query}")
        logger.info(f"Combined memory content:\n{combined_memory}")  # log first 200 chars for brevity
        
        # create a memory message to inject into the context
        memory_message = HumanMessage(
            content=f"I'm providing you with previous successful experiences that may be relevant to this task. Use these as references to help complete the current task:\n\n{combined_memory}"
        )
        
        # calculate the number of tokens in the memory message
        memory_tokens = self.message_manager._count_tokens(memory_message)
        memory_metadata = MessageMetadata(tokens=memory_tokens, message_type='retrieved_memory')
        
        # Find the position to insert the memory message (after the system message)
        insert_position = 0
        for i, msg in enumerate(self.message_manager.state.history.messages):
            if isinstance(msg.message, SystemMessage):
                insert_position = i + 1  # insert after the system message
                
        # if the position is at the end, append it; otherwise, insert it at the calculated position
        self.message_manager.state.history.add_message(memory_message, memory_metadata, position=insert_position)
        logger.info(f"Injected {len(memories)} relevant memories into context")