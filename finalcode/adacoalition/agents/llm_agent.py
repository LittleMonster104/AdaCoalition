"""LLM-based agent implementation with real Qwen3.5:9B"""
from typing import Dict, Any, Optional
import numpy as np
import subprocess
import json
import requests
from .base_agent import BaseAgent


class LLMAgent(BaseAgent):
    """Agent backed by Qwen3.5:9B language model via Ollama"""
    
    def __init__(self, 
                 agent_id: str, 
                 capability_profile: np.ndarray,
                 model_name: str = "qwen2.5vl:3b",
                 system_prompt: str = "You are a helpful AI assistant.",
                 use_real_llm: bool = True):
        super().__init__(agent_id, capability_profile)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.use_real_llm = use_real_llm
        
    def forward(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input using real Qwen LLM via Ollama
        
        Args:
            input_data: Dictionary containing:
                - 'task' or 'query': The input text
                - 'context': Optional context
                - 'role': Optional role specification
        
        Returns:
            Dictionary with response, confidence, agent_id
        """
        # Extract query
        query = input_data.get('task', input_data.get('query', ''))
        context = input_data.get('context', '')
        role = input_data.get('role', '')
        
        if not query and isinstance(input_data, dict) and 'text' in input_data:
            query = input_data['text']
        
        # Update context
        self.context.append({
            'query': query,
            'context': context,
            'role': role
        })
        
        # Use real LLM if enabled
        if self.use_real_llm:
            try:
                response = self._call_qwen_ollama(query, context, role)
                confidence = self._estimate_confidence(response)
            except Exception as e:
                print(f"LLM call failed: {e}, using fallback")
                response = f"[{self.agent_id}] Processing: {query[:50]}..."
                confidence = 0.5
        else:
            # Fallback to placeholder
            response = f"[{self.agent_id}] Processing: {query[:50]}..."
            confidence = 0.5
        
        return {
            'response': response,
            'confidence': confidence,
            'agent_id': self.agent_id
        }
    
    def _call_qwen_ollama(self, query: str, context: str = "", role: str = "") -> str:
        """Call Qwen model via Ollama HTTP API"""
        
        # Build prompt with system message
        system_msg = self.system_prompt
        if role:
            system_msg += f" You are acting as a {role}."
        
        # Combine context and query
        user_prompt = ""
        if context:
            user_prompt += f"Context: {context}\n\n"
        user_prompt += query
        
        # Use Ollama HTTP API for better control
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model_name,
                    'prompt': user_prompt,
                    'system': system_msg,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'num_predict': 512  # Enough for thinking + response
                    }
                },
                timeout=180  # Much longer timeout to handle concurrent requests queuing
            )
            
            if response.status_code == 200:
                result = response.json()
                # qwen3.5 uses extended thinking, actual answer is in 'response' field
                answer = result.get('response', '').strip()
                if not answer:
                    # Fallback: extract from thinking if response is empty
                    thinking = result.get('thinking', '')
                    if thinking:
                        # Try to find the last substantive line
                        lines = [l.strip() for l in thinking.split('\n') if l.strip()]
                        answer = lines[-1] if lines else "No response generated."
                return answer if answer else "No response generated."
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out after 180s")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama - is it running?")
    
    def _estimate_confidence(self, response: str) -> float:
        """
        Estimate confidence from response characteristics
        """
        # Simple heuristics for confidence
        confidence = 0.7  # Default
        
        # Longer, more detailed responses = higher confidence
        if len(response) > 100:
            confidence += 0.1
        
        # Presence of hedging words = lower confidence
        hedging_words = ['maybe', 'perhaps', 'might', 'possibly', 'unsure', 'not sure']
        if any(word in response.lower() for word in hedging_words):
            confidence -= 0.2
        
        # Presence of confident words = higher confidence
        confident_words = ['definitely', 'certainly', 'clearly', 'obviously']
        if any(word in response.lower() for word in confident_words):
            confidence += 0.1
        
        return np.clip(confidence, 0.1, 1.0)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt with system message and context (legacy, kept for compatibility)"""
        prompt_parts = [self.system_prompt]
        
        if context:
            prompt_parts.append(f"Context: {context}")
        
        # Add conversation history
        if self.context:
            prompt_parts.append("\nConversation history:")
            for msg in self.context[-3:]:  # Last 3 messages
                q = msg.get('query', msg.get('content', ''))
                if q:
                    prompt_parts.append(f"- {q[:100]}")
        
        prompt_parts.append(f"\nQuery: {query}")
        
        return "\n".join(prompt_parts)
    
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the agent with a task
        
        Args:
            query: The query/task text
            context: Optional context dictionary
        
        Returns:
            Dictionary with response, confidence, prediction, tokens
        """
        input_data = {
            'query': query,
            'context': context or {}
        }
        
        result = self.forward(input_data)
        
        # Add prediction based on response
        response_text = result.get('response', '').lower()
        
        # More comprehensive binary classification heuristic
        positive_indicators = [
            'yes', 'accept', 'approve', 'correct', 'true', 'positive', 
            'good', 'suitable', 'likely', 'succeed', 'pass', 'agree',
            'recommend', 'should', 'will', 'can', 'able'
        ]
        negative_indicators = [
            'no', 'reject', 'decline', 'incorrect', 'false', 'negative', 
            'bad', 'unsuitable', 'unlikely', 'fail', 'cannot', 'disagree',
            'should not', 'will not', 'unable'
        ]
        
        # Count indicators
        positive_count = sum(1 for word in positive_indicators if word in response_text)
        negative_count = sum(1 for word in negative_indicators if word in response_text)
        
        # Decision logic
        if positive_count > negative_count:
            prediction = 1
        elif negative_count > positive_count:
            prediction = 0
        else:
            # If tied or no indicators, use confidence threshold
            # Default to positive if confidence is high (optimistic bias for tasks)
            prediction = 1 if result.get('confidence', 0.5) >= 0.5 else 0
        
        result['prediction'] = prediction
        result['tokens'] = len(response_text.split()) * 1.3  # Rough token estimate
        
        return result
