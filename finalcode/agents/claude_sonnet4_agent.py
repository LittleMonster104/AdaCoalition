"""
Claude Sonnet 4 Agent - 使用你的API
"""
import requests
import time
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ada_coalition.agents.base_agent import BaseAgent


class ClaudeSonnet4Agent(BaseAgent):
    """
    Claude Sonnet 4 Agent - 使用兼容API
    """
    
    def __init__(self, 
                 agent_id: str,
                 capability_profile: np.ndarray,
                 system_prompt: str = "You are a helpful AI assistant."):
        super().__init__(agent_id, capability_profile)
        
        self.api_key = "sk-5dfc649c53c8ec059ddd872e6fb95988477fe21e35bf3e513172cfc5d4f783a8"
        self.base_url = "https://api.jiu96.com"
        self.model = "claude-sonnet-4-20250514"
        self.system_prompt = system_prompt
        
    def forward(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """调用Claude Sonnet 4"""
        query = input_data.get('task', input_data.get('query', ''))
        context = input_data.get('context', '')
        
        if not query and isinstance(input_data, dict) and 'text' in input_data:
            query = input_data['text']
        
        if context:
            full_query = f"{context}\n\n{query}"
        else:
            full_query = query
        
        try:
            response_text = self._call_claude_api(full_query)
            confidence = self._estimate_confidence(response_text)
        except Exception as e:
            print(f"Claude API failed: {e}")
            response_text = f"[{self.agent_id}] Error: {str(e)[:50]}"
            confidence = 0.5
        
        return {
            'response': response_text,
            'confidence': confidence,
            'agent_id': self.agent_id
        }
    
    def _call_claude_api(self, query: str) -> str:
        """调用Claude API"""
        url = f"{self.base_url}/v1/messages"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": f"{self.system_prompt}\n\n{query}"
            }]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        
        if response.status_code == 200:
            data = response.json()
            return data['content'][0]['text']
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")
    
    def _estimate_confidence(self, response: str) -> float:
        """估算置信度"""
        confidence = 0.7
        
        if len(response) > 100:
            confidence += 0.1
        
        hedging = ['maybe', 'perhaps', 'might', 'possibly', 'unsure']
        if any(word in response.lower() for word in hedging):
            confidence -= 0.2
        
        confident = ['definitely', 'certainly', 'clearly', 'obviously']
        if any(word in response.lower() for word in confident):
            confidence += 0.1
        
        return np.clip(confidence, 0.1, 1.0)
    
    def query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """兼容LLMAgent的query方法"""
        result = self.forward({'query': query, 'context': context or {}})
        
        # 添加prediction
        response_text = result.get('response', '').lower()
        
        positive_indicators = [
            'yes', 'accept', 'correct', 'true', 'positive', 'good', 
            'suitable', 'likely', 'succeed', 'recommend', 'should', 'agree'
        ]
        negative_indicators = [
            'no', 'reject', 'incorrect', 'false', 'negative', 'bad',
            'unsuitable', 'unlikely', 'fail', 'cannot', 'disagree'
        ]
        
        pos_count = sum(1 for w in positive_indicators if w in response_text)
        neg_count = sum(1 for w in negative_indicators if w in response_text)
        
        if pos_count > neg_count:
            prediction = 1
        elif neg_count > pos_count:
            prediction = 0
        else:
            prediction = 1 if result.get('confidence', 0.5) >= 0.5 else 0
        
        result['prediction'] = prediction
        result['tokens'] = len(response_text.split()) * 1.3
        
        return result


# 测试
if __name__ == '__main__':
    print("测试Claude Sonnet 4 Agent")
    
    agent = ClaudeSonnet4Agent(
        agent_id="claude_test",
        capability_profile=np.array([0.9, 0.8, 0.7, 0.6, 0.9])
    )
    
    result = agent.query("What is 2+2? Answer briefly.")
    print(f"✓ 响应: {result['response']}")
    print(f"✓ 置信度: {result['confidence']}")
    print(f"✓ 预测: {result['prediction']}")
