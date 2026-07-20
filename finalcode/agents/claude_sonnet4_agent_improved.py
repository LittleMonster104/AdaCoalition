"""
改进版Claude Sonnet 4 Agent - 修复预测提取逻辑
关键改进：
1. 使用明确的指令格式
2. 要求Claude以DECISION标记结尾
3. 更robust的预测提取
"""
import requests
import time
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from ada_coalition.agents.base_agent import BaseAgent


class ClaudeSonnet4AgentImproved(BaseAgent):
    """
    改进版Claude Sonnet 4 Agent
    使用明确的decision格式
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
        """调用Claude with improved prompt"""
        query = input_data.get('task', input_data.get('query', ''))
        context = input_data.get('context', '')
        
        if not query and isinstance(input_data, dict) and 'text' in input_data:
            query = input_data['text']
        
        # 构建改进的prompt
        improved_query = self._build_decision_prompt(query, context)
        
        try:
            response_text = self._call_claude_api(improved_query)
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
    
    def _build_decision_prompt(self, query: str, context: str = "") -> str:
        """
        构建明确要求decision的prompt
        """
        full_context = f"{context}\n\n" if context else ""
        
        decision_prompt = f"""{full_context}{query}

IMPORTANT: You MUST end your response with a clear decision in this exact format:
- If positive/accept/yes: End with "FINAL_DECISION: POSITIVE"
- If negative/reject/no: End with "FINAL_DECISION: NEGATIVE"

Provide your analysis first, then end with the FINAL_DECISION line."""
        
        return decision_prompt
    
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
        
        # 如果有明确的DECISION标记，提高置信度
        if "FINAL_DECISION:" in response:
            confidence += 0.2
        
        if len(response) > 100:
            confidence += 0.05
        
        hedging = ['maybe', 'perhaps', 'might', 'possibly', 'unsure', 'unclear']
        if any(word in response.lower() for word in hedging):
            confidence -= 0.15
        
        return np.clip(confidence, 0.1, 1.0)
    
    def query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """兼容LLMAgent的query方法 - 改进的预测提取"""
        result = self.forward({'query': query, 'context': context or {}})
        
        # 改进的预测提取逻辑
        prediction = self._extract_prediction_improved(result.get('response', ''))
        
        result['prediction'] = prediction
        result['tokens'] = len(result.get('response', '').split()) * 1.3
        
        return result
    
    def _extract_prediction_improved(self, response: str) -> int:
        """
        改进的预测提取逻辑
        
        优先级：
        1. 查找FINAL_DECISION标记
        2. 查找其他明确的decision标记
        3. 退回到keyword counting（但更谨慎）
        """
        response_lower = response.lower()
        
        # 优先级1：查找FINAL_DECISION标记
        if "FINAL_DECISION: POSITIVE" in response.upper():
            return 1
        elif "FINAL_DECISION: NEGATIVE" in response.upper():
            return 0
        
        # 优先级2：查找其他明确标记
        decision_patterns = [
            (r'\bDECISION:\s*(ACCEPT|YES|POSITIVE|APPROVED)', 1),
            (r'\bDECISION:\s*(REJECT|NO|NEGATIVE|DENIED)', 0),
            (r'\bFINAL\s+ANSWER:\s*(YES|POSITIVE|ACCEPT)', 1),
            (r'\bFINAL\s+ANSWER:\s*(NO|NEGATIVE|REJECT)', 0),
            (r'\bCONCLUSION:\s*(ACCEPT|POSITIVE|YES)', 1),
            (r'\bCONCLUSION:\s*(REJECT|NEGATIVE|NO)', 0),
        ]
        
        for pattern, prediction in decision_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return prediction
        
        # 优先级3：改进的keyword counting
        # 只在句子末尾或结论部分计算
        last_sentences = ' '.join(response.split('.')[-3:])  # 最后3句
        last_lower = last_sentences.lower()
        
        strong_positive = ['recommend accept', 'should accept', 'is suitable', 
                          'good match', 'strong candidate', 'clearly qualified']
        strong_negative = ['recommend reject', 'should reject', 'not suitable',
                          'poor match', 'weak candidate', 'not qualified']
        
        for phrase in strong_positive:
            if phrase in last_lower:
                return 1
        
        for phrase in strong_negative:
            if phrase in last_lower:
                return 0
        
        # 如果都没有，使用整体sentiment（更保守）
        positive_indicators = ['yes', 'accept', 'good', 'suitable', 'recommend', 'strong']
        negative_indicators = ['no', 'reject', 'bad', 'unsuitable', 'weak', 'not recommend']
        
        pos_count = sum(1 for w in positive_indicators if w in response_lower)
        neg_count = sum(1 for w in negative_indicators if w in response_lower)
        
        # 需要明显优势才判断
        if pos_count > neg_count + 2:
            return 1
        elif neg_count > pos_count + 2:
            return 0
        else:
            # 无法判断，默认基于长度和整体tone
            return 1 if len(response) > 200 else 0


# 测试改进版
if __name__ == '__main__':
    print("测试改进版Claude Agent")
    print("="*80)
    
    agent = ClaudeSonnet4AgentImproved(
        agent_id="claude_improved",
        capability_profile=np.array([0.9, 0.8, 0.7, 0.6, 0.9])
    )
    
    # 测试
    test_query = """Should we hire this candidate?

Candidate: 5 years Python, ML experience, good communication.
Job: Senior SWE, requires Python, ML, teamwork.

Evaluate and make a decision."""
    
    result = agent.query(test_query)
    
    print(f"\nQuery:\n{test_query}\n")
    print(f"Response:\n{result['response']}\n")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # 检查是否有FINAL_DECISION标记
    if "FINAL_DECISION:" in result['response']:
        print("\n✓ 找到FINAL_DECISION标记!")
    else:
        print("\n⚠️ 未找到FINAL_DECISION标记，使用fallback逻辑")
