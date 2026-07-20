"""
Compatible Claude API Agent - 使用提供的API端点
"""
import requests
import time
import json
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ada_coalition.agents.base_agent import BaseAgent
import numpy as np


class CompatibleClaudeAgent(BaseAgent):
    """
    使用Compatible Claude API的Agent
    支持通过代理访问Claude模型
    """
    
    def __init__(self, 
                 agent_id: str,
                 capability_profile: np.ndarray,
                 api_key: str,
                 base_url: str = "https://api.jiu96.com",
                 model: str = "claude-3-5-sonnet-20241022",
                 system_prompt: str = "You are a helpful AI assistant."):
        super().__init__(agent_id, capability_profile)
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.system_prompt = system_prompt
        
    def forward(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用Claude API进行推理
        """
        # 提取query
        query = input_data.get('task', input_data.get('query', ''))
        context = input_data.get('context', '')
        
        if not query and isinstance(input_data, dict) and 'text' in input_data:
            query = input_data['text']
        
        # 构建完整prompt
        if context:
            full_query = f"{context}\n\n{query}"
        else:
            full_query = query
        
        # 调用API
        try:
            response_text = self._call_claude_api(full_query)
            confidence = self._estimate_confidence(response_text)
        except Exception as e:
            print(f"Claude API call failed: {e}, using fallback")
            response_text = f"[{self.agent_id}] Processing: {query[:50]}..."
            confidence = 0.5
        
        return {
            'response': response_text,
            'confidence': confidence,
            'agent_id': self.agent_id
        }
    
    def _call_claude_api(self, query: str) -> str:
        """
        调用Claude API (Anthropic格式通过代理)
        """
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
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                data = response.json()
                # Anthropic API格式
                if 'content' in data and len(data['content']) > 0:
                    return data['content'][0]['text']
                else:
                    raise Exception(f"Unexpected response format: {data}")
            else:
                raise Exception(f"API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("Claude API request timed out after 180s")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Claude API")
    
    def _estimate_confidence(self, response: str) -> float:
        """估算置信度"""
        confidence = 0.7
        
        if len(response) > 100:
            confidence += 0.1
        
        hedging_words = ['maybe', 'perhaps', 'might', 'possibly', 'unsure', 'not sure']
        if any(word in response.lower() for word in hedging_words):
            confidence -= 0.2
        
        confident_words = ['definitely', 'certainly', 'clearly', 'obviously']
        if any(word in response.lower() for word in confident_words):
            confidence += 0.1
        
        return np.clip(confidence, 0.1, 1.0)
    
    def query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Query方法（与LLMAgent兼容）
        """
        input_data = {
            'query': query,
            'context': context or {}
        }
        
        result = self.forward(input_data)
        
        # 添加prediction
        response_text = result.get('response', '').lower()
        
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
        
        positive_count = sum(1 for word in positive_indicators if word in response_text)
        negative_count = sum(1 for word in negative_indicators if word in response_text)
        
        if positive_count > negative_count:
            prediction = 1
        elif negative_count > positive_count:
            prediction = 0
        else:
            prediction = 1 if result.get('confidence', 0.5) >= 0.5 else 0
        
        result['prediction'] = prediction
        result['tokens'] = len(response_text.split()) * 1.3
        
        return result


def test_claude_api():
    """测试Claude API连接"""
    print("="*80)
    print("测试Compatible Claude API连接")
    print("="*80)
    
    API_KEY = "sk-5dfc649c53c8ec059ddd872e6fb95988477fe21e35bf3e513172cfc5d4f783a8"
    BASE_URL = "https://api.jiu96.com"
    
    # 测试不同模型
    models_to_test = [
        "claude-3-5-sonnet-20241022",  # 最新最强
        "claude-3-5-sonnet-20240620",  # 备选
        "claude-3-sonnet-20240229"     # 备选
    ]
    
    for model in models_to_test:
        print(f"\n测试模型: {model}")
        print("-"*80)
        
        try:
            agent = CompatibleClaudeAgent(
                agent_id="test_agent",
                capability_profile=np.array([0.8, 0.7, 0.6, 0.5, 0.9]),
                api_key=API_KEY,
                base_url=BASE_URL,
                model=model,
                system_prompt="You are a helpful AI assistant."
            )
            
            # 测试查询
            test_query = "What is 2+2? Answer briefly."
            
            start_time = time.time()
            result = agent.forward({'query': test_query})
            elapsed = time.time() - start_time
            
            print(f"✓ API连接成功!")
            print(f"  响应: {result['response'][:200]}")
            print(f"  置信度: {result['confidence']:.2f}")
            print(f"  耗时: {elapsed:.2f}秒")
            print(f"\n✓ 推荐使用模型: {model}")
            return model
            
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            continue
    
    print("\n✗ 所有模型测试失败")
    return None


if __name__ == '__main__':
    recommended_model = test_claude_api()
    
    if recommended_model:
        print("\n" + "="*80)
        print("准备运行实验")
        print("="*80)
        print(f"推荐模型: {recommended_model}")
        print("下一步: 运行 run_claude_experiments.py")
