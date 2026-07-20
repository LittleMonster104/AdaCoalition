"""Domain-specific agent creation with meaningful specializations"""
import numpy as np
from typing import List
from .agents import LLMAgent, CapabilityProfile


def create_education_agents(num_agents: int = 5, use_real_llm: bool = False) -> List[LLMAgent]:
    """
    Create agents specialized for education domain
    
    Education dimensions:
    - reasoning: Logical reasoning about learning outcomes
    - knowledge_retrieval: Access educational resources/content
    - data_analysis: Analyze student performance data
    - text_generation: Generate educational materials
    - domain_expertise: Pedagogical knowledge
    """
    agents = []
    
    # Define education-specific roles with meaningful profiles
    roles = [
        {
            'name': 'pedagogy_expert',
            'capabilities': {
                'domain_expertise': 0.95,  # Strong pedagogical knowledge
                'reasoning': 0.60,
                'text_generation': 0.50,
                'knowledge_retrieval': 0.40,
                'data_analysis': 0.30
            },
            'prompt': 'You are an expert in pedagogy and teaching methods for educational tasks.'
        },
        {
            'name': 'content_specialist',
            'capabilities': {
                'knowledge_retrieval': 0.95,  # Strong content knowledge
                'domain_expertise': 0.60,
                'text_generation': 0.50,
                'reasoning': 0.40,
                'data_analysis': 0.30
            },
            'prompt': 'You are an expert in educational content and curriculum design.'
        },
        {
            'name': 'assessment_expert',
            'capabilities': {
                'data_analysis': 0.95,  # Strong in evaluation
                'reasoning': 0.60,
                'domain_expertise': 0.50,
                'text_generation': 0.40,
                'knowledge_retrieval': 0.30
            },
            'prompt': 'You are an expert in student assessment and learning analytics.'
        },
        {
            'name': 'learning_designer',
            'capabilities': {
                'text_generation': 0.95,  # Strong in creating materials
                'reasoning': 0.60,
                'knowledge_retrieval': 0.50,
                'domain_expertise': 0.40,
                'data_analysis': 0.30
            },
            'prompt': 'You are an expert in instructional design and learning materials.'
        },
        {
            'name': 'reasoning_specialist',
            'capabilities': {
                'reasoning': 0.95,  # Strong logical reasoning
                'domain_expertise': 0.60,
                'data_analysis': 0.50,
                'knowledge_retrieval': 0.40,
                'text_generation': 0.30
            },
            'prompt': 'You are an expert in logical reasoning and problem-solving for education.'
        }
    ]
    
    for i in range(min(num_agents, len(roles))):
        role = roles[i]
        profile = CapabilityProfile.create_profile(role['capabilities'])
        
        agent = LLMAgent(
            agent_id=f"agent_{i}_{role['name']}",
            capability_profile=profile,
            system_prompt=role['prompt'],
            use_real_llm=use_real_llm
        )
        agents.append(agent)
    
    return agents


def create_science_agents(num_agents: int = 5, use_real_llm: bool = False) -> List[LLMAgent]:
    """
    Create agents specialized for science domain (paper review)
    
    Science dimensions:
    - reasoning: Logical rigor and methodology
    - knowledge_retrieval: Literature and background knowledge
    - data_analysis: Experimental design and results analysis
    - text_generation: Writing quality and clarity
    - domain_expertise: Field-specific knowledge
    """
    agents = []
    
    roles = [
        {
            'name': 'methodology_expert',
            'capabilities': {
                'reasoning': 0.95,  # Strong methodology evaluation
                'domain_expertise': 0.60,
                'data_analysis': 0.60,
                'knowledge_retrieval': 0.40,
                'text_generation': 0.30
            },
            'prompt': 'You are an expert in research methodology and experimental design.'
        },
        {
            'name': 'novelty_assessor',
            'capabilities': {
                'knowledge_retrieval': 0.95,  # Strong literature knowledge
                'domain_expertise': 0.60,
                'reasoning': 0.50,
                'text_generation': 0.40,
                'data_analysis': 0.30
            },
            'prompt': 'You are an expert in assessing research novelty and related work.'
        },
        {
            'name': 'results_analyst',
            'capabilities': {
                'data_analysis': 0.95,  # Strong in analyzing results
                'reasoning': 0.60,
                'domain_expertise': 0.50,
                'knowledge_retrieval': 0.40,
                'text_generation': 0.30
            },
            'prompt': 'You are an expert in analyzing experimental results and statistical validity.'
        },
        {
            'name': 'writing_specialist',
            'capabilities': {
                'text_generation': 0.95,  # Strong in clarity and presentation
                'reasoning': 0.60,
                'knowledge_retrieval': 0.50,
                'domain_expertise': 0.40,
                'data_analysis': 0.30
            },
            'prompt': 'You are an expert in scientific writing quality and clarity.'
        },
        {
            'name': 'domain_expert',
            'capabilities': {
                'domain_expertise': 0.95,  # Strong field knowledge
                'knowledge_retrieval': 0.60,
                'reasoning': 0.50,
                'data_analysis': 0.40,
                'text_generation': 0.30
            },
            'prompt': 'You are an expert in the specific research domain and field.'
        }
    ]
    
    for i in range(min(num_agents, len(roles))):
        role = roles[i]
        profile = CapabilityProfile.create_profile(role['capabilities'])
        
        agent = LLMAgent(
            agent_id=f"agent_{i}_{role['name']}",
            capability_profile=profile,
            system_prompt=role['prompt'],
            use_real_llm=use_real_llm
        )
        agents.append(agent)
    
    return agents


def create_talent_agents(num_agents: int = 5, use_real_llm: bool = False) -> List[LLMAgent]:
    """
    Create agents specialized for talent domain (resume matching)
    
    Talent dimensions:
    - reasoning: Logical matching and inference
    - knowledge_retrieval: Job requirements and market knowledge
    - data_analysis: Skills and experience analysis
    - text_generation: Profile summarization
    - domain_expertise: HR and recruitment expertise
    """
    agents = []
    
    roles = [
        {
            'name': 'skills_matcher',
            'capabilities': {
                'data_analysis': 0.95,  # Strong in skills analysis
                'reasoning': 0.60,
                'domain_expertise': 0.50,
                'knowledge_retrieval': 0.40,
                'text_generation': 0.30
            },
            'prompt': 'You are an expert in analyzing technical skills and job requirements.'
        },
        {
            'name': 'experience_evaluator',
            'capabilities': {
                'reasoning': 0.95,  # Strong in experience assessment
                'domain_expertise': 0.60,
                'data_analysis': 0.50,
                'knowledge_retrieval': 0.40,
                'text_generation': 0.30
            },
            'prompt': 'You are an expert in evaluating work experience and career progression.'
        },
        {
            'name': 'culture_assessor',
            'capabilities': {
                'domain_expertise': 0.95,  # Strong in culture fit
                'reasoning': 0.60,
                'knowledge_retrieval': 0.50,
                'text_generation': 0.40,
                'data_analysis': 0.30
            },
            'prompt': 'You are an expert in assessing organizational culture fit and values alignment.'
        },
        {
            'name': 'market_specialist',
            'capabilities': {
                'knowledge_retrieval': 0.95,  # Strong market knowledge
                'domain_expertise': 0.60,
                'reasoning': 0.50,
                'data_analysis': 0.40,
                'text_generation': 0.30
            },
            'prompt': 'You are an expert in job market trends and talent availability.'
        },
        {
            'name': 'profile_analyst',
            'capabilities': {
                'text_generation': 0.95,  # Strong in summarization
                'data_analysis': 0.60,
                'reasoning': 0.50,
                'knowledge_retrieval': 0.40,
                'domain_expertise': 0.30
            },
            'prompt': 'You are an expert in analyzing and summarizing candidate profiles.'
        }
    ]
    
    for i in range(min(num_agents, len(roles))):
        role = roles[i]
        profile = CapabilityProfile.create_profile(role['capabilities'])
        
        agent = LLMAgent(
            agent_id=f"agent_{i}_{role['name']}",
            capability_profile=profile,
            system_prompt=role['prompt'],
            use_real_llm=use_real_llm
        )
        agents.append(agent)
    
    return agents


def create_domain_agents(domain: str, num_agents: int = 5, use_real_llm: bool = False) -> List[LLMAgent]:
    """
    Create agents appropriate for a specific domain
    
    Args:
        domain: One of 'education', 'science', 'talent'
        num_agents: Number of agents to create
        use_real_llm: Whether to use real LLM backend
        
    Returns:
        List of specialized agents
    """
    domain_lower = domain.lower()
    
    if domain_lower == 'education':
        return create_education_agents(num_agents, use_real_llm)
    elif domain_lower == 'science':
        return create_science_agents(num_agents, use_real_llm)
    elif domain_lower == 'talent':
        return create_talent_agents(num_agents, use_real_llm)
    else:
        # Default: use generic specialists
        agents = []
        dimensions = ['reasoning', 'knowledge_retrieval', 'data_analysis', 'text_generation', 'domain_expertise']
        
        for i in range(min(num_agents, len(dimensions))):
            profile = CapabilityProfile.specialist_profile(dimensions[i], strength=0.85)
            agent = LLMAgent(
                agent_id=f"agent_{i}_{dimensions[i]}",
                capability_profile=profile,
                system_prompt=f"You are an expert in {dimensions[i]}.",
                use_real_llm=use_real_llm
            )
            agents.append(agent)
        
        return agents
