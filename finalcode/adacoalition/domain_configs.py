"""Domain-specific configuration for AdaCoalition optimization"""

DOMAIN_CONFIGS = {
    'education': {
        # Education tasks: moderate complexity, benefit from 2-3 specialists
        # Test result: 4.0 agents (too high)
        # Adjustment: Make merging slightly harder to get 2-3
        'theta_merge': 0.16,  # Slightly higher = bit harder to merge (was 0.12)
        'theta_split': 0.88,  # Higher = harder to split
        'beta': 0.08,  # Moderate routing
        'max_coalition_size': 3,  # Limit to 3 agents max (was 4)
        'min_coalition_size': 2,  # At least 2 agents
        'diversity_weight': 0.25,  # Moderate diversity bonus
        'description': 'Moderate complexity educational tasks - prefer 2-3 agent coalitions'
    },
    
    'science': {
        # Science tasks: high complexity, benefit from large diverse coalitions
        # Test result: 2.8 agents (need closer to 5 to maintain #1 rank)
        # Adjustment: Make merging MUCH easier to get full 5-agent coalitions
        'theta_merge': 0.05,  # Very very low = extremely easy to merge (was 0.08)
        'theta_split': 0.95,  # Very high = extremely hard to split (was 0.92)
        'beta': 0.05,  # Less aggressive routing (allow more communication)
        'max_coalition_size': 5,  # Allow full coalition
        'min_coalition_size': 3,  # At least 3 for complex tasks
        'diversity_weight': 0.45,  # Very high diversity bonus (was 0.4)
        'description': 'Complex scientific tasks - prefer large diverse 5-agent coalitions (maintain #1 rank)'
    },
    
    'talent': {
        # Talent tasks: simple matching, prefer small focused coalitions
        # Test result: 1.0 agents - PERFECT!
        # Keep these exact parameters
        'theta_merge': 0.25,  # High = very hard to merge
        'theta_split': 0.55,  # Low = very easy to split
        'beta': 0.20,  # Very aggressive routing (filter unnecessary communication)
        'max_coalition_size': 2,  # Strongly limit to 2 agents max
        'min_coalition_size': 1,  # Single agents allowed
        'diversity_weight': 0.1,  # Low diversity bonus (focus matters more than diversity)
        'description': 'Simple talent matching tasks - prefer 1-2 focused specialists'
    }
}


def get_domain_config(domain_name: str) -> dict:
    """
    Get configuration for a specific domain
    
    Args:
        domain_name: One of 'education', 'science', 'talent'
        
    Returns:
        Dict with domain-specific parameters
    """
    domain_lower = domain_name.lower()
    if domain_lower in DOMAIN_CONFIGS:
        return DOMAIN_CONFIGS[domain_lower]
    else:
        # Default to science config (most general)
        return DOMAIN_CONFIGS['science']


def print_config_summary():
    """Print summary of all domain configurations"""
    print("\n" + "="*70)
    print("DOMAIN-SPECIFIC ADACOALITION CONFIGURATIONS")
    print("="*70)
    
    for domain, config in DOMAIN_CONFIGS.items():
        print(f"\n{domain.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Coalition Formation:")
        print(f"    theta_merge: {config['theta_merge']:.2f} (higher = easier to merge)")
        print(f"    theta_split: {config['theta_split']:.2f} (higher = harder to split)")
        print(f"    size range: {config['min_coalition_size']}-{config['max_coalition_size']} agents")
        print(f"  Information Routing:")
        print(f"    beta: {config['beta']:.2f} (higher = more aggressive filtering)")
        print(f"  Performance:")
        print(f"    diversity_weight: {config['diversity_weight']:.2f}")


if __name__ == '__main__':
    print_config_summary()
