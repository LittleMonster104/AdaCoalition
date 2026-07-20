# AdaCoalition Code Structure

## Directory Overview

```
finalcode/
├── README.md                    # Main documentation
├── requirements.txt             # Python dependencies
├── run_all_domains.py          # Main experiment runner
│
├── adacoalition/               # Core framework (19 files)
│   ├── __init__.py
│   ├── domain_agents.py        # Agent pool management
│   ├── domain_configs.py       # Domain configurations
│   ├── agents/                 # Agent implementations
│   │   ├── base_agent.py
│   │   ├── capability.py
│   │   └── llm_agent.py
│   ├── coalition/              # Coalition formation
│   │   ├── formation.py
│   │   ├── stability.py
│   │   └── metrics.py
│   ├── routing/                # Information routing
│   │   ├── router.py
│   │   └── value_function.py
│   ├── meta_learning/          # Meta-learning
│   │   └── maml.py
│   ├── evaluation/             # Metrics
│   │   └── metrics.py
│   └── utils/                  # Utilities
│       └── logger.py
│
├── baselines/                  # Baseline methods (6 files)
│   ├── single_best.py
│   ├── static_pipeline.py
│   ├── full_broadcast.py
│   ├── metagpt_style.py
│   ├── debate_style.py
│   └── hierarchical_style.py
│
├── datasets/                   # Dataset loaders (4 files)
│   ├── ednet.py               # Education domain
│   ├── ednet_fixed.py         # EdNet with fixed data loading
│   ├── resume.py              # Talent domain
│   └── aibs.py                # Science domain
│
├── experiments/                # Experiment scripts (4 files)
│   ├── run_education.py
│   ├── run_science.py
│   ├── run_talent.py
│   └── run_ablation.py
│
├── agents/                     # LLM agent wrappers (4 files)
│   └── claude_agent.py        # Example agent implementation
│
└── utils/                      # Utility functions (2 files)
    ├── __init__.py
    └── metrics.py             # Evaluation metrics
```

## File Count

- **Total Python files**: 43
- Core framework: 19 files
- Baselines: 6 files
- Datasets: 4 files
- Experiments: 4 files
- Agents: 4 files  
- Utils: 2 files
- Main scripts: 1 file
- Documentation: 1 README
- Dependencies: 1 requirements.txt

## Key Components

### Core Framework (`adacoalition/`)
- Dynamic coalition formation with Core stability
- Information-theoretic routing
- Meta-learning for cross-domain transfer
- Agent pool management

### Baselines (`baselines/`)
- SingleBest: Best single agent
- StaticPipeline: Fixed three-stage pipeline
- FullBroadcast: All agents communicate
- MetaGPT-Style: Role-based workflow
- Debate-Style: Adversarial debate
- Hierarchical-Style: Three-layer structure

### Datasets (`datasets/`)
- EdNet: Education domain (knowledge tracing)
- Resume: Talent management (job matching)
- AIBS: Science domain (peer review)

### Experiments (`experiments/`)
- Individual domain runs
- Ablation studies
- Cross-domain transfer

## Usage

```bash
# Run all experiments
python run_all_domains.py

# Run individual domain
python experiments/run_education.py

# Run ablation studies
python experiments/run_ablation.py
```

## Documentation

See `README.md` for:
- Installation instructions
- Quick start guide
- Dataset setup
- Configuration options
- Citation information
