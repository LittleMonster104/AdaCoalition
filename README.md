# AdaCoalition: Domain-Agnostic Multi-Agent Collaboration
 
**"AdaCoalition: Domain-Agnostic Multi-Agent Collaboration with Dynamic Coalition Formation"**

## 📋 Overview

AdaCoalition is a domain-agnostic framework for dynamic multi-agent collaboration that:
- Dynamically forms agent coalitions based on task requirements
- Routes information using information-theoretic principles
- Achieves 96% zero-shot cross-domain transfer

**Key Results:**
- Education (EdNet): 69.6% F1 (+35% vs best baseline)
- Talent (Resume): 87.5% F1 (+44% vs best baseline, Cohen's Kappa 0.379)
- Science (PeerRead): 80.9% F1 (+68% vs best baseline)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd finalcode/

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-api-key"  # If using OpenAI models
```

### Running Experiments

**Run all three domains:**
```bash
python run_all_domains_clean.py
```

**Run individual domains:**
```bash
# Education domain
python experiments/run_education.py

# Science domain  
python experiments/run_science.py

# Talent domain
python experiments/run_talent.py
```

**Run cross-domain transfer:**
```bash
python experiments/run_transfer.py
```

## 📁 Code Structure

```
finalcode/
├── adacoalition/           # Core framework
│   ├── domain_agents.py    # Agent pool management
│   └── domain_configs.py   # Domain-specific configurations
│
├── baselines/              # Baseline implementations
│   ├── single_best.py
│   ├── metagpt_style.py
│   ├── debate_style.py
│   └── hierarchical_style.py
│
├── datasets/               # Dataset loaders
│   ├── ednet.py           # Education domain (EdNet)
│   ├── resume.py          # Talent domain
│   └── aibs.py            # Science domain (synthetic + PeerRead)
│
├── experiments/            # Experiment scripts
│   ├── run_education.py
│   ├── run_science.py
│   ├── run_talent.py
│   └── run_transfer.py
│
├── agents/                 # LLM agent implementations
│   └── claude_agent.py    # Example LLM agent wrapper
│
└── utils/                  # Utility functions
    └── metrics.py         # Evaluation metrics
```

## 🔬 Reproducing Paper Results

### Main Results (Table 1)

```bash
# Run all methods on all domains with Qwen2.5-VL:3B
python run_all_domains_clean.py --model qwen2.5-vl:3b --save_results
```

Expected output:
```
Education: AdaCoalition F1=0.696 (Best)
Talent:    AdaCoalition F1=0.875 (Best, Kappa=0.379)
Science:   AdaCoalition F1=0.809 (Best)
```

### Cross-LLM Validation (Supplementary)

```bash
# Run with Gemma4:e2b
python run_all_domains_clean.py --model gemma4:e2b --save_results
```

### Ablation Studies

```bash
python experiments/run_ablation.py
```

### Cross-Domain Transfer

```bash
python experiments/run_transfer.py
```

## 📊 Datasets

### Education Domain: EdNet-KT3
- **Source**: Choi et al., 2020 (AIED)
- **Size**: 131M learning interactions, 784K students
- **Task**: Knowledge tracing (predict next response)
- **Download**: Follow instructions in `datasets/ednet.py`

### Science Domain: PeerRead + Synthetic AIBS
- **PeerRead**: Kang et al., 2018 (NAACL)
  - 3,000 papers from ArXiv/ACL
- **AIBS**: 1,500 synthetic biomedical proposals (augmentation)
- **Task**: Paper acceptance prediction

### Talent Domain: Resume Corpus
- **Source**: Dave et al., 2018 (NAACL)
- **Size**: 2,000+ resume-job pairs
- **Task**: Job-candidate matching (3-class)

## ⚙️ Configuration

Edit `config/default_config.yaml` to customize:

```yaml
# Model configuration
model_name: "qwen2.5-vl:3b"  # or "gemma4:e2b"
api_base: "http://localhost:11434"

# Coalition formation parameters
theta_merge: 0.3
theta_split: 0.7
max_iterations: 20

# Information routing
beta: 0.1  # Communication cost parameter

# Meta-learning
meta_lr: 1e-4
num_meta_steps: 100
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Quick functionality test
python test_quick.py
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{adacoalition2027,
  title={AdaCoalition: Domain-Agnostic Multi-Agent Collaboration with Dynamic Coalition Formation},
  author={Anonymous Authors},
  booktitle={Proceedings of AAAI},
  year={2027}
}
```

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See `requirements.txt` for complete list

## 📧 Contact

For questions about the code or paper, please open an issue or contact:
- [Anonymous for review]

## 📄 License

MIT License (to be updated after acceptance)

## 🙏 Acknowledgments

- EdNet dataset: Riiid AI Research
- PeerRead dataset: Kang et al.
- Resume Corpus: Dave et al.

## 📚 Additional Resources

- **Paper**: [Link to be added]
- **Supplementary Material**: See `supplementary.tex`
- **Project Page**: [To be added]

---

**Note**: This code is released for review purposes. Full documentation and model checkpoints will be released upon paper acceptance.
