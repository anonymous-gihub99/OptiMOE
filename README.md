# OptiMoE: Dynamic Topology-Aware Scheduling for Efficient Mixture-of-Experts Training

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS%202025-ML4Sys%20Workshop-red.svg)](https://neurips.cc)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Paper Highlights

OptiMoE is a novel scheduling framework that jointly optimizes expert routing and network topology configuration to minimize communication overhead in Mixture-of-Experts (MoE) training. Our key contributions from the paper:

- 🚀 **27.2% latency reduction** compared to static topologies (from 4.17μs to 3.03μs)
- ⚡ **Rapid convergence** within 30-40 iterations with minimal switching (1-2 reconfigurations)
- 📈 **Excellent scalability** from 16 to 128 nodes (13.9% → 34.0% improvement)
- 🔄 **Minimal overhead** with only 2.5% reconfiguration rate
- 🎯 **Traffic-aware** topology selection for hotspot, uniform, regional, and skewed patterns

### Key Insight
MoE communication exhibits temporal locality with predictable pattern changes during training. OptiMoE leverages this by dynamically reconfiguring optical circuit switches based on real-time traffic patterns, achieving significant performance improvements while maintaining low reconfiguration rates.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/anonymous-gihub99/OptiMOE.git
cd OptiMOE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- NetworkX 2.8+
- NumPy 1.21+
- Matplotlib 3.5+
- Pandas 1.0+

## 🚀 Quick Start

### 1. Run Basic OptiMoE Simulation
```bash
# Run the core OptiMoE implementation
python scripts/optimoe_core_implementation.py

# Expected output:
# OptiMoE converges at iteration 30-40
# Average latency: 3.03 μs
# Improvement: 27.2%
# Switch rate: 2.5%
```

### 2. Run Baseline Comparisons
```bash
# Compare against static topologies (Fat-tree, Mesh, Torus)
python scripts/run_baseline_comparison.py --nodes 64 --iterations 100

# Results will show:
# Fat-tree: 4.50 μs
# Mesh: 4.20 μs  
# Torus: 5.80 μs
# OptiMoE: 3.03 μs (27.2% improvement)
```

### 3. Run Quick Simulation (No External Dependencies)
```bash
# Lightweight version without NetworkX
python scripts/optimoe_quick_run.py

# Generates:
# - optimoe_quick_results.png
# - optimoe_quick_results.json
```

### 4. Generate Paper Figures and Tables
```bash
# Generate NeurIPS publication-quality figures
python scripts/neurips_figure_generation.py

# Generate LaTeX tables for paper
python scripts/neurips_table_generation.py

# Output files:
# - optimoe_figure1_architecture.pdf
# - optimoe_figure2_performance.pdf
# - neurips_tables.tex
```

### 5. Run Full NetworkX Simulation
```bash
# Complete simulation with traffic pattern analysis
python scripts/optimoe_networkx_simulator.py

# This runs:
# - Baseline experiments (5 iterations)
# - OptiMoE experiments (100 iterations)
# - Generates comprehensive analysis plots
```

## 📁 Repository Structure

```
OptiMOE/
├── scripts/                      # Main implementation scripts
│   ├── optimoe_core_implementation.py
│   ├── optimoe_networkx_simulator.py
│   ├── optimoe_quick_run.py
│   └── neurips_figure_generation.py
├── experiments/                   # Experiment configurations
│   └── config.yaml
├── results/                      # Output directory for results
├── paper/                        # Paper LaTeX source and figures
│   └── OptiMoE_NeurIPS2025_ML4Sys.tex
├── requirements.txt
└── README.md
```

## 📈 Key Results

### Performance Improvements by Scale
| Nodes | Baseline (μs) | OptiMoE (μs) | Improvement | Switch Rate |
|-------|---------------|--------------|-------------|-------------|
| 16    | 5.82 ± 1.4   | 5.01 ± 1.2   | 13.9%       | 0%          |
| 32    | 4.73 ± 1.3   | 3.82 ± 1.1   | 19.2%       | 2.0%        |
| 64    | 4.17 ± 1.4   | 3.03 ± 1.3   | **27.2%**   | 2.5%        |
| 128   | 3.91 ± 1.5   | 2.58 ± 1.0   | 34.0%       | 4.0%        |

### Traffic Pattern Performance
| Pattern | Optimal Topology | Latency (μs) | Description |
|---------|-----------------|--------------|-------------|
| Hotspot | Fat-tree        | 3.0          | Few experts handle majority |
| Uniform | Torus           | 5.0          | Equal distribution |
| Regional| Mesh            | 2.5          | Local clustering |
| Skewed  | Fat-tree        | 3.2          | Power-law distribution |

## 🔮 Future Work

- **Learning-based traffic prediction**: Incorporate ML models to predict traffic patterns ahead of time
- **Hierarchical topology optimization**: Extend to multi-tier datacenter networks
- **Co-optimization with model parallelism**: Joint optimization with data/pipeline parallelism strategies
- **Hardware integration**: Direct integration with optical circuit switch controllers
- **Extended evaluation**: Testing with larger MoE models (trillion+ parameters)

## 📝 Citation

If you use OptiMoE in your research, please cite our NeurIPS ML4Sys workshop paper:

```bibtex
@inproceedings{optimoe2025neurips,
  title={OptiMoE: Dynamic Topology-Aware Scheduling for Efficient Mixture-of-Experts Training},
  author={Anonymous Authors},
  booktitle={Machine Learning for Systems Workshop at NeurIPS 2025},
  year={2025},
  url={https://github.com/anonymous-gihub99/OptiMOE}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the NeurIPS 2025 ML4Sys Workshop organizers and anonymous reviewers for their valuable feedback.

## 📧 Contact

For questions or issues, please open a GitHub issue in this repository.

---

**Note**: This is an anonymous submission for NeurIPS 2025 ML4Sys Workshop. The repository will be de-anonymized after the review process.