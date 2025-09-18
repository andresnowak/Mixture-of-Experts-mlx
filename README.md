# Mixture of Experts and Advanced Transformers in MLX

This repository provides a from-scratch implementation of a Transformer decoder model using Apple's MLX framework. It is designed to be a clear, modular, and educational resource for exploring advanced concepts like Mixture of Experts (MoE), novel attention mechanisms, and different positional embedding strategies.

The project includes detailed explanations of the core concepts, configurable model architectures via YAML files, and scripts for both training and text generation.

## Features

-   **Transformer Decoder**: A complete implementation of a decoder-only Transformer built from the ground up in MLX.
-   **Mixture of Experts (MoE) Layers**:
    -   **Token-based Top-K Routing**: The standard approach where each token selects the top 'k' experts.
    -   **Expert Choice Routing**: An alternative where each expert selects its top 'k' tokens, ensuring perfect load balancing.
    -   **Shared Experts**: Implements the concept from DeepSeek-MoE, where some experts are dense (shared) to learn common knowledge representations.
-   **Advanced Attention Mechanisms**:
    -   Standard Multi-Head Attention.
    -   **Gated Attention**: An implementation from the paper "Gated Attention for LLM's" which introduces non-linearity and query-dependent sparsity to improve performance (I think it is the implementation used in Qwen3-Next).
-   **Positional Embeddings**:
    -   **Absolute**: Standard learned positional embeddings.
    -   **Sinusoidal**: Classic fixed positional embeddings that can generalize to longer sequences.
    -   **Rotary Positional Embeddings (RoPE)**: Encodes relative position information by rotating embeddings, widely used in modern LLMs.
-   **Auxiliary Losses**:
    - **Expert load balancing loss**: Implementation from DeepSeek-MoE
    - **Load balancing loss**: Implementation from Switch Transformer
-   **Highly Configurable**: Model architecture and training parameters are easily managed through YAML configuration files.

## Implemented Concepts Explained

In this repository I also tried to do explanations of what I could understand from each implementation (but there are better explanations for each of them on the internet), they are available in the `/writings` directory.

-   **MoE Models & Routing**: [writings/models.md](writings/models.md), [writings/router_types.md](writings/router_types.md)
-   **Auxiliary Losses for MoE**: [writings/auxiliary_losses.md](writings/auxiliary_losses.md)
-   **Gated Attention**: [writings/attention.md](writings/attention.md)
-   **Positional Embeddings (Absolute, Sinusoidal, RoPE)**: [writings/positional_embeddings.md](writings/positional_embeddings.md)

## Requirements & Installation

The project is built on Python 3.11+ and uses `uv` for package management.

1.  **Install `uv`**:
    Make sure you have `uv` installed first.

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/andresnowak/mixture-of-experts-mlx.git
    cd mixture-of-experts-mlx
    ```

3.  **Create a virtual environment and install dependencies:**
    This command will create a virtual environment in a `.venv` directory and install the packages listed in `pyproject.toml`.
    ```bash
    uv venv
    uv sync
    source ./.venv/bin/activate
    ```

4.  **Download the dataset:**
    The provided configurations use the Tiny Shakespeare dataset.
    ```bash
    mkdir -p dataset
    wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O dataset/shakespeare.txt
    ```

## Usage

The primary entry point is `main.py`, which handles both training and generation based on a specified configuration file.

### Training

To train a new model, use the `--train` flag. You must specify a configuration file and a path to save the final model weights.

**Example 1: Train a Standard Transformer**
```bash
python main.py \
    --config configs/shakespeare.yaml \
    --checkpoint models/shakespeare_vanilla.safetensors \
    --train
```

## What type of MoE implementations do new LLM's use:
- DeepSeek
  - Seems to use the normal Expert Load balance loss for the auxiliary loss
  - They use 1 shared expert in DeepSeek-MoE
  - The routing function they use is the normal top-k experts
  - I think they don't use a capacity factor
- Qwen 2.5 and 3
  - For the routing function it seems they use the normal top-k experts with Switch load balance loss
- Qwen 3 Next
  - They use the shared expert idea and it seems they increase the amount of finely grained experts and the amount of top-k experts that are used
  - For the routing function I think they use the same one of 2.5

## Implemenations
- The noisy top-k from Shazeer is not implemented.

## Citation

```bibtex
@misc{shazeer2017outrageously,
    title   = {Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer},
    author  = {Noam Shazeer and Azalia Mirhoseini and Krzysztof Maziarz and Andy Davis and Quoc Le and Geoffrey Hinton and Jeff Dean},
    year    = {2017},
    eprint  = {1701.06538},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@misc{dai2024deepseekmoeultimateexpertspecialization,
      title={DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models},
      author={Damai Dai and Chengqi Deng and Chenggang Zhao and R. X. Xu and Huazuo Gao and Deli Chen and Jiashi Li and Wangding Zeng and Xingkai Yu and Y. Wu and Zhenda Xie and Y. K. Li and Panpan Huang and Fuli Luo and Chong Ruan and Zhifang Sui and Wenfeng Liang},
      year={2024},
      eprint={2401.06066},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2401.06066},
}
```

```bibtex
@misc{fedus2022switchtransformersscalingtrillion,
      title={Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity},
      author={William Fedus and Barret Zoph and Noam Shazeer},
      year={2022},
      eprint={2101.03961},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2101.03961},
}
```

```bibtex
@misc{zhou2022mixtureofexpertsexpertchoicerouting,
      title={Mixture-of-Experts with Expert Choice Routing},
      author={Yanqi Zhou and Tao Lei and Hanxiao Liu and Nan Du and Yanping Huang and Vincent Zhao and Andrew Dai and Zhifeng Chen and Quoc Le and James Laudon},
      year={2022},
      eprint={2202.09368},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2202.09368},
}
```

```bibtex
@misc{xue2024openmoeearlyeffortopen,
      title={OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models},
      author={Fuzhao Xue and Zian Zheng and Yao Fu and Jinjie Ni and Zangwei Zheng and Wangchunshu Zhou and Yang You},
      year={2024},
      eprint={2402.01739},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.01739},
}
```

```bibtex
@misc{su2023roformerenhancedtransformerrotary,
      title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
      author={Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu},
      year={2023},
      eprint={2104.09864},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2104.09864},
}
```

```bibtex
@misc{jin2024moeacceleratingmixtureofexpertsmethods,
      title={MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts},
      author={Peng Jin and Bo Zhu and Li Yuan and Shuicheng Yan},
      year={2024},
      eprint={2410.07348},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.07348},
}
```
