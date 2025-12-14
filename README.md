<h2 align="center">
  QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation
</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2511.20100">
    <img
      src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&logoColor=red"
      alt="QiMeng-Kernel Paper on arXiv"
    />
  </a>
</p>

This is the public release of the official implementation for the paper [QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation](https://arxiv.org/abs/2511.20100)

Please note that this repository is still under active development. We will release the training and offline dataset in the near future.

## Methods
<p align="center">
  <img src="./docs/static/images//overview.png" alt="Overview" width="80%" />
</p>

### Main results

#### KernelBench

<p align="center">
  <img src="./docs/static/images/result-kernelbenchv0.png" alt="Overview" width="80%" />
</p>

#### TritonBench

<p align="center">
  <img src="./docs/static/images/result-tritonbench.png" alt="Overview" width="80%" />
</p>


## Quick Start
### Install
We recommend installing the dependencies using pip.
```bash
git clone https://github.com/QiMeng-IPRC/QiMeng-Kernel.git
cd QiMeng-Kernel
pip install -r requirements.txt
pip install -e .
```

### Usage
Run eval
```bash
bash scripts/eval_generations.sh \
  <benchmark> \
  <run_dir> \
  <subset_name>
```

Generate Kernel with checkpoint
```bash
bash scripts/inference.sh \
  <checkpoint_path> \
  <llm_api_baseurl> \
  <llm_api_key> \
  <llm_api_model> \
  <dataset_name>
```

Train Macro-Thinking model
```bash
bash scripts/train.sh \
  <base_model> \
  <num_envs>
```

## Citation
```bibtex
@article{zhu2025qimeng,
  title={QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation},
  author={Zhu, Xinguo and Peng, Shaohui and Guo, Jiaming and Chen, Yunji and Guo, Qi and Wen, Yuanbo and Qin, Hang and Chen, Ruizhi and Zhou, Qirui and Gao, Ke and others},
  journal={arXiv preprint arXiv:2511.20100},
  year={2025}
}
```


