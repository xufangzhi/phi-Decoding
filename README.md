### This version contains the original code of phi_docoding. Successfully run this code may take a little time

<h1 align="center">
φ-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation
</h1>

<p align="center">
  <a href="https://github.com/xufangzhi/phi-Decoding/"><b>[🌐 PyPi Package]</b></a> •
  <a href="https://arxiv.org/abs/2503.13288"><b>[📜 Paper]</b></a> •
  <a href="https://github.com/xufangzhi/phi-Decoding"><b>[🐱 GitHub]</b></a>
  
</p>

<p align="center">
Repo for "<a href="https://arxiv.org/abs/2503.13288" target="_blank">φ-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation</a>"
</p>

## 🔥 News

- [2025/02/16] 🔥🔥🔥 $\phi$-Decoding is released !

## 📖 Results

$\phi$-Decoding provides balanced inference-time exploration and exploitation. The following scaling curve offers the comparisons with other strong methods on LLaMA3.1-8B models. For more results, please refer to our [paper](https://arxiv.org/abs/2311.09278).

<p align="center">
    <img src="./assets/scaling_law.png" alt="scaling" width="400">
</p>

## 🚀 Quick Start

To use the $\phi$-Decoding, we can try with the following command.

Firstly, create the environment and install the requirements. This implementation is accelerated and supported by vllm.

```bash
# env
conda create -n phi-decoding python==3.10
conda activate phi-decoding
pip install -r requirements.txt
```

Next, please modify the model path and data path in phi_decoding.py.

Finally, simply run the following command after the basic configuration:

```bash
python phi_decoding.py
```

## 🔧 PyPi Package

We are currently working on providing a PyPI package. Stay tuned !

## Citation

If you find it helpful, please kindly cite the paper.

```
@article{xu2025phi,
  title={$\phi$-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation},
  author={Xu, Fangzhi and Yan, Hang and Ma, Chang and Zhao, Haiteng and Liu, Jun and Lin, Qika and Wu, Zhiyong},
  journal={arXiv preprint arXiv:2503.13288},
  year={2025}
}
```
