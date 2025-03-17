<h1 align="center">
φ-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation
</h1>

<p align="center">
  <a href="https://xufangzhi.github.io/symbol-llm-page/"><b>[🌐 Website]</b></a> •
  <a href="https://arxiv.org/abs/2311.09278"><b>[📜 Paper]</b></a> •
  <a href="https://github.com/xufangzhi/phi-Decoding"><b>[🐱 GitHub]</b></a>
  
</p>

<p align="center">
Repo for "<a href="https://arxiv.org/abs/2311.09278" target="_blank">φ-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation</a>"
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

Next, simply run the following command after the basic configuration:

```bash
python phi_decoding.py
```


## 🔧 PyPi Package

We are currently working on providing a PyPI package. Stay tuned !


## Citation

If you find it helpful, please kindly cite the paper.

```
@article{xu2023symbol,
  title={Symbol-LLM: Towards Foundational Symbol-centric Interface For Large Language Models},
  author={Xu, Fangzhi and Wu, Zhiyong and Sun, Qiushi and Ren, Siyu and Yuan, Fei and Yuan, Shuai and Lin, Qika and Qiao, Yu and Liu, Jun},
  journal={arXiv preprint arXiv:2311.09278},
  year={2023}
}
```
