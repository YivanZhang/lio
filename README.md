# LIO: Learning from Indirect Observations

### A package for weakly supervised learning research based on PyTorch

<a href="https://github.com/YivanZhang/lio/blob/master/LICENSE">
  <img alt="license" src="https://img.shields.io/github/license/YivanZhang/lio">
</a>
<a href="https://pypi.org/project/lio/">
  <img alt="pypi" src="https://badge.fury.io/py/lio.svg" alt="PyPI version">
</a>

## Installation

```sh
pip install lio
```
or
```sh
git clone https://github.com/YivanZhang/lio.git
pip install -e .
```

Most of the modules are designed as small (higher-order) functions.  
Feel free to copy-paste only what you need for your existing workflow to reduce dependencies.

## References

- **Learning from Indirect Observations**  
  Yivan Zhang, Nontawat Charoenphakdee, and Masashi Sugiyama  
  [[arXiv]](https://arxiv.org/abs/1910.04394)

- **Learning from Aggregate Observations**  
  Yivan Zhang, Nontawat Charoenphakdee, Zhenguo Wu, and Masashi Sugiyama  
  [[arXiv]](https://arxiv.org/abs/2004.06316)
  [[NeurIPS'20]](https://proceedings.neurips.cc/paper/2020/hash/5b0fa0e4c041548bb6289e15d865a696-Abstract.html)
  [[poster]](posters/neurips20_aggregate_observations.pdf)


- **Learning Noise Transition Matrix from Only Noisy Labels**  
  **via Total Variation Regularization**  
  Yivan Zhang, Gang Niu, and Masashi Sugiyama  
  [[arXiv]](https://arxiv.org/abs/2102.02414)
  [[code]](/ex/transition-matrix)

- **Approximating Instance-Dependent Noise**  
  **via Instance-Confidence Embedding**  
  Yivan Zhang and Masashi Sugiyama  
  [[arXiv]](https://arxiv.org/abs/2103.13569)
  [[code]](/ex/instance-embedding)
