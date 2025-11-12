

<div align="center">
<h1>VTON-HandFit: Virtual Try-on for Arbitrary Hand Pose Guided by Hand Priors Embedding</h1>

<a href='https://vton-handfit.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/pdf/2408.12340'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

This is the offical code repository of “[VTON-HandFit: Virtual Try-on for Arbitrary Hand Pose Guided by Hand Priors Embedding.](https://arxiv.org/pdf/2408.12340)” 

![image-20240810162757820](./examples/figure1_0827.png)

## Installation
We recommend creating a virtual environment for Handfit, you can create it with
```
conda env create -f environment.yml
conda activate handfit
pip install torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
```

Install hamer
```
cd hamer
pip install -e .[all]
pip install pyopengl==3.1.4
```

Install pytorch3d
```
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
```
Install DensePose
```
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
```

Install ViTPose
```
cd third-party/ViTPose/
pip install -v -e .
```

## Citation
```
@article{liang2024vton,
  title={VTON-HandFit: Virtual Try-on for Arbitrary Hand Pose Guided by Hand Priors Embedding},
  author={Liang, Yujie and Hu, Xiaobin and Jiang, Boyuan and Luo, Donghao and Wu, Kai and Han, Wenhui and Jin, Taisong and Wang, Chengjie},
  journal={arXiv preprint arXiv:2408.12340},
  year={2024}
}
```
## License
The codes and checkpoints in this repository are under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).