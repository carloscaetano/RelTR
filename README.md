
# RelTR: Relation Transformer for Scene Graph Generation

PyTorch Implementation of the Paper [**RelTR: Relation Transformer for Scene Graph Generation**](https://arxiv.org/abs/2201.11460v3)

Different from most existing advanced approaches that infer the **dense** relationships between all entity proposals, our one-stage method can directly generate a **sparse** scene graph by decoding the visual appearance. If our work is helpful for your research, please cite our publication:
```
@article{cong2023reltr,
  title={Reltr: Relation transformer for scene graph generation},
  author={Cong, Yuren and Yang, Michael Ying and Rosenhahn, Bodo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```

# 1. Installation
Download **RelTR Repo** with:
```
git clone https://github.com/carloscaetano/RelTR.git
cd RelTR
```
Change to [adaptations_to_run_PF](https://github.com/carloscaetano/RelTR/tree/adaptations_to_run_PF) branch with:
```
git checkout adaptations_to_run_PF
```

## For Inference
:smile: It is super easy to configure the RelTR environment.

If you want to **infer an image**, only python=3.6, PyTorch=1.6, scipy and matplotlib are required!
You can configure the environment as follows:
```
# create a conda environment 
conda create -n reltr python=3.6
conda activate reltr

# install packages
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
conda install matplotlib scipy
```

# 2. Usage

## Inference
a) Download our [RelTR model](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) pretrained on the Visual Genome dataset and put it under 
```
ckpt/checkpoint0149.pth
```
b) Infer the relationships in a folder image with the command:
```
python inference.py --folder_path $FOLDER_PATH --outputfolder $OUTPUTFOLDER --resume $MODEL_PATH
```
c) If necessary to infer with cpu, just pass the device argument:
```
python inference.py --folder_path $FOLDER_PATH --outputfolder $OUTPUTFOLDER --resume $MODEL_PATH --device cpu
```
