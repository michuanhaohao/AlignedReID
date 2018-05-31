# AlignedReID
Official Reproduce AlignedReID: Surpassing Human-Level Performance in Person Re-Identification using Pytorch.

```
@article{zhang2017alignedreid,
  title={Alignedreid: Surpassing human-level performance in person re-identification},
  author={Zhang, Xuan and Luo, Hao and Fan, Xing and Xiang, Weilai and Sun, Yixiao and Xiao, Qiqi and Jiang, Wei and Zhang, Chi and Sun, Jian},
  journal={arXiv preprint arXiv:1711.08184},
  year={2017}
}
```

#### Market1501
| Model | Param Size (M) | Loss | Rank-1/5/10 (%) | mAP (%) | RK:Rank-1/5/10 (%) | RK:mAP (%) | 
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Resnet50 | 25.05 | softmax | 81.2/92.2/94.6 | 64.2 |83.4/90.7/93/2|76.4|
| Resnet50 | 25.05 | softmax+label smooth | 82.6/92.3/95.1 | 64.4 |84.0/90.9/93.4|76.8|
| Resnet50 | 25.05 | softmax+trihard | 86.4/95.5/97.2 | 70.9 |88.5/94.1/95.7|83.3|
| Resnet50 | 25.05 | AlignedReID | 87.5/95.8/97.2 | 72.5 |89.0/94.7/96.1|84.7|
| Resnet50 | 25.05 | AlignedReID+label smooth | 88.7/95.8/97.7 | 74.1 |90.3/94.8/96.3|85.8|

# Prepare data
Create a directory to store reid datasets under this repo via
```bash
cd AlignedReID/
mkdir data/
```

If you wanna store datasets in another directory, you need to specify `--root path_to_your/data` when running the training code. Please follow the instructions below to prepare each dataset. After that, you can simply do `-d the_dataset` when running the training code. 

**Market1501** :
1. Download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html.
2. Extract dataset and rename to `market1501`. The data structure would look like:
```
market1501/
    bounding_box_test/
    bounding_box_train/
    ...
```
3. Use `-d market1501` when running the training code.

# Train
```bash
python train_class.py  -d market1501 -a resnet50 
```
```bash
python train_alignedreid.py  -d market1501 -a resnet50 --aligned
```

**Note:** You can add your experimental settings for 'args'
# Test
```bash
python train_img_model_xent.py -d market1501 -a resnet50 --evaluate --resume saved-models/best_model.pth.tar --save-dir log/resnet50-market1501 (--reranking)
```

**Note:** (--reranking) means whether you use 'Re-ranking with k-reciprocal Encoding (CVPR2017)' to boost the performance.
