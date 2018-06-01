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
| Model | Param Size (M) | Loss | Distance |Rank-1(%) | mAP (%) | RK:Rank-1(%) | RK:mAP (%) | 
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Resnet50 | 25.05 | softmax                    | Global |81.2 | 64.2 |83.4|76.4|
| Resnet50 | 25.05 | softmax+label smooth       | Global |82.6 | 64.4 |84.0|76.8|
| Resnet50 | 25.05 | softmax+trihard            | Global |86.4 | 70.9 |88.5|83.3|
| Resnet50 | 25.05 | AlignedReID                | Global |87.5 | 72.5 |89.0|84.7|
| Resnet50 | 25.05 | AlignedReID                | Local  |87.5 | 71.9 |89.6|84.9|
| Resnet50 | 25.05 | AlignedReID                | Local  |88.4 | 73.2 |90.2|85.5|


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
python train_alignedreid.py  -d market1501 -a resnet50 --test_distance global_local
```

**Note:** You can add your experimental settings for 'args'
# Test
```bash
python train_alignedreid.py -d market1501 -a resnet50 --evaluate --resume saved-models/best_model.pth.tar --save-dir log/resnet50-market1501 --test_distance global_local (--reranking)
```

**Note:** (--reranking) means whether you use 'Re-ranking with k-reciprocal Encoding (CVPR2017)' to boost the performance.
