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
# Train
```bash
python train_class.py  -d market1501 -a resnet50 
```

**Note:** You can add your experimental settings for 'args'
# Test
```bash
python train_img_model_xent.py -d market1501 -a resnet50 --evaluate --resume saved-models/best_model.pth.tar --save-dir log/resnet50-market1501 (--reranking)
```

**Note:** (--reranking) means whether you use 'Re-ranking with k-reciprocal Encoding (CVPR2017)' to boost the performance.
