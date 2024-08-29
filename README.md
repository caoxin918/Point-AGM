# Note: If your work uses this algorithm or makes improvements based on it, please be sure to cite this paper. Thank you for your cooperation.

# 注意：如果您的工作用到了本算法，或者基于本算法进行了改进，请您务必引用本论文，谢谢配合。

# Point-AGM : Attention Guided Masked Auto-Encoder for Joint Self-supervised Learning on Point Clouds

*Jie Liu*<sup>1,2</sup>,
*Mengna Yang*<sup>3,4</sup>,
*Yu Tian*<sup>5</sup>,
*Yancui Li*<sup>1,2</sup>,
*Da Song*<sup>3,4</sup>,
*Kang Li*<sup>3,4</sup>,
*Xin Cao*<sup>3,4</sup>

1 Henan Normal University, College of Computer and Information Engineering, China <br>
2 Big Data Engineering Laboratory for Teaching Resources & Assessment of Education Quality, China <br>
3 Northwest University, School of Information Science and Technology, China <br>
4 National and Local Joint Engineering Research Center for Cultural Heritage Digitization, China <br>
5 Bioresource Engineering Department, McGill University, Montreal, QC, Canada

## Installation

### 1. Dependencies

```bash
pip install -r requirements.txt

# Compile C++ extensions
cd ./extensions/chamfer_dist && python setup.py install --user
```

### 2. Datasets

Please download the used dataset with the following links:

- ShapeNet55 [https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view?usp=sharing](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view?usp=sharing)
- ModelNet40 [https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)
- ScanObjectNN [https://hkust-vgd.github.io/scanobjectnn](https://hkust-vgd.github.io/scanobjectnn)
- ModelNet Few Shot [https://drive.google.com/drive/folders/1gqvidcQsvdxP_3MdUr424Vkyjb_gt7TW?usp=sharing](https://drive.google.com/drive/folders/1gqvidcQsvdxP_3MdUr424Vkyjb_gt7TW?usp=sharing)
- ShapeNetPart [https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)

#### Make sure to put the files in the following structure:

```
|-- ROOT
|	|-- data
|		|-- ShapeNet55
|		|-- modelnet40_normal_resampled
|		|-- ScanObjectNN
|		|-- ModelNetFewshot
|		|-- shapenetcore_partanno_segmentation_benchmark_v0_normal
```

### 3. Pre-training

Due to involve normal vector operations, Please perform the following operations:
- open lines 162-164, 182-185 and close lines 158-160, 174-179 of the point2vec/modules/pointnet.py 
- open lines 342-354 of the point2vec/models/point2vec.py.
- close lines 144 of the point2vec/models/point2vec.py.

```bash
./scripts/pretraining_shapenet.bash --data.in_memory true
```
If you want to test other mask strategies, please close lines 342-354 and open lines 144, 340 of the point2vec/models/point2vec.py

### 4. Downstream

Due to not involve normal vector operations, Please perform the opposite of the pre-training.

#### Classification on ScanObjectNN

```bash
./scripts/classification_scanobjectnn.bash --config configs/classification/_pretrained.yaml --model.pretrained_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/XXX.ckpt
```

#### Classification on ModelNet40

```bash
./scripts/classification_modelnet40.bash --config configs/classification/_pretrained.yaml --model.pretrained_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/XXX.ckpt 
```

#### Voting on ModelNet40

```bash
./scripts/voting_modelnet40.bash --finetuned_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/epoch=XXX-step=XXXXX-val_acc=0.XXX.ckpt
```

#### Classification on ModelNet Few-Shot

You may also pass e.g. `--data.way 5` or `--data.shot 20` to select the desired m-way&ndash;n-shot setting.

```bash
for i in $(seq 0 9);
do
    SLURM_ARRAY_TASK_ID=$i ./scripts/classification_modelnet_fewshot.bash --model.pretrained_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/XXX.ckpt
done
```

#### Part segmentation on ShapeNetPart

```bash
./scripts/part_segmentation_shapenetpart.bash --model.pretrained_ckpt_path artifacts/point2vec-Pretraining-ShapeNet/XXXXXXXX/checkpoints/XXX.ckpt
```

## Citation
If you find Point-AGM useful in your research, please consider citing:
```
@article{PointAGM,
  title={Point-AGM : Attention Guided Masked Auto-Encoder for Joint Self-supervised Learning on Point Clouds},
  author={Jie Liu, Mengna Yang, Yu Tian, Yancui Li, Da Song, Kang Li, Xin Cao},
  year={2024},
}
```

## Acknowledgements 
We would like to thank and acknowledge referenced codes from the following repositories:

https://github.com/bytedance/ibot <br>
https://github.com/yichen928/STRL <br>
https://github.com/gkakogeorgiou/attmask <br>
https://point2vec.ka.codes/