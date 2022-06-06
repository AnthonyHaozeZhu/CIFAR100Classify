# 复现与新的方法

> 朱浩泽 计算机科学与技术 1911530 

### 文件组织架构

- Base 文件夹中是基础网络架构
- Coordinate Attention for Efficient Mobile Network Design文件夹中，是按照CVPR2021的Coordinate Attention for Efficient Mobile Network Design文章中所提出的Coordinate Attention一文中，改进的网络架构
- advanced是在Coordinate Attention的基础上，继续增加multi head self-attention改进的架构

### 环境依赖

```
python3
pytorch > 1.7
tqdm
tensorboard
torchvision
```



### 运行方法

每个文件的运行方法均相同，如下：

```bash
python main.py --data_path "数据集所在位置" --batch_size BATCH_SIZE --device "训练用的设备" --learning_rate 学习率 --epochs 训练轮次 --logdir "结果文件夹位置"
```

