# 一步一步实现INQ

## 一、训练模型

### 1.1、训练模型

```powershell
python train.py # 开始训练
python inference.py # 训练完成后推理 此时本例精度为78.96%
```

该脚本会训练resnet18，数据集为CIFAR100

每个epoch结束后记录在测试集上的精度，默认训练30代

使用Pytorch官网提供的ImageNet预训练模型，由于类别数不同，需要删除全连接层

默认取其中在测试集上精度最高的一次的权重保存为best.pth进入下一步

本实验在第22代时得到最佳权重，精度为78.96%

理论上精度越高，收敛地越久，稍后INQ重训练时越能证明其有效性（因为如果模型未收敛，仅普通重训练本身也可以提高精度）

> 仅作演示，如果想要更高精度，可以调整训练策略，并且增加训练代数
>
> 由于某些未知原因，推理时的测试精度大于训练时的测试精度，正在排查原因



## 二、量化模型

### 2.1、算子融合

```powershell
python fuse.py # 开始算子融合
python fuse_inference.py # 算子融合完成后推理，此时本例精度为78.96%
```

该代码在inference.py的基础上修改，将网络中的BatchNormal算子融合进附近卷积的权重中

a、删除网络中的BatchNormal算子

b、将准备融入的卷积算子的偏置打开并置零（存在BatchNormal时一般关闭卷积的偏置）

c、使用公式融合权重

d、保存融合后的权重

本步骤得到的权重为fuse.pth

### 2.2、结构优化

```powershell
python optimize.py # 开始结构优化
python optimize_inference.py # 结构优化完成后推理，此时本例精度为78.96%
```

该代码在fuse_inference.py的基础上修改，对网络进行优化，使后续量化更方便。

将所有待量化的卷积算子都用BasicConv2d类包起来，方便插入量化算子。

将第一层卷积用FirstConv2d类包起来，因为第一层后续操作与其他层不同。

需要修改权重名字，将修改后的权重保存起来。

本步骤得到的权重为optimize.pth

### 2.3、归一化权重

```powershell
python change_weights.py # 开始归一化权重
python quantize.py ./weights_max.pth # 归一化权重完成后推理，此时本例精度为77.59%量化方式导致精度下降了1.37个百分点
```

使用权重为optimize.pth。将特征图重放缩至统一值，将重放缩系数融入到权重中，保存此时的权重为weights_max.pth。由于需要记录Eltwise的还原系数，还要保存bn_bag.pth



## 三、重训练模型

### 3.1、预推理

```powershell
python record_C_W_Scale.py
```

使用2.3步得到的权重在校准集上预推理，得到C_S_bag.pth和W_S_bag.pth，即量化时需要用到的量化放缩系数Scale

### 3.2、固定50%重训练

```powershell
python retrain.py ./weights_max.pth 0.5 # 开始固定50%重训练
python quantize.py ./0.5_best.pth # 固定50%重训练完成后推理，此时本例精度为78.31%
```

使用weights_max.pth、bn_bag.pth、C_S_bag.pth、W_S_bag.pth文件，修改量化算子和优化器算子的代码，使其量化和冻结50%的权重。

每个epoch结束后记录在测试集上的精度，默认训练30代

默认取其中在测试集上精度最高的一次的权重保存为0.5_best.pth进入下一步

本实验在第19代时得到最佳权重，精度为78.82%

### 3.3、固定75%重训练

如果对3.2步的精度不满意，将固定比例调整至75%再次迭代。重复步骤，直到精度符合要求或固定至100%。

```powershell
python retrain.py ./0.5_best.pth 0.75 # 开始固定75%重训练
python quantize.py ./0.75_best.pth # 固定75%重训练完成后推理
```

