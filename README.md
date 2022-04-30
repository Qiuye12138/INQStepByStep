# 一步一步实现INQ

## 一、训练模型

### 1.1、训练模型

```powershell
python train.py # 开始训练
python inference.py # 训练完成后推理 此时本例精度为80.28%
```

该脚本会训练`ResNet18`，数据集为`CIFAR100`

每个`Epoch`结束后记录在测试集上的精度，默认训练`300`代

使用`Pytorch`官网提供的`ImageNet`预训练模型，由于类别数不同，需要删除全连接层

默认取其中在测试集上精度最高的一次的权重保存为`best.pth`进入下一步

本实验在第`69`代时得到最佳权重，精度为`80.28%`

理论上精度越高，收敛地越久，稍后`INQ`重训练时越能证明其有效性（因为如果模型未收敛，仅普通重训练本身也可以提高精度）

该模型量化精度损失太小，只是演示`INQ`流程



## 二、量化模型

### 2.1、算子融合

```powershell
python fuse.py # 开始算子融合
python fuse_inference.py # 算子融合完成后推理，此时本例精度为80.28%
```

该代码在`inference.py`的基础上修改，将网络中的`BatchNormal`算子融合进附近卷积的权重中

a、删除网络中的`BatchNormal`算子

b、将准备融入的卷积算子的偏置打开并置零（存在`BatchNormal`时一般关闭卷积的偏置）

c、使用公式融合权重

d、保存融合后的权重

本步骤得到的权重为`fuse.pth`

### 2.2、结构优化

```powershell
python optimize.py # 开始结构优化
python optimize_inference.py # 结构优化完成后推理，此时本例精度为80.28%
```

该代码在`fuse_inference.py`的基础上修改，对网络进行优化，使后续量化更方便。

将所有待量化的卷积算子都用`BasicConv2d`类包起来，方便插入量化算子。

将第一层卷积用`FirstConv2d`类包起来，因为第一层后续操作与其他层不同。

需要修改权重名字，将修改后的权重保存起来。

本步骤得到的权重为`optimize.pth`

### 2.3、归一化权重

```powershell
python change_weights.py # 开始归一化权重
python quantize.py ./weights_max.pth # 归一化权重完成后推理，此时本例精度为80.09%，量化导致精度下降了0.19个百分点
```

使用权重为`optimize.pth`。将特征图重放缩至统一值，将重放缩系数融入到权重中，保存此时的权重为`weights_max.pth`。由于需要记录`Eltwise`的还原系数，还要保存`bn_bag.pth`



## 三、重训练模型

### 3.1、预推理

```powershell
python record_C_W_Scale.py
```

使用2.3步得到的权重在校准集上预推理，得到`C_S_bag.pth`和`W_S_bag.pth`，即量化时需要用到的量化放缩系数Scale

### 3.2、固定50%重训练

```powershell
python retrain.py ./weights_max.pth 0.5 # 开始固定50%重训练
python quantize.py ./0.5_best.pth # 固定50%重训练完成后推理，此时本例精度为80.16%，量化导致精度下降了0.12个百分点
```

使用`weights_max.pth`、`bn_bag.pth`、`C_S_bag.pth`、`W_S_bag.pth`文件，修改量化算子和优化器算子的代码，使其量化和冻结`50%`的权重。

每个`Epoch`结束后记录在测试集上的精度，默认训练`300`代

默认取其中在测试集上精度最高的一次的权重保存为`0.5_best.pth`进入下一步

本实验在第`10`代时得到最佳权重，精度为`80.16%`

### 3.3、固定75%重训练

如果对3.2步的精度不满意，将固定比例调整至`75%`再次迭代。重复步骤，直到精度符合要求或固定至`100%`。

```powershell
python retrain.py ./0.5_best.pth 0.75 # 开始固定75%重训练
python quantize.py ./0.75_best.pth # 固定75%重训练完成后推理
```

