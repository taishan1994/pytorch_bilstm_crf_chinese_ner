# 基于pytorch+bilstm_crf的中文命名实体识别
已训练好的模型可以在下面地址下载：<br>
链接：<a href="https://pan.baidu.com/s/1D1Z2YXeCQNmCz3CkNuYQTw">https://pan.baidu.com/s/1D1Z2YXeCQNmCz3CkNuYQTw</a><br>
提取码：86ar<br>
下载好之后放在chceckpoints文件夹下，也可以自己自行训练、验证、测试和预测。<br>
vocab.txt是结合训练集、验证集、测试集里面的字生成的，可能比较少，针对于自己的数据集，可以自行生成相关的字表。


# 依赖
```python
python==3.6
pytorch==1.6.0
pytorch-crf
```

# 文件说明
--checkpoints：模型保存的位置<br>
--data：数据位置<br>
--|--cnews：数据集名称<br>
--|--|--raw_data：原始数据存储位置<br>
--|--|--final_data：存储标签、词汇表等<br>
--logs：日志存储位置<br>
--utils：辅助函数存储位置，包含了解码、评价指标、设置随机种子、设置日志等<br>
--config.py：配置文件<br>
--dataset.py：数据转换为pytorch的DataSet<br>
--main.py：主运行程序<br>
--main.sh：运行命令<br>
--models.py：模型<br>
--process.py：预处理，主要是处理数据然后转换成DataSet<br>

# 运行命令
```python
python main.py --data_dir="../data/cnews/final_data/" --log_dir="./logs/" --output_dir="./checkpoints/" --num_tags=33 --seed=123 --gpu_ids="0" --max_seq_len=128 --lr=3e-5 --train_batch_size=32 --train_epochs=3 --eval_batch_size=32 --dropout=0.3 --dropout2=0.5  --hidden_size=128
```

# 结果
## 训练和验证（部分记录）
```python
2021-07-26 20:54:58,212 - INFO - main.py - train - 82 - 【train】 epoch：59 step:7198/7200 loss：2.283885
2021-07-26 20:54:58,464 - INFO - main.py - train - 82 - 【train】 epoch：59 step:7199/7200 loss：2.496460
2021-07-26 20:54:58,714 - INFO - main.py - train - 82 - 【train】 epoch：59 step:7200/7200 loss：1.282612
2021-07-26 20:55:00,311 - INFO - main.py - train - 89 - 【dev】 loss：82.923920 precision：0.6591 recall：0.8510 micro_f1：0.7429
```
## 测试
```python
2021-07-26 20:55:02,553 - INFO - main.py - test - 185 -           precision    recall  f1-score   support

     PRO       0.23      0.55      0.32        11
     ORG       0.13      0.75      0.22        85
    CONT       1.00      1.00      1.00        28
    RACE       1.00      0.93      0.96        14
    NAME       0.99      0.88      0.93       112
     EDU       0.98      0.90      0.94       112
     LOC       0.00      0.00      0.00         6
   TITLE       0.95      0.91      0.92       769

micro-f1       0.66      0.88      0.76      1137
```
## 预测
```python
2021-07-26 20:55:03,116 - INFO - main.py - predict - 203 - 虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。
2021-07-26 20:55:03,116 - INFO - main.py - predict - 204 - {'NAME': [('虞兔良', 0)], 'RACE': [('汉族', 17)], 'CONT': [('中国国籍', 20)], 'TITLE': [('中共党员', 40), ('经济师', 49)], 'EDU': [('MBA', 45)]}
```

## 最后
这里是随机初始化字向量，也可以加载相关的预训练的字向量，应该会有更好的结果。