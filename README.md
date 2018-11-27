# CHIP-Task2

CHIP 2018 Task2 DUTNLP.未来数据研究所 rank 5

CHIP T2文件夹为clean之后的比赛代码，下面介绍一下各个文件夹

--Models

----BaseModel.py 后续模型基类

----toymodel.py  比赛初期的基类测试程序（可能因为基类接口更新失效）

--All_model

----Bi_GRU2_based_Model.py	baseline模型，孪生兄弟网络

----ESIM.py					比赛基础模型，ESIM

----play_model.py			比赛中用于调试的模型，可以经过简单的修改跑字符级、词级、字-词级的网络

--utils.py  工具代码，包括计算F值、提交文件间的皮尔逊系数，利用问题相关性的传递扩充训练集，投票法、平均法融合模型。

--config.py	默认参数代码

--run.py    原定整体工程运行入口（比赛中为了调试方面，运行入口改在各个模型文件里）

--instances

----char_embed.txt	用于本工程读取的字向量

----word_embed.txt	用于本工程读取的词向量

--preprocess

----csv_reader.py		csv的读取代码

----example_reader.py	读取预处理结果生成模型输入的文件（basemodel里集成了此流程）

----feature_reader.py	生成图特征的代码（对模型融合未起到作用，未用于最终的提交）

--layers

----DecomposableAttention.py	用于ESIM模型的层，用于计算相似矩阵，公式为相减绝对值，是尝试了多种公式之后得到的最好的。



> From DUTNLP.未来数据研究所
