# CNN+GloVe文本分类

## 使用资源
- **维基中文语料库** 原始的corpus语料库文件，可以根据模型的需要，自行训练出相应的模型。 [Github](https://github.com/zhezhaoa/ngram2vec) | [百度网盘](https://pan.baidu.com/s/1kURV0rl)
- **GloVe预训练模型** 由维基中文语料库训练好的词向量文件。向量维度=50，词表大小=83W+。代码使用了这个预训练模型。[Github](https://github.com/YingZhuY/GloVe_Chinese_word_embedding) | [百度网盘（提取码543x）](https://pan.baidu.com/s/1tFbPrh25H5PEp-i6ELQ8Ig)

## 设计思路
使用CNN搭建网络模型，并将词嵌入层更换为由GloVe词向量表示的词嵌入向量。参考资料：[新闻分类器的模型训练与单篇分类（cnn+word2vec）](https://blog.csdn.net/mawenqi0729/article/details/80700926)  
由于word2vec与glove都是将词表示为向量的形式，因此只要稍作处理，word2vec的gensim API可以直接加载glove的词向量文件，如同在使用word2vec一般，非常方便。

## 步骤
1.  **数据清洗** 读取`comments.csv`并提取其中的评论与评分。对评论使用正则表达式去除英文后（已完成），用jieba分词，并移除停用词。最后将评分与分词后的评论单独保存在一个文件中，并作为之后的数据集输入。
2.  **模型读取** 将`glove_vectors.txt`处理为gensim可以读取的形式后（直接在文本编辑器里完成了），读取词向量并保存为模型，之后就可以使用数据集对CNN网络进行训练。
3.  **网络训练** 编写一个卷积神经网络，输入为100（词数）×50（词向量维数）的数组，输出为0~1之间的预测值，判断情感正负。写好迭代器与其他杂项后，就可以进行网络训练了。