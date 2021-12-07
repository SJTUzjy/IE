## Log

11.30 
+ The data cleaner script has been finished, clean the data without label or with a too short comment.

12.1 
+ The dataset class 'Mydataset' has been created, with Dataloader, it returns a list, [comment, label].

12.4  

* 修改了dataset的定义，加入了bert需要的tokenizer
* 定义了一个原始的模型，需要进一步优化，详见`model.py`
* 编写了测试和训练的函数，以及保存和加载模型的函数，详见`main.py`
* 使用cpu进行了一些测试，基本可以运行，但是可能出现未知错误（比如保存和加载函数）
* 参考视频：https://www.bilibili.com/video/BV1Dz4y1d7am

12.7
+ Develop the dataclean_script, now it will generate non-english dataset in default