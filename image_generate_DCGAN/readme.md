# 任务  
使用深度卷积生成对抗网络（DCGAN）生成人物头像  

# 说明  
1. **image_collect.py**  
爬虫文件，爬取百度搜索的头像结果 ： 爬取图片链接--> 下载图片到本地  
2. **dataread.py**  
将爬取到的图片进行灰度转换，和维度变换，使统一,这里设置 resize=112,112  
并变换后的统一格式分批 ，设置批 batchsize=64   
3. **create_DCGAN.py**  
核心运行文件  
**生成器共6层**：  
> 全连接层 & 维度转换: 1000--> 7,7,256  
> 转置卷积1:   
> &ensp;&ensp;filter=128,kernel_size=(5,5),s=(1,1),activation=带泄露的ReLU(后2到4层同)   outputsize=7,7,128  
> 转置卷积2：  
> &ensp;&ensp;filter=64,s=(2,2)  outputsize=14,14,64  
> 转置卷积3：    
> &ensp;&ensp;filter=32,s=(2,2)  outputsize=28,28,32  
> 转置卷积4：   
> &ensp;&ensp;filter=16,s=(2,2)  outputsize=56,56,16  
> 转置卷积5：  
> &ensp;&ensp;filter=1,s=(2,2)  outputsize=112,112,1  # 输出  

**判别器共5层**：  
> 卷积1 :  
> &ensp;&ensp;filter=32,kernel_size=(5,5),s=(2,2),activation=带泄露的ReLU(后2到3层同)     
> 卷积2:   
> &ensp;&ensp;filter=64     
> 卷积3：  
> &ensp;&ensp;filter=128  
> 扁平层：展开维度      
> 输出层：输出1个值      

# 运行   
image number: ≈8500  
per epoch : 15mins+   
epochs=100 :30hours+  
# 结果    
（目前全量8500images，输入维度扩大到1000,迭代100次需要30hours+）
先使用5000左右的images，在输入维度为100情况下，效果如下：
在当前数据集的质量和数量下，迭代100次生成的结果虽然还没有人的样子，但是每20次输出生成结果，可以发现模型一直在进步。  
有充足时间和质量比较高的数据集的朋友们，可以使用此模型自己去训练试试，相信可以获得不错的结果。
![少量迭代观察](https://github.com/laura-zhang-cn/digital_image_preprocessing_and_recognition/blob/master/image_generate_DCGAN/imagesrst/generate_rst_epoch5_compare.png?raw=true)  

# 附：数据集概览  
![百度爬虫数据集截图](https://github.com/laura-zhang-cn/digital_image_preprocessing_and_recognition/blob/master/image_generate_DCGAN/imagesrst/%E7%99%BE%E5%BA%A6%E7%88%AC%E8%99%AB%E6%95%B0%E6%8D%AE%E9%9B%86%E6%88%AA%E5%9B%BE.png)
