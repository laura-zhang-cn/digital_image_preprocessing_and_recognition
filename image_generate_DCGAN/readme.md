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
> 全连接层 & 维度转换: 300--> 7,7,256  
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
image number: 5500  
per epoch : 6.5min   
epochs=20 :2hours+


