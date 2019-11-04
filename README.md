RecycleNet
================================
In the era of mass production and mass consumption, trash disposal has become an important national issue. With this trend, the social and economic importance of ***trash collection and reusing*** is increasing. An alternative is to allow the machine to classify automatically once the user discharge the trash regardless of the material.

Using two methods for creating an ***effective trash classification model*** using only a small number of annotated trash images(2527).

***1) Transfer learning: Using ImageNet pre-trained model***  
***2) Effective feature learning with attention module***

To demonstrate that the proposed methodologies were effective, a large number of ablation studies were conducted and were more effective than state-of-the-art attention modules.

-  Backbone Network: ResNet
-  Attention Module: RecycleNet

Requirements
-----------
Install all the python dependencies using pip:
```
$ git clone https://github.com/sangminwoo/RecycleNet.git
$ cd RecycleNet
$ pip install -r requirements.txt
```
* PyTorch is not inside. Please go to [official website](https://pytorch.org/get-started/locally/).

Data Preparation(TrashNet[1]: https://github.com/garythung/trashnet)
--------------------------------------------------------------------
* Total: 2527 (contains 6 classes)
  -  Glass 501
  -  Paper 594
  -  Cardboard 403
  -  Plastic 482
  -  Metal 410
  -  Non-recyclable Trash 137

* Train/Val/Test set: 70/13/17
* Data Augmentation

* :warning: You may use *additional_dataset.zip* as another version of dataset. But if you use both of them on training phase, it will increase intra-class variance thus will leads to decrease of accuracy. Maybe you can try to use it for just testing true-generalizability on totally different dataset.(In terms of real world problem, trashes have high intra-class variance so it's very important!)

Data Augmentation(Albumentations[4])
------------------------------------
```
$ python augmentation.py --root_dir $ROOT --save_dir $SAVE --probability $PROB
```
**$ROOT**: 'dataset-resized/' (default)  
**$SAVE**: 'augmented/' (default)  
**$PROB**: low(default), mid, high (probability of applying the transform)  

Training
---------
Without pre-train(Training from scratch)
```
$ python main.py --gpu $GPUNUM --arch $ARCHITECTURE --no_pretrain
```

Without Attention Module
```
$ python main.py --gpu $GPUNUM --arch $ARCHITECTURE
```

With Attention Module
```
$ python main.py --gpu $GPUNUM --arch $ARCHITECTURE --use_att --att_mode $ATT
```
**$GPUNUM**: 0; 0,1; 0,3; 0,1,2; whatever  
**$ARCHITECTURE**: resnet18_base(default), resnet34_base, resnet52_base, resnet101_base, resnet152_base  
**$ATT**: ours(default), cbam, se  

You can find more configurations in *main.py*.

Evaluation
----------
```
$ python main.py --gpu $GPUNUM --resume save/model_best.pth.tar --use_att -e
```
**$resume**: save/model_best.pth.tar(default) (If you have changed save path, you should change resume path as well.)  
**$e** (or evaluate): set evaluation mode

Webcam Inference
----------------
```
$ python webcam.py --resume save/model_best_pth.tar
```

Configuration
-------------
* Loss Function: Cross Entropy Loss
* Optimizer: SGD
* Initial Learning Rate: 2e-4
* epochs: 100
* For every 40 epochs, learning rate = learning rate * 1/10

Attention Module
----------------
![Alt text](/images/Attention.jpg)

* Attention Module
  - **Attention mechanism** learns parameters with a high weight for important features and a low weight for unnecessary features.  
  𝒙′′ = (𝒙,𝜽) ∗ 𝑨(𝒙′, ∅), 𝒘𝒉𝒆𝒓𝒆 𝟎 ≤ 𝑨(𝒙′, ∅) ≤ 𝟏.  
  𝒙: Input Feature, 𝒙′: CNN or later features, 𝒙′′: Output Feature,  
  θ, ∅: learable parameters, A: Attention operation
  
  - When looking at the network from a **forward perspective**, the features are refined through attention modules.  
  (𝒅(𝒙, 𝜽)𝑨(𝒙′, ∅))/𝒅𝜽 = (𝒅(𝒙, 𝜽))/𝒅𝜽 ∗ 𝑨(𝒙′, ∅), 𝒘𝒉𝒆𝒓𝒆 𝟎 ≤ 𝑨(𝒙′, ∅) ≤ 𝟏.  
  - From a **backward perspective**, the greater the attention value, the greater the gradient value, so effective learning is achieved.

![Alt text](/images/Attention%20Visualization.jpg)

* Attention Visualization
  - **Visualization comparison** of feature map extracted after the last convolution block.
  - **ResNet18 + Ours** vs. ResNet18(baseline)
  - While **ResNet18 + Ours** successfully classified, ResNet18 failed classification.
  - Feature map shows that when Attention module is inserted, it attend more precisely on the **object extent**.

Ablation Study
--------------
* Non Pre-trained Model vs. Pre-trained Model (Transfer Learning)

|        Method        | Accuracy@1  | Parameters(M) |
|----------------------|-------------|---------------|
|       ResNet18       |   70.302    |      11.18    |
|       ResNet34       |   64.965    |      21.29    |
|       ResNet50       |   58.701    |      23.52    |
| Pre-trained ResNet18 |   **90.023**    |      11.18    |
| Pre-trained ResNet34 |   **93.271**    |      21.29    |
| Pre-trained ResNet50 |   **93.735**    |      23.52    |


* Attention Module(SENet vs. CBAM vs. Ours)

|        Method        | Accuracy@1  | Parameters(M) |
|----------------------|-------------|---------------|
|  ResNet18 + SE[2]    |   87.703    |      11.27    |
|  ResNet34 + SE[2]    |   88.863    |      21.45    |
|  ResNet50 + SE[2]    |   91.879    |      26.05    |
|  ResNet18 + CBAM[3]  |   79.814    |      11.27    |
|  ResNet34 + CBAM[3]  |   81.439    |      21.45    |
|  ResNet50 + CBAM[3]  |   82.135    |      26.05    |
|  ResNet18 + Ours     |   **93.039**    |      11.24    |
|  ResNet34 + Ours     |   **93.968**    |      21.35    |
|  ResNet50 + Ours     |   **94.2**      |      24.15    |


* Channel Attention & Spatial Attention

|  Network ablation  | Accuracy@1  | Parameters(M) |
|--------------------|-------------|---------------|
|      ResNet18      |    90.023   |     11.18     |
|    ResNet18 + s    |    92.807   |     11.20     |
|  ResNet18 + s + c  |    **93.039**   |     11.24     |

| Combination ablation | Accuracy@1  | Parameters(M) |
|----------------------|-------------|---------------|
|          Mul         |    91.647   |     11.24     |
|          Max         |    92.575   |     11.24     |
|          Sum         |    **93.039**   |     11.24     |

Conclusion
----------
While proposing deep-learning model which is specialized in trash classification, there was two difficult problems faced experimentally:

*1) Insufficiency of data set*  
*2) The absence of effective feature learning methods*  
was solved by **transfer learning and attention mechanism.**

The methodology proposed through quantitative and qualitative assessments was experimentally significant. Because the proposed method exhibits significant performance improvements without significantly increasing the number of parameters, it is expected that the experimental value is also high for other applications.

Reference
----------
| # | Reference      |                    Link                      |
|---|----------------|----------------------------------------------|
| 1 | TrashNet       | https://github.com/garythung/trashnet        |
| 2 | SENet          | https://github.com/hujie-frank/SENet         |
| 3 | CBAM           | https://github.com/Jongchan/attention-module |
| 4 | Albumentations | https://github.com/albu/albumentations       |

Acknowledgement
---------------
We appreciate much the dataset [TrashNet](https://github.com/garythung/trashnet) and the well organized code [CBAM](https://github.com/Jongchan/attention-module). Our codebase is mostly built based on them.
