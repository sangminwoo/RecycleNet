DL-for-recycle
==============
-  Baseline Network: ResNet
-  Attention Module: BotoxNet

Dataset(TrashNet[1]: https://github.com/garythung/trashnet)
-----------------------------------------------------------
* Total: 2527 (contains 6 classes)
  -  Glass 501
  -  Paper 594
  -  Cardboard 403
  -  Plastic 482
  -  Metal 410
  -  Non-recyclable Trash 137

* Train/Val/Test set: 70/13/17
* Data Augmentation

Experiment
----------
* Loss Function: Cross Entropy Loss
* Optimizer: SGD
* Initial Learning Rate: 2e-4
* 50 epoch
* For every 20 epoch, learning rate = learning rate * 1/10

Ablation Study
--------------
* Non Pre-trained Model vs. Pre-trained Model (Transfer Learning)

|        Method        | Accuracy(%) | Parameters(M) |
|----------------------|-------------|---------------|
|       ResNet18       |   70.302    |      11.18    |
|       ResNet34       |   64.965    |      21.29    |
|       ResNet50       |   58.701    |      23.52    |
| Pre-trained ResNet18 |   90.023    |      11.18    |
| Pre-trained ResNet34 |   93.271    |      21.29    |
| Pre-trained ResNet50 |   93.735    |      23.52    |


* Attention Module(SENet vs. CBAM vs. BotoxNet)

|        Method        | Accuracy(%) | Parameters(M) |
|----------------------|-------------|---------------|
|  ResNet18 + SE[2]    |   87.703    |      11.27    |
|  ResNet34 + SE[2]    |   88.863    |      21.45    |
|  ResNet50 + SE[2]    |   91.879    |      26.05    |
|  ResNet18 + CBAM[3]  |   79.814    |      11.27    |
|  ResNet34 + CBAM[3]  |   81.439    |      21.45    |
|  ResNet50 + CBAM[3]  |   82.135    |      26.05    |
|  ResNet18 + Botox    |   93.039    |      11.24    |
|  ResNet34 + Botox    |   93.968    |      21.35    |
|  ResNet50 + Botox    |   94.2      |      24.15    |


* Channel Attention & Spatial Attention

|  Network ablation  | Accuracy(%) | Parameters(M) |
|--------------------|-------------|---------------|
|      ResNet18      |    90.023   |     11.18     |
|    ResNet18 + s    |    92.807   |     11.20     |
|  ResNet18 + s + c  |    93.039   |     11.24     |

| Combination ablation | Accuracy(%) | Parameters(M) |
|----------------------|-------------|---------------|
|          Mul         |    91.647   |     11.24     |
|          Max         |    92.575   |     11.24     |
|          Sum         |    93.039   |     11.24     |

Result
-------
* Attention Module

![Alt text](/data/images/Attention.jpg)

* Attention Visualization

![Alt text](/data/images/Attention%20Visualization.jpg)

References
----------
| # | Reference |                    Link                      |
|---|-----------|----------------------------------------------|
| 1 | TrashNet  | https://github.com/garythung/trashnet        |
| 2 | SENet     | https://github.com/hujie-frank/SENet         |
| 3 | CBAM      | https://github.com/Jongchan/attention-module |
