Implement the code from the paper[Paper](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf), all the architecture and parameters are the same as paper, the only different thing is that I trained the model with only 100 classes, because 1000 classes might take too much time. 

The accuracy and loss are shown as below:

<img src="https://github.com/AlgorithmicIntelligence/ZFNet_Pytorch/blob/master/README/Accuracy.png" width="450">

<img src="https://github.com/AlgorithmicIntelligence/ZFNet_Pytorch/blob/master/README/Loss.png" width="450">

The results of the feature visualization are shown as below:

(1) Same Layer:
<img src="https://github.com/AlgorithmicIntelligence/ZFNet_Pytorch/blob/master/README/FeatureVisualization_SameLayer.jpg" width="900">

(2) From shallow to deep:
<img src="https://github.com/AlgorithmicIntelligence/ZFNet_Pytorch/blob/master/README/FeatureVisualization_DifferentLayer.jpg" width="900">
