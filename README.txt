Best Model-

RESNET-34 Model + Residual ( Very Fast in Training , probably due to Residual Block)- Best Execution Model
GELU() was giving CUDA Error so used ReLU6()
DropBlock Slowed down my training Accuracy and due to Time Constraint I avoided to use it.

Batch_Size=256
Learning Rate= 0.1
Epochs = 30 epochs with 97% Training Accuracy

Stem Layer- nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
nn.BatchNorm2d(num_features=64),
nn.ReLU

Stage1_MaxPool= nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

Convolution Layers-
nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=self.stride),
nn.BatchNorm2d(num_features=in_channels),
nn.ReLU(),
nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1),
nn.BatchNorm2d(num_features=in_channels),
nn.ReLU(),
nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1),
nn.BatchNorm2d(num_features=out_channels).

Stage Blocks-[in_channels, out_channels, stride(No.of times block will get executed)]
[64, 128, 3],
[128, 256, 4], 
[256, 512, 6], 
[512, 512, 3],

cls layer: 
nn.AdaptiveAvgPool2d((1,1)),
nn.Flatten(),
nn.Linear(512,num_classes),

Output- Classification Accuracy-80.3% Verification Accuracy- 0.943
**********************************************************************
Steps on how to execute the model-

1) Import all Libraries/ Drop Block as well
2) Download Data From Kaggle & unzip the files.
3) Implement the HyperParameters Cell
4) Implement Dataset & DataLoader
5) Implement RESNET-34 Model
6) Initialize Model object. 
7) Load the model from Google Drive if ou are saving from the epochs.
8) Load the Training Parameters. Criterion, Optimizer, Scheduler.
9) Train the Model
10) Load the model to the model object and evaluate it by running Validation Cell.
11) Run the Classification Test Set Class cell
12) Run the Dataset & DataLoader Cell
13) Test the model.
14) Create a .csv file
15) Submit it to Kaggle Competition. *** Classification Complete****
16) Run Cell to check the number of images in verification - training dataset.
17) Run the Verification Dataset Class
18) Initialize the Dataset & Dataloader for the class.
19) You can load the desired model(Optional)
20) Validate the Model 
21) Use Cosine Similarity Metric to find for the required set of images, and find AUC.
22) Run the Dataset & Dataloader for Test set.
23) Test the Model 
24) Create a csv file for the same.
25) Submit it to Kaggle. *** Verification Complete*** 

**********************************************************************
Experiment 1) Execute Very Simple Network

Model-
a) Batch Size=256
b) Learning Rate= 0.1
c) Epochs=20

Structure:

nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=4,padding=1),
nn.BatchNorm2d(num_features=64),
nn.ReLU(),

nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
nn.BatchNorm2d(num_features=128),
nn.ReLU(),

nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=2),
nn.BatchNorm2d(num_features=256),
nn.ReLU(),

nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=0),
nn.BatchNorm2d(num_features=512),
nn.ReLU(),

nn.AvgPool3d(kernel_size=(1,7,7)),
nn.Flatten()

Output - Classification: 38.3% Verification Accuracy: 0.864

Experiment 2) MobileNetV2 Model

Batch_Size=256
Learning Rate=0.1
Epochs=22

Model:
stem layer- Conv2D (in_channels=3, out_channels=32,Kernel=3,Stride=2)
stage_layers- (which include Residual Path)
Residual Block-
1) Feature Mixing, Conv2D(in_channels,hidden_dims,Kernel=1,stride=1) 
2) Spatial Mixing, Conv2D(hidden_dims,hidden_dims,Kernel=3,stride=stride)-Depth-wise Convolution.
3) Bottlenecking, Conv2D(hidden_dims,out_channels,Kernel=1,stride=1)-Point-wise Convolution  

Stage_layers[expand_ratio,channels,blocks,stride]
[6,  16, 1, 1],
[6,  24, 2, 2], 
[6,  32, 3, 2], 
[6,  64, 4, 2],
[6,  96, 3, 1], 
[6, 160, 3, 2], 
[6, 320, 1, 1]

Final Block- Conv2D (in_channels=320,out_channels=1280,Kernel=1,stride=1)

cls.layer- 
nn.AvgPool2d(kernel_size=(7,7),stride=1),
nn.Flatten(),
nn.Linear(1280,num_classes)

Output - Classification: 70% 

Experiment 3)RESNET-34 Model + Residual ( Very Fast in Training , probably due to Residual Block)- Best Execution Model
GELU() was giving CUDA Error so used ReLU6()
DropBlock Slowed down my training Accuracy and due to Time Constraint I avoided to use it.

Batch_Size=256
Learning Rate= 0.1
Epochs = 30 epochs with 97% Training Accuracy

Stem Layer- nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
nn.BatchNorm2d(num_features=64),
nn.ReLU

Stage1_MaxPool= nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

Convolution Layers-
nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=self.stride),
nn.BatchNorm2d(num_features=in_channels),
nn.ReLU(),
nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1),
nn.BatchNorm2d(num_features=in_channels),
nn.ReLU(),
nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1),
nn.BatchNorm2d(num_features=out_channels).

Stage Blocks-[in_channels, out_channels, stride(No.of times block will get executed)]
[64, 128, 3],
[128, 256, 4], 
[256, 512, 6], 
[512, 512, 3],

cls layer: 
nn.AdaptiveAvgPool2d((1,1)),
nn.Flatten(),
nn.Linear(512,num_classes),

Output- Classification Accuracy-80.3% Verification Accuracy- 0.943

Experiment 4: ResNet 50(Without Residual)- (Very Slow in Learning) 

Batch_Size=256
Learning Rate=0.1
Epochs=44

Stem Layer- nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
nn.BatchNorm2d(num_features=64),
nn.GELU(),

Max Pool_Stage- nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

Convolution Layers: (Block Execution)
nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=self.stride),
nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1),
nn.Conv2d(in_channels=in_channels,out_channels=in_channels*4,kernel_size=1,stride=1),
nn.Conv2d(in_channels=in_channels*4,out_channels=out_channels,kernel_size=1,stride=1),

stage_cfgs-[in_channels,out_channels,strides/No.of times Block Execution Takes Place]
[64,  128, 3],
[128, 256, 4], 
[256, 512, 6], 
[512, 1000, 3],  

stage_cls_layer- 
nn.AdaptiveAvgPool2d((1,1)),
nn.Flatten(),
nn.Linear(1000,num_classes),

Output- Classification Accuracy: 70% , Verification:0.89
