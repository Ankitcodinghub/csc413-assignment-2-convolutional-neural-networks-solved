# csc413-assignment-2-convolutional-neural-networks-solved
**TO GET THIS SOLUTION VISIT:** [CSC413 Assignment 2-Convolutional Neural Networks Solved](https://www.ankitcodinghub.com/product/csc413-assignment-2-convolutional-neural-networks-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;100109&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;5&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (5 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSC413 Assignment 2-Convolutional Neural Networks Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (5 votes)    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
<p style="text-align: center;">

Introduction

This assignment will focus on the applications of convolutional neural networks in various image processing tasks. The starter code is provided as a Python Notebook on Colab (https://colab. research.google.com/drive/11sH_zV08QvCAYrGDv83lI9-5mnmln3SV#scrollTo=JyzOT64xkqy6). First, we will train a convolutional neural network for a task known as image colourization. Given a greyscale image, we will predict the colour at each pixel. This a difficult problem for many reasons, one of which being that it is ill-posed: for a single greyscale image, there can be multiple, equally valid colourings. In the second half of the assignment, we will perform fine-tuning on a pre-trained semantic segmentation model. Semantic segmentation attempts to clusters the areas of an image which belongs to the same object (label), and treats each pixel as a classification problem. We will fine-tune a pre-trained conv net featuring dilated convolution to segment flowers from the Oxford17 flower dataset3.

Part A: Colourization as Classification (2 pts)

In this section, we will perform image colourization using a convolutional neural network. Given a grayscale image, we wish to predict the color of each pixel. We have provided a subset of 24 output colours, selected using k-means clustering4. The colourization task will be framed as a pixel-wise

1https://markus.teach.cs.toronto.edu/csc413-2020-01 2https://csc413-2020.github.io/assets/misc/syllabus.pdf 3 http://www.robots.ox.ac.uk/~vgg/data/flowers/17/ 4https://en.wikipedia.org/wiki/K-means_clustering

1

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
CSC413/2516 Winter 2020 with Professor Jimmy Ba Programming Assignment 2

classification problem, where we will label each pixel with one of the 24 colours. For simplicity, we measure distance in RGB space. This is not ideal but reduces the software dependencies for this assignment.

We will use the CIFAR-10 data set, which consists of images of size 32×32 pixels. For most of the questions we will use a subset of the dataset. The data loading script is included with the notebooks, and should download automatically the first time it is loaded.

Helper code for Part A is provided in a2-cnn.ipynb, which will define the main training loop as well as utilities for data manipulation. Run the helper code to setup for this question and answer the following questions.

1. Complete the model CNN, following the diagram provided below. Use the PyTorch layers nn.ReLU, nn.BatchNorm2d, nn.Upsample, and nn.MaxPool2d, but do not use nn.Conv2d. Instead, use the convolutional layer MyConv2d included in the file to better understand its in- ternals. Your CNN should be configurable by parameters kernel, num filters, num colours, and num in channels. In the diagram, num filters and num colours are denoted NF and NC respectively. Use the following parameterizations (if not specified, assume default pa- rameters):

<ul>
<li>MyConv2d: Number of output filters to use shown after the hyphen. For example, MyConv2D-2NF has 2NF output filters. Set kernel size to parameter kernel. Set number of input filters for first MyConv2d to num in channels.</li>
<li>nn.BatchNorm2d: the number of features to use for a layer is shown after the hyphen.</li>
<li>nn.Upsample: use scaling factor = 2</li>
<li>nn.MaxPool2d: use kernel size = 2</li>
</ul>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
&nbsp;

Grouping layers according to the diagram (those not separated by white space) by using nn.Sequential containers will aid implementation of the forward method.

<ol start="2">
<li>Run main training loop of CNN. This will train the CNN for a few epochs using the cross- entropy objective. It will generate some images showing the trained result at the end. Do these results look good to you? Why or why not?</li>
<li>Compute the number of weights, outputs, and connections in the model, as a function of NF and NC. Compute these values when each input dimension (width/height) is doubled. Report all 6 values.</li>
<li>Consider an pre-processing step where each input pixel is multiplied elementwise by scalar a, and is shifted by some scalar b. That is, where the original pixel value is denoted x, the new value is calculated y = ax + b. Assume this operation does not result in any overflows. How does this pre-processing step affect the output of the conv net from Question 1 and 2?</li>
</ol>
Part B: Skip Connections (2 pts)

A skip connection in a neural network is a connection which skips one or more layer and connects to a later layer. We will introduce skip connections to our previous model.

1. Add a skip connection from the first layer to the last, second layer to the second last, etc. That is, the final convolution should have both the output of the previous layer and the initial greyscale input as input. This type of skip-connection is introduced by Ronneberger et al. [2015], and is called a “UNet”. Following the CNN class that you have completed, complete the init and forward methods of the UNet class in Part B of the notebook.

Hint: You will need to use the function torch.cat.

<ol start="2">
<li>Train the model for at least 25 epochs and plot the training curve using a batch size of 100.</li>
<li>How does the result compare to the previous model? Did skip connections improve the validation loss and accuracy? Did the skip connections improve the output qualitatively? How? Give at least two reasons why skip connections might improve the performance of our CNN models.</li>
<li>Re-train a few more “UNet” models using different mini batch sizes with a fixed number of epochs. Describe the effect of batch sizes on the training/validation loss, and the final image output.</li>
</ol>
Part C: Fine-tune Semantic Segmentation Model (2 pts)

In the previous two parts, we worked on training models for image colourization. Now we will switch gears and perform semantic segmentation by fine-tuning a pre-trained model.

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
Semantic segmentation can be considered as a pixel-wise classification problem where we need to predict the class label for each pixel. Fine-tuning is often used when you only have limited labeled data.

Here, we take a pre-trained model on the Microsoft COCO [Lin et al., 2014] dataset and fine- tune it to perform segmentation with the classes it was never trained on. To be more specific, we use deeplabv3 [Chen et al., 2017]5 pre-trained model and fine-tune it on the Oxford17 [Nilsback and Zisserman, 2008] flower dataset.

We simplify the task to be a binary semantic segmentation task (background and flower). In the following code, you will first see some examples from the Oxford17 dataset and load the finetune the model by truncating the last layer of the network and replacing it with a randomly initialized convolutional layer. Note that we only update the weights of the newly introduced layer.

<ol>
<li>For this assignment, we want to fine-tune only the last layer in our downloaded deeplabv3. We do this by keeping track of weights we want to update in learned parameters.
Use the PyTorch utility Model.named parameters()6, which returns an iterator over all the weight matrices of the model.

The last layer weights have names prefix “classifier.4”. We will select the corresponding weights then passing them to learned parameters.

Complete the ‘train‘ function in Part C of the notebook by adding 2-3 lines of code where indicated.
</li>
<li>For fine-tuning we also want to
<ul>
<li>use Model.requires grad () to prevent back-prop through all the layers that should be frozen.</li>
<li>replace the last layer with a new nn.Conv2d layer with appropriate input output channels and kernel sizes. Since we are performing binary segmentation for this assignment, this new layer should have 2 output channels.

Complete the script in Question 2 of Part C by adding around 2 lines of code and train the model.</li>
</ul>
</li>
<li>Visualize the predictions by running the helper code provided.</li>
<li>Consider a case of fine-tuning a pre-trained model with n number of layers. Each of the layers have a similar number of parameters, so the total number of parameters for the model is proportional to n. Describe the difference in memory complexity in terms of n between fine-tuning an entire pre-trained model versus fine-tuning only the last layer (freezing all the other layers). What about the computational complexity?</li>
</ol>
5deeplabv3 details: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/ 6See examples at https://pytorch.org/docs/stable/nn.html

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
&nbsp;

5. If we increase the height and the width of the input image by a factor of 2, how does this affect the memory complexity of fine-tuning? What about the number of parameters?

What you have to submit

For reference, here is everything you need to hand in. See the top of this handout for submission directions.

• A PDF file titled a2-writeup.pdf containing the following:

– Answers to questions from Part A – Answers to questions from Part B – Answers to questions from Part C

• Your code file a2-cnn.ipynb References

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedi- cal image segmentation. In International Conference on Medical image computing and computer- assisted intervention, pages 234–241. Springer, 2015.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll ́ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017.

Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian Conference on Computer Vision, Graphics &amp; Image Processing, pages 722–729. IEEE, 2008.

</div>
</div>
<div class="layoutArea">
<div class="column">
5

</div>
</div>
</div>
