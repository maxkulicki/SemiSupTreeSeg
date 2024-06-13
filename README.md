# Semi-supervised tree segmentation using a discriminator model

**DISCLAIMER: The discriminator models in their current form don't really work, they are not able to reliably distinguish bad and good predictions. But the pipeline is finished, so if you figure out how to train a working discriminator, you can use this code to run semi-supervised training**.

Based on [TreeLearn](https://github.com/ecker-lab/TreeLearn), [PointGL](https://github.com/Roywangj/PointGL/) and [TLSpecies](https://github.com/mataln/TLSpecies).

This repo builds upon the TreeLearn architecture for individual tree segmentation in a way that enables training on unlabelled data.
The model can be trained on its own predictions by incorporating an additional discriminator model, which identifies badly segmented samples and makes sure that these are not used for training. 
The discriminator model training is based on distinguishing real, ground-truth segmented trees from predicted trees.

The scripts in the **semisup_pipeline** folder enable generating a dataset for training the discriminator model, training it, using it to generate a semi-supervised dataset and then training the segmentation model.
The pipeline is based on an image-based classification model from [TLSpecies](https://github.com/mataln/TLSpecies). The repo also includes two other classification approaches: various point cloud deep learning architectures implemented in the [OpenPoints](https://github.com/maxkulicki/SemiSupTreeSeg/tree/main/PointGL/openpoints) library and a Random Forest model based on various handcrafted features of the point cloud. I was not able to reliably train any of the models. 
