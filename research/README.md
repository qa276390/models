# For Fashion Object Detection
This project depends on <a href='object_detection'>Tensorflow Object Detection API</a><br>
## Installation
To start installation please see <a href='object_detection/g3doc/installation.md'>Installation</a><br> and also see my <a href='object_detection/g3doc/Installation_Note.md'>Installation Note</a><br>

## Folder and Dataset
    .				  		# tf-models/research/
    ├── ...
    ├── Yahoo_Data                    		
    │   ├── my_exported_graph-411163		# exported model
    │   ├── clean_gd_list.txt			# the gd list 
    |   ├── clothes.data               	        # another gd list
    │   └── gid2class.csv     		        # the mapping from gid to the class(category)
    ├── ModalNetDetect
    │   ├── data   
    │   │   ├── modalnet_label_map.pbtxt		# the mapping from index to label
    │   └── models                		              
    ├── predict.py
    ├── Predict.sh
    ├── Predict_All.sh
    ├── 
    └── ...
please copy Yahoo_Data from `/eds/research/2019_intern_data/Yahoo_Data` on rs1.

## Prediction 

### Pair Detection
We detect the object from the images, but only save the one which contains more than 1 object.
```bash
sh Predict.sh 
```
in Predict.sh,
```bash
#gid list with format "GID\tURL\tDESCRIPTION\tCLASS_1\tCLASS_2\tCLASS_3\tCLASS_4\t"
GD_LIST="Yahoo_Data/clean_gd_list.txt"
OUT_DIR="ex_output"
MODEL_PATH="Yahoo_Data/my_exported_graphs-411163/frozen_inference_graph.pb" 
PBTXT_PATH="./ModalNetDetect/data/modalnet_label_map.pbtxt"
IMG_DIR="/eds/research/bhsin/yahoo_clothes/img/"

time PYTHONIOENCODING=utf-8 python3 predict.py --info-path=$GD_LIST \
	--output-path=$OUT_DIR \
	--model-path=$MODEL_PATH \
	--pbtxt-path=$PBTXT_PATH \
	--img-dir=$IMG_DIR
```
### Multiple Objects
We save all the objects.
```bash
sh Predict_All.sh
```
in Predict_All.sh,
```bash
time PYTHONIOENCODING=utf-8 python3 predict.py --prob-thr 0.6 --no-cut \
	--info-path ./Yahoo_Data/clothes.data \
	--output-path crop_all_tbdetect \
	--img-dir /eds/research/bhsin/yahoo_clothes/clothes_full_image/ \
	--crop-all \
	--gid2class \
	--gid2class-path ./Yahoo_Data/gid2class.csv \

```



# TensorFlow Research Models

This folder contains machine learning models implemented by researchers in
[TensorFlow](https://tensorflow.org). The models are maintained by their
respective authors. To propose a model for inclusion, please submit a pull
request.

## Models

-   [adversarial_crypto](adversarial_crypto): protecting communications with
    adversarial neural cryptography.
-   [adversarial_text](adversarial_text): semi-supervised sequence learning with
    adversarial training.
-   [attention_ocr](attention_ocr): a model for real-world image text
    extraction.
-   [audioset](audioset): Models and supporting code for use with
    [AudioSet](http://g.co/audioset).
-   [autoencoder](autoencoder): various autoencoders.
-   [brain_coder](brain_coder): Program synthesis with reinforcement learning.
-   [cognitive_mapping_and_planning](cognitive_mapping_and_planning):
    implementation of a spatial memory based mapping and planning architecture
    for visual navigation.
-   [compression](compression): compressing and decompressing images using a
    pre-trained Residual GRU network.
-   [cvt_text](cvt_text): semi-supervised sequence learning with cross-view
    training.
-   [deep_contextual_bandits](deep_contextual_bandits): code for a variety of contextual bandits algorithms using deep neural networks and Thompson sampling.
-   [deep_speech](deep_speech): automatic speech recognition.
-   [deeplab](deeplab): deep labeling for semantic image segmentation.
-   [delf](delf): deep local features for image matching and retrieval.
-   [differential_privacy](differential_privacy): differential privacy for training
    data.
-   [domain_adaptation](domain_adaptation): domain separation networks.
-   [fivo](fivo): filtering variational objectives for training generative
    sequence models.
-   [gan](gan): generative adversarial networks.
-   [im2txt](im2txt): image-to-text neural network for image captioning.
-   [inception](inception): deep convolutional networks for computer vision.
-   [keypointnet](keypointnet): discovery of latent 3D keypoints via end-to-end
    geometric eeasoning [[demo](https://keypointnet.github.io/)].
-   [learning_to_remember_rare_events](learning_to_remember_rare_events): a
    large-scale life-long memory module for use in deep learning.
-   [learning_unsupervised_learning](learning_unsupervised_learning): a
    meta-learned unsupervised learning update rule.
-   [lexnet_nc](lexnet_nc): a distributed model for noun compound relationship
    classification.
-   [lfads](lfads): sequential variational autoencoder for analyzing
    neuroscience data.
-   [lm_1b](lm_1b): language modeling on the one billion word benchmark.
-   [lm_commonsense](lm_commonsense): commonsense reasoning using language models.
-   [maskgan](maskgan): text generation with GANs.
-   [namignizer](namignizer): recognize and generate names.
-   [neural_gpu](neural_gpu): highly parallel neural computer.
-   [neural_programmer](neural_programmer): neural network augmented with logic
    and mathematic operations.
-   [next_frame_prediction](next_frame_prediction): probabilistic future frame
    synthesis via cross convolutional networks.
-   [object_detection](object_detection): localizing and identifying multiple
    objects in a single image.
-   [pcl_rl](pcl_rl): code for several reinforcement learning algorithms,
    including Path Consistency Learning.
-   [ptn](ptn): perspective transformer nets for 3D object reconstruction.
-   [marco](marco): automating the evaluation of crystallization experiments.
-   [qa_kg](qa_kg): module networks for question answering on knowledge graphs.
-   [real_nvp](real_nvp): density estimation using real-valued non-volume
    preserving (real NVP) transformations.
-   [rebar](rebar): low-variance, unbiased gradient estimates for discrete
    latent variable models.
-   [resnet](resnet): deep and wide residual networks.
-   [seq2species](seq2species): deep learning solution for read-level taxonomic
    classification.
-   [skip_thoughts](skip_thoughts): recurrent neural network sentence-to-vector
    encoder.
-   [slim](slim): image classification models in TF-Slim.
-   [street](street): identify the name of a street (in France) from an image
    using a Deep RNN.
-   [struct2depth](struct2depth): unsupervised learning of depth and ego-motion.
-   [swivel](swivel): the Swivel algorithm for generating word embeddings.
-   [syntaxnet](syntaxnet): neural models of natural language syntax.
-   [tcn](tcn): Self-supervised representation learning from multi-view video.
-   [textsum](textsum): sequence-to-sequence with attention model for text
    summarization.
-   [transformer](transformer): spatial transformer network, which allows the
    spatial manipulation of data within the network.
-   [vid2depth](vid2depth): learning depth and ego-motion unsupervised from
    raw monocular video.
-   [video_prediction](video_prediction): predicting future video frames with
    neural advection.
