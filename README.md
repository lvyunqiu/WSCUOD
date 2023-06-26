# WSCUOD
The code for "Weakly-supervised Contrastive Learning for Unsupervised Object Discovery"

The checkpoint for evaluation could be downloaded from [wcl-16-final.pth](https://drive.google.com/file/d/1eh8Y7yLTngjEu5EGPJ0RI9blEQ_5qyvS/view?usp=sharing)

The results for object segmentation in VOC2007, VOC2012, COCO20K, DUTS-Test, DUT-OMRON and ECSSD could be downloaded from [seg_results](https://drive.google.com/file/d/17NXXunDKbIkbJ800yVE4LTo-xYNIGI_I/view?usp=drive_link)

## 1. Dependencies
This code was implemented with Python 3.8, PyTorch 1.9.1+cu111 and CUDA 11.6. 

## 2. Data

The dataset for segmentation could be downloaded from [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [DUTS](http://saliencydetection.net/duts/) and [DUT-OMRON](http://saliencydetection.net/dut-omron/);
The dataset for detection could be downloaded from [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/), [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) and [COCO20K](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

## 3. Training
To train the mode, please check the dataset directory in data/dataloader.py (`image_folder_prefix`) and the pre-trained model directory in wcl.py (`--pretrained_path`) and run:

```python
bash run_pipeline.sh
```

## 4. Testing and Evaluation
Please check the testing dataset directory in ddt_dino_sig.py (`--test_root` and `test_dataset`), and the checkpoint directory in ddt_dino_sig.py (`--pretr_path`) and run:

```python
python ddt_dino_sig.py
python measure.py
```

## 5. Acknowledgement
We acknowledge these excellent works that inspire our project: [Weakly Contrastive Learning](https://github.com/mingkai-zheng/WCL), [DDT](https://github.com/GeoffreyChen777/DDT.git) and [DINO](https://github.com/facebookresearch/dino). 
