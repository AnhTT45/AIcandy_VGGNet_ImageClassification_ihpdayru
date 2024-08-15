# VGGNet and Image Classification

<p align="justify">
<strong>VGGNet</strong> is a deep convolutional neural network (CNN) architecture introduced by the Visual Geometry Group (VGG) at the University of Oxford in 2014. It is renowned for its simplicity and uniformity, using very small 3x3 convolutional filters and a consistent architecture of stacking layers. VGGNet is known for its deep structure, with variants such as VGG16 and VGG19, which contain 16 and 19 layers, respectively. The model achieved high performance in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) and is widely used as a feature extractor in various computer vision tasks due to its ability to capture detailed features and patterns.
</p>

## Image Classification
<p align="justify">
Image classification is a fundamental problem in computer vision where the goal is to assign a label or category to an image based on its content. This task is critical for a variety of applications, including medical imaging, autonomous vehicles, content-based image retrieval, and social media tagging.
</p>


## ❤️❤️❤️


```bash
If you find this project useful, please give it a star to show your support and help others discover it!
```

## Getting Started

### Clone the Repository

To get started with this project, clone the repository using the following command:

```bash
git clone https://github.com/TruongNV-hut/AIcandy_VGGNet_ImageClassification_ihpdayru.git
```

### Install Dependencies
Before running the scripts, you need to install the required libraries. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the model, use the following command:

```bash
python aicandy_vggnet_train_rabvochc.py --train_dir ../dataset --num_epochs 10 --batch_size 32 --model_path aicandy_model_out_gxbruthb/aicandy_model_pth_aootldae.pth
```

### Testing the Model

After training, you can test the model using:

```bash
python aicandy_vggnet_test_xxkivhyd.py --image_path ../image_test.jpg --model_path aicandy_model_out_gxbruthb/aicandy_model_pth_aootldae.pth --label_path label.txt
```

### Converting to ONNX Format

To convert the model to ONNX format, run:

```bash
python aicandy_vggnet_convert_onnx_rkpcsdxp.py --model_path aicandy_model_out_gxbruthb/aicandy_model_pth_aootldae.pth --onnx_path aicandy_model_out_gxbruthb/aicandy_model_onnx_atqoumxp.onnx --num_classes 2
```

### More Information

For a detailed overview of AlexNet and image classification, visit [aicandy.vn](https://aicandy.vn/).

❤️❤️❤️




