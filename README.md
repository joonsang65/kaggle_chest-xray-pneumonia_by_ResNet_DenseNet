## kaggle_chest-xray-pneumonia

---

## Abstract
>This project aims to solve the pneumonia classification problem using the chest-xray-pneumonia image dataset from Kaggle. The experiment compares the ResNet and DenseNet models, and the dataset is divided into GrayScale and RGB versions for testing. Additionally, a crop technique is applied to remove text from the images, allowing the models to focus on the key features. Class Activation Mapping (CAM) is used to visualize the weights assigned by the models to different regions of the images.

#### The part that needs to be understood before running the code.

---

## WHat Is ResNet & DenseNet ?
![image](https://github.com/user-attachments/assets/29344944-71aa-4722-b748-ff4b2ff84968)
**ResNet** 
>uses Residual Connections to ensure stable learning even in deep networks. This structure helps mitigate the vanishing gradient problem and provides strong performance with relatively fewer parameters. In this experiment, we used the ResNet34 model.

**DenseNet** 
>connects each layer to every previous layer, maximizing feature reuse and minimizing performance degradation. While it requires more computational resources, DenseNet performs exceptionally well even on small datasets. In this experiment, we used the DenseNet121 model.

---

Key Differences Between ResNet and DenseNet
1. **Structural Difference**: ResNet uses residual connections, whereas DenseNet connects every layer to all previous layers.
2. **Parameter Count**: ResNet performs well with relatively fewer parameters, while DenseNet requires significantly more parameters and computational resources.
3. **Learning Efficiency**: ResNet excels with large datasets, while DenseNet is particularly strong with smaller datasets.

---

### 1. Installation
Clone the repository:
```
git clone https://github.com/yourusername/your_repo.git
cd your_repo
```

### 2.Install dependencies: To install the required libraries, run the following command:
```
pip install -r requirements.txt
```

### 3. Running the Project
1. Run main.py: When you run main.py, the terminal will guide you to set up the model and dataset to be used for training.

2. Example setup process: After running the script, you'll be prompted to answer the following questions:
```
Crop model has only RGB images.
Do you want to crop the images? (y/n): n
Do you want to fine-tune the model? (Full/partial): Full
Which model do you want to use? (ResNet/DenseNet): ResNet
Do you want to convert the images to grayscale? (y/n): y
```

3. Confirm your choices: The script will display your selections for confirmation. Here's an example of what it looks like:
```
You selected
Model: ResNet
Tuning: Full
Crop: n
Grayscale: y
```

4. Confirm or modify your choices: After reviewing the options, you can either confirm by pressing 'y' or modify your choices by pressing 'n'. If you choose 'n', the script will prompt you to re-enter the information.
```
Is it right? (y/n): n
===================================================
Please re-enter the information
```
The process will continue until you confirm with 'y'.

---

## Feature List
### 1. Dataset and Model Configuration

- Image Cropping Option: Choose whether the images used for model training should be cropped.
- Grayscale Conversion Option: Choose whether the images should be converted to grayscale.
- Model Selection: Select either the ResNet or DenseNet model for training.
- Model Tuning Option: Choose whether to train the entire model (Full) or only part of it (Partial).

### 2. Pneumonia Image Classification Training
Process the data and train the model according to the selected configuration.
Split the data into training and validation sets and proceed with the training.
Validation Performance Evaluation

### 3. Confusion Matrix: 
Visualize the model's prediction results to see the relationship between the actual and predicted classes.
Precision, Recall, F1-Score: Calculate Precision, Recall, and F1-Score to evaluate the performance of the model.


### 4. Weight Visualization (CAM)
Class Activation Mapping (CAM): Visualize the regions the model is focusing on in the images, highlighting the learned weights and important areas.

---

You can view the example results at the following link:  
https://www.kaggle.com/code/jsjs0125/pneumonia-chest-x-ray-images-by-pytorch
