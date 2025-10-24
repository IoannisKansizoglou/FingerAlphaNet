# 📂 Data Folder

This folder is intended to store the **Sign Language MNIST** dataset used in this project.  

---

## 📊 Dataset

The project uses the **Sign Language MNIST** dataset from Kaggle:  

[Sign Language MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)  

This dataset contains **25 classes** corresponding to the letters A-Y (excluding J, which requires motion).

---

## 🗂️ Folder Structure

After downloading, place the CSV files in this folder:

```text
data/
├── sign_mnist_train.csv
└── sign_mnist_test.csv

* sign_mnist_train.csv — Training data
* sign_mnist_test.csv — Test/validation data

## ⚡ Usage

Once the CSV files are in the data/ folder, the training and evaluation scripts will automatically load them using the paths:

```python
train_csv = "data/sign_mnist_train.csv"
test_csv = "data/sign_mnist_test.csv"
```

No additional preprocessing is required; the scripts will handle image reshaping and normalization.
