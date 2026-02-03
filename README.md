### RAASID – AI-Powered Crime Analysis Platform

RAASID is an AI-powered platform designed to enhance community safety by analyzing crime-related visual content and Arabic textual data.
The system is built around three core AI modules: Crime Detection, Arabic Text Summarization, and Fake News Detection.

### Core 1: Crime Detection from Images & Videos

This core focuses on detecting and classifying crime types from image and video data using CNN-based models and transformer-based video architectures.

### Dataset

UCF-Crime Dataset (Kaggle subset)

Classes: Fighting, Robbery, Shooting, Stealing, Normal

Balanced dataset (max 7,140 images per class)

Multiple input resolutions tested (64×64 to 200×200)

## Models & Experiments

Custom CNN models tested with different input sizes and dropout rates

# Best CNN accuracy: ~98%

MobileNetV2 and MobileNetV3Small for lightweight classification

### MobileNetV3Small (partial fine-tuning) achieved 98.19% test accuracy

### ResNet50 showed overfitting due to dataset limitations

### Final Model: VideoMAE Transformer

Input: 224×224, 16 frames per clip

Accuracy: 92.96%

Evaluation Loss: 0.15

### ✔ Best generalization achieved using partial fine-tuning and transformer-based video modeling.

## Core 2: Arabic Text Summarization (NLP)

This core performs abstractive summarization of Arabic news articles using deep learning models.

### Dataset

XLSum Arabic Dataset

## 30,000 training samples, 4,700 testing samples

Input articles truncated to 423 tokens

Summaries limited to 25 tokens

### Models & Experiments

AraT5-base
mT5-base-ar
Custom CNN-based summarization model
AraGPT2

### Results

AraT5-base achieved the best overall performance

Optimal at 10 epochs without lemmatization

Lemmatization slightly reduced ROUGE-2 scores

Custom CNN achieved very high ROUGE scores but showed signs of overfitting

AraGPT2 performed poorly due to lack of summarization pretraining

## ✔ AraT5-base provided the best balance between fluency and content coverage.

### Core 3: Fake News Detection (Arabic NLP)

This core detects real vs. fake Arabic news articles using Machine Learning and Deep Learning models.

### Dataset

## David Ozil Arabic News Dataset (46K articles)

## Mina Alhashimi Dataset (2K balanced articles)

Data imbalance handled using Back Translation data augmentation

### Preprocessing

Arabic text normalization and cleaning
Stopword and punctuation removal
TF-IDF vectorization (ML models)
Tokenization and padding (CNN)

### Models & Results

Machine Learning (TF-IDF):
Logistic Regression: 96%
Random Forest: 96.6%
XGBoost: 96.3%
Linear SVC (Best): 97.1%

Deep Learning (CNN):
Test Accuracy: 92%
Macro F1-Score: 92%
Fake news recall: 99%

## ✔ ML models achieved higher accuracy, while CNN showed strong generalization and robustness.

### Technologies Used

Deep Learning: CNNs, Transformers (VideoMAE)
NLP: Arabic text processing, summarization, fake news detection
Computer Vision: Image & video analysis
Frameworks: PyTorch, TensorFlow
Training Environment: Google Colab (GPU)
