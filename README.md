# Machine Learning Deep Learning project
> A collection of hands-on machine learning and deep learning projects, including image classification, text classification, and summarization with modern frameworks (TensorFlow, PyTorch, HuggingFace)

**Live Demo:** [View Project on GitHub Pages](https://jean-granger.github.io/Machine-Learning---Deep-Learning-project/)

## Overview
This repository contains three end-to-end machine learning and deep learning projects designed to illustrate practical AI workflows from dataset preparation to model evaluation and deployment readiness.

Each notebook demonstrates a different aspect of data-centric AI and model engineering:
- Image Classification (CNN): Trains a Convolutional Neural Network (CNN) to classify images (CIFAR-10 or custom dataset) with TensorFlow / Keras.
- Text Classification (Sentiment Analysis): Fine-tunes DistilBERT on the IMDB dataset for binary sentiment classification with Hugging Face Transformers, PyTorch.
- Text Summarization (BART Model): Generates concise summaries from long passages using facebook/bart-large-cnn and Hugging Face Transformers.

## Example Results
The executed versions of each notebook with visible outputs are available at:
- [Image Classification (CNN)](https://jean-granger.github.io/Machine-Learning---Deep-Learning-project/exports/image_classification_cnn.html)
- **Image Classification (CNN)** → [`exports/image_classification_cnn.html`](exports/image_classification_cnn.html)
- [Text Classification (DistilBERT - Sentiment Analysis)](https://jean-granger.github.io/Machine-Learning---Deep-Learning-project/exports/text_classification_bert.html)
- [Text Summarization (BART)](https://jean-granger.github.io/Machine-Learning---Deep-Learning-project/exports/text_summarization_transformer.html)

These outputs include model summaries, training progress, evaluation metrics, and visualizations.

## Requirements
To run any notebook locally:
```bash
pip install tensorflow torch transformers datasets matplotlib scikit-learn
```
## Notes
- All notebooks are designed to run independently.
- The results links are recommended for quick viewing without re-running models.

## Future Work

- Extend the text summarization task with domain-specific datasets (e.g., news or legal text).
- Add experiment tracking with MLflow or Weights & Biases.
- Explore lightweight model deployment using FastAPI or Streamlit.

## Acknowledgement 

This work leverages open-source AI frameworks and datasets, including:

- [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) for deep learning frameworks.
- [Hugging Face Transformers](https://huggingface.co/transformers/) for pre-trained models and NLP tools.
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) datasets for training and evaluation.

## Author
Jean Granger - 
[LinkedIn](https://linkedin.com/in/ange-granger-jean-365b94320) — jeannange001@gmail.com - aeagsjean@st.knust.edu.gh
