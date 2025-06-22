# lstm-gru-nextword
A deep learning project for next word prediction using LSTM and GRU models, trained on Shakespeare’s Hamlet.

### Overview
This project demonstrates how to build and deploy a next word prediction system using LSTM (Long Short-Term Memory) neural networks. The model learns to predict the most probable next word in a sequence, which is useful for applications like text auto-completion and conversational AI.

### Features
LSTM-based next word prediction

Trained on the text of Shakespeare’s Hamlet

End-to-end ETL pipeline: data cleaning, tokenization, and feature extraction

Model checkpointing and early stopping for robust training

Interactive web app deployment using Streamlit

### Repository Structure
app.py – Streamlit web app for interactive prediction

experiments.ipynb – Jupyter notebook for model development and experiments

hamlet.txt – Training corpus (Shakespeare’s Hamlet)

next_word_lstm.h5 – Saved LSTM model

next_word_lstm_model_with_early_stopping.h5 – LSTM model with early stopping

requirements.txt – Python dependencies

tokenizer.pickle – Saved tokenizer for text preprocessing

README.md – Project documentation

## **Getting Started**

### **Prerequisites**
- Python 3.7+

### **Install dependencies**
pip install -r requirements.txt


### **Usage**

1. **Train or use the model:**  
   The trained LSTM model and tokenizer are provided.

2. **Run the web app:**
   streamlit run app.py


3. **Try interactive prediction:**  
Enter a sequence of words and get the predicted next word.

---

### **How It Works**

- **Data Preprocessing:**  
The text corpus is cleaned, tokenized, and converted into input sequences for training.

- **Model Training:**  
An LSTM model is trained to predict the next word in a sequence, with early stopping to prevent overfitting.

- **Deployment:**  
The Streamlit app provides a user-friendly interface for real-time next word prediction.

---

## **Example**
Input: To be or not
Output: to




