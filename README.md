# 🧠 Next Word Prediction with LSTM & GRU

A deep learning system for next word prediction using **LSTM and GRU architectures**, trained on Shakespeare's *Hamlet*. Features model training, early stopping, and Streamlit deployment.

---

## 🚀 Features
- **Dual-model architecture**: LSTM and GRU implementations
- **Early stopping** with patience=3 and best weight restoration
- **Streamlit web app** for real-time predictions
- **Tokenizer serialization** for consistent preprocessing
- **Model checkpointing** (.h5 files)

---

## 📂 Repository Structure
| File | Purpose |
|------|---------|
| `app.py` | Streamlit prediction interface |
| `experiments.ipynb` | Model training & evaluation |
| `hamlet.txt` | Shakespeare training corpus |
| `next_word_lstm.h5` | Trained LSTM model |
|`next_word_lstm_model_with_early_stopping.h5` | LSTM model with early stopping |
| `tokenizer.pickle` | Serialized text processor |
| `requirements.txt` | Python dependencies |

---

## ⚙️ Technical Implementation

### Model Architectures

LSTM Model
model = Sequential([
Embedding(total_words, 100, input_length=max_sequence_len-1),
LSTM(150, return_sequences=True),
Dropout(0.2),
LSTM(100),
Dense(total_words, activation='softmax')
])

GRU Model
model = Sequential([
Embedding(total_words, 100, input_length=max_sequence_len-1),
GRU(150, return_sequences=True),
Dropout(0.2),
GRU(100),
Dense(total_words, activation='softmax')
])


### Training Details

- **Optimizer**: Adam
- **Loss**: Categorical crossentropy
- **Early Stopping**: Monitor val_loss, patience=3
- **Final Accuracy**: 
  - Training: ~9.47% (Epoch 6)
  - Validation: ~7.48% (Epoch 6)

### Prediction Logic

def predict_next_word(model, tokenizer, text, max_sequence_len):
token_list = tokenizer.texts_to_sequences([text])
token_list = token_list[-(max_sequence_len-1):]
token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
predicted = model.predict(token_list, verbose=0)
return tokenizer.index_word[np.argmax(predicted)]


---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow 2.x

### Installation

pip install -r requirements.txt


### **Usage**

1. **Train or use the model:**  
   The trained LSTM model and tokenizer are provided.

2. **Run the web app:**
   streamlit run app.py


3. **Try interactive prediction:**  
Enter a sequence of words and get the predicted next word.

Streamlit UI
st.title("Next Word Prediction With LSTM")
input_text = st.text_input("Enter text sequence", "To be or not to")
if st.button("Predict"):
next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
st.success(f"Next word: {next_word}")

---

### **How It Works**

- **Data Preprocessing:**  
The text corpus is cleaned, tokenized, and converted into input sequences for training.

- **Model Training:**  
An LSTM model is trained to predict the next word in a sequence, with early stopping to prevent overfitting.

- **Deployment:**  
The Streamlit app provides a user-friendly interface for real-time next word prediction.

---

### **Example**
Input: To be or not

Output: to

---

## 📊 Training Results
| Epoch | Loss | Accuracy | Val Loss | Val Acc |
|-------|------|----------|----------|---------|
| 1 | 7.08 | 0.0299 | 6.94 | 0.0352 |
| 2 | 6.55 | 0.0361 | 6.87 | 0.0493 |
| 3 | 6.32 | 0.0495 | 6.79 | 0.0554 |
| 4 | 6.07 | 0.0644 | 6.81 | 0.0731 |
| 5 | 5.79 | 0.0859 | 6.81 | 0.0738 |
| 6 | 5.51 | 0.0947 | 6.91 | 0.0748 |

---



