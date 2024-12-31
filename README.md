
# Sentiment Analysis with Sequential Deep Learning Architectures 🚀
A comprehensive implementation and comparison of various deep learning architectures for sentiment analysis, focusing on sequential models and their performance on the IMDB movie reviews dataset.

## 🎯 Project Overview

This project explores and compares different neural network architectures for sentiment analysis, implementing:
- Basic sequential models (RNN, LSTM, GRU)
- Advanced encoder-decoder architectures
- A custom transformer implementation
- Performance analysis and comparison metrics

## 📊 Dataset Details

### IMDB Movie Reviews Dataset
- **Total Size**: 50,000 reviews
- **Class Distribution**: 
  - 25,000 positive reviews (50%)
  - 25,000 negative reviews (50%)
- **Text Characteristics**:
  - Variable length reviews
  - Raw text with HTML tags
  - Mixed case, punctuation, and special characters
- **Labels**: Binary (Positive/Negative)

- You can get the dataset from the website https://ai.stanford.edu/~amaas/data/sentiment/

  
### Data Preprocessing Pipeline

1. **Text Cleaning**
   ```python
   # Load the dataset
   data = pd.read_csv('/path/to/IMDB Dataset.csv')
   ```

2. **Tokenization**
   ```python
   tokenizer = Tokenizer(num_words=10000)
   tokenizer.fit_on_texts(reviews)
   sequences = tokenizer.texts_to_sequences(reviews)
   ```

3. **Sequence Padding**
   ```python
   X = pad_sequences(sequences, maxlen=250)
   ```

4. **Label Encoding**
   ```python
   labels = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values
   ```

5. **Train-Test Split**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, labels, test_size=0.25, random_state=42
   )
   ```

## 🛠️ Model Architectures

### 1. Simple RNN Implementation
```python
rnn_model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_len),
    SimpleRNN(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])
```
- **Embedding Layer**: Converts token indices to dense vectors
- **RNN Layer**: Processes sequential data with tanh activation
- **Dense Layer**: Binary classification output

### 2. LSTM Architecture
```python
lstm_model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=False),
    Dense(1, activation='sigmoid')
])
```
- **Advanced Features**:
  - Forget gate: Controls information flow
  - Input gate: Regulates new information
  - Output gate: Manages cell state impact
  - Cell state: Long-term memory storage

### 3. GRU Implementation
```python
gru_model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_len),
    GRU(128, return_sequences=False),
    Dense(1, activation='sigmoid')
])
```
- **Key Components**:
  - Reset gate: Controls past state influence
  - Update gate: Manages information flow
  - Simplified architecture compared to LSTM

### 4. Encoder-Decoder Architectures

#### RNN Encoder-Decoder
```python
enc_dec_model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_len),
    SimpleRNN(128, return_sequences=False),
    RepeatVector(max_len),
    SimpleRNN(128, return_sequences=True),
    TimeDistributed(Dense(1, activation='sigmoid'))
])
```

#### LSTM Encoder-Decoder
```python
enc_dec_model_lstm = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=False),
    RepeatVector(max_len),
    LSTM(128, return_sequences=True),
    TimeDistributed(Dense(1, activation='sigmoid'))
])
```

#### GRU Encoder-Decoder
```python
enc_dec_model_gru = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_len),
    GRU(128, return_sequences=False),
    RepeatVector(max_len),
    GRU(128, return_sequences=True),
    TimeDistributed(Dense(1, activation='sigmoid'))
])
```

### 5. Transformer Architecture
```python
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
```

## 📈 Detailed Results and Analysis

- Accuracy: The proportion of correct predictions.
- Precision: The proportion of true positive predictions over all positive predictions.
- Recall: The proportion of true positive predictions over all actual positives.
- F1 Score: The harmonic mean of precision and recall.
- AUC-ROC: The area under the receiver operating characteristic curve.

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC | Training Time* |
|-------|----------|-----------|---------|-----------|---------|---------------|
| RNN | 0.83 | 0.84 | 0.83 | 0.83 | 0.91 | Fast |
| LSTM | 0.89 | 0.89 | 0.90 | 0.90 | 0.96 | Medium |
| GRU | 0.90 | 0.87 | 0.90 | 0.90 | 0.96 | Medium |
| Enc-Dec (RNN) | 0.56 | 0.64 | 0.28 | 0.39 | 0.56 | Slow |
| Enc-Dec (LSTM) | 0.87 | 0.87 | 0.88 | 0.87 | 0.87 | Slow |
| Enc-Dec (GRU) | 0.88 | 0.87 | 0.91 | 0.89 | 0.88 | Slow |
| Transformer | 0.49 | 0.00 | 0.00 | 0.00 | 0.50 | Medium |

*Training time is relative and depends on hardware configuration

### Detailed Analysis

1. **Simple RNN Performance**
   - Good baseline performance (0.83 accuracy)
   - Fast training and inference
   - Limited ability to capture long-term dependencies
   - Suitable for quick prototyping

2. **LSTM Advantages**
   - Excellent performance (0.89 accuracy)
   - Better handling of long-term dependencies
   - More stable training compared to RNN
   - Higher computational cost than RNN

3. **GRU Benefits**
   - Best overall performance (0.90 accuracy)
   - Faster training than LSTM
   - Similar capability to LSTM with fewer parameters
   - Better generalization on this dataset

4. **Encoder-Decoder Results**
   - Mixed performance across variants
   - RNN variant struggled significantly
   - LSTM and GRU variants showed good performance
   - Higher computational overhead

5. **Transformer Challenges**
   - Transformers struggled with performance on the IMDB dataset due to the relatively smaller size compared to what Transformers are typically used for
   - Poor performance in basic implementation
   - Requires more sophisticated architecture
   - Potential for improvement with modifications
   - May need larger dataset or better preprocessing


### Requirements.txt
```text
tensorflow>=2.0.0
numpy>=1.19.2
pandas>=1.1.3
scikit-learn>=0.23.2
matplotlib>=3.3.2
seaborn>=0.11.0
```

## 📊 Training and Evaluation

### Training Process
```python
# Common training configuration
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Model training
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)
```

### Evaluation Code
```python
def evaluate_model(model, X_test, y_test):
    # Get predictions
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': roc_auc
    }
```

## 🚀 Future Improvements

### Model Enhancements
1. **Architecture Improvements**
   - Add attention mechanisms to RNN variants
   - Implement bidirectional layers
   - Try hybrid architectures

2. **Training Optimizations**
   - Implement learning rate scheduling
   - Try different optimizers
   - Add gradient clipping

3. **Data Processing**
   - Implement more sophisticated text preprocessing
   - Try different embedding techniques
   - Experiment with data augmentation

### Additional Features
1. **Model Analysis**
   - Add visualization of attention weights
   - Implement interpretation techniques
   - Add confusion matrix analysis

2. **Performance Optimization**
   - Add model quantization
   - Implement batch prediction
   - Optimize memory usage

## 🤝 Contributing

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📚 References

1. Understanding LSTM Networks - Colah's Blog
2. Attention Is All You Need - Vaswani et al.
3. Deep Learning for Sentiment Analysis - Stanford NLP
