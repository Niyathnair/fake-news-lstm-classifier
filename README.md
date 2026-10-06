# Fake News Classifier (LSTM)

A notebook that trains an LSTM binary classifier on article titles from the Kaggle Fake News competition. Titles are cleaned and stemmed with NLTK, converted to integer sequences with Keras `one_hot`, and fed to an Embedding + LSTM network. Two variants are trained: one without dropout and one with it.

## Approach

1. Load `train.csv` (columns `id`, `title`, `author`, `text`, `label`) and drop rows with missing values, leaving 18,285 rows. `label` is the target.
2. Use only the `title` column as input text.
3. Clean each title: replace non-letter characters with spaces, lowercase, split into words, remove NLTK English stopwords, and apply `PorterStemmer`.
4. Encode each cleaned title with `one_hot` using a vocabulary size of 5,000, then pre-pad to a fixed length of 20 with `pad_sequences`.
5. Split 67/33 with `train_test_split(test_size=0.33, random_state=42)`.
6. Model 1: `Embedding(5000, 40, input_length=20)` -> `LSTM(100)` -> `Dense(1, activation='sigmoid')` (256,501 parameters), compiled with binary cross-entropy, Adam, and accuracy.
7. Train for 15 epochs with batch size 64, using the 33% split as validation data.
8. Evaluate on the same 33% split with `confusion_matrix` and `accuracy_score`.
9. Model 2: the same architecture with `Dropout(0.3)` after the embedding and after the LSTM, trained and evaluated the same way.

## Results

From the saved notebook outputs, on the 33% hold-out split (6,035 titles). This split is also the validation set passed to `model.fit`.

| Model | Accuracy | Confusion matrix (rows = true 0, 1; columns = predicted 0, 1) |
|---|---|---|
| LSTM, no dropout | 0.9135045567522784 | [[3119, 300], [222, 2394]] |
| LSTM with Dropout(0.3) | 0.9035625517812759 | [[3159, 260], [322, 2294]] |

Caveat for the first row: the saved training log for the no-dropout model begins at a training accuracy of 0.9981 in epoch 1, which indicates the fit cell was re-run on weights that had already been trained in that session. The log for the dropout model starts from an untrained state (0.7365 in epoch 1).

The competition's `test.csv` is not used and no submission file is produced.

## Repository contents

- `Fake_news_Classifier.ipynb` - preprocessing, encoding, both LSTM models, and evaluation.
- `README.md` - this file.

## Running it

```
pip install pandas numpy tensorflow nltk scikit-learn
```

- Download `train.csv` from the competition below and place it in the repository root, next to the notebook; it is read with a relative path.
- The notebook downloads the NLTK `stopwords` corpus at run time.
- The saved run used TensorFlow 2.4.1. The evaluation cells call `model.predict_classes`, which is deprecated in that version and removed in later ones; on newer TensorFlow replace it with `(model.predict(X_test) > 0.5).astype("int32")`.
- Open with `jupyter notebook Fake_news_Classifier.ipynb` and run the cells in order.

## Data

Kaggle Fake News competition (`train.csv`; the competition also provides `test.csv`): https://www.kaggle.com/competitions/fake-news/

The data is not included in this repository.
