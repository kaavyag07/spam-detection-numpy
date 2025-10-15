



import numpy as np
import csv

def load_dataset(filename):

    with open(filename , 'r',newline='') as csvfile:
        texts = []
        lables = []
        for line in csvfile:
            parts = line.strip().split(',', 1)
            if len(parts)<2:
                continue
            lable, text = parts
            texts.append(text.lower())
            lables.append(1 if lable.strip().lower() == "spam" else 0)
        return texts, np.array(lables)
    
def tokenize(text):
    return text.lower().split()

def build_vocab(texts):
    vocab = {}
    for text in texts:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def vectorize(text, vocab):
    vec = np.zeros(len(vocab))
    for word in tokenize(text):
        if word in vocab:
            vec[vocab[word]] += 1
    return vec

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cal_loss(X,y,b,w):
    z = np.dot(X,w) + b
    pred = sigmoid(z)
    loss = -np.mean(y*np.log(pred + 1e-15) + (1-y)*np.log(1 - pred + 1e-15))
    return loss

def compute_gradient(X,y,b,w):
    m = X.shape[0]
    z = np.dot(X,w) + b
    pred = sigmoid(z)
    dw = np.dot(X.T, pred - y)/m
    db = np.mean(pred - y)
    return dw, db

def train(X, y, learning_rate=0.1, iterations=1000):
    w = np.zeros(X.shape[1])
    b = 0
    for i in range(iterations):
        dw, db = compute_gradient(X, y, b, w)
        w -= learning_rate * dw
        b -= learning_rate * db
        if i % 100 == 0:
            print(f"Iter {i}, Loss: {cal_loss(X, y, b, w):.4f}")
    return w, b

def predict(text, vocab, w, b):
    x = vectorize(text, vocab)
    prob = sigmoid(np.dot(x,w)+b)
    return 1 if prob>0.5 else 0


texts, labels = load_dataset("/Users/priyanshuswami/Downloads/smsspamcollection/SMSSpamCollection")


vocab = build_vocab(texts)
X = np.array([vectorize(text, vocab) for text in texts], dtype=np.float64)

y = labels

# Train model
w, b = train(X, y)

# Test on new messages
test_msgs = [
    "Claim your free vacation now!",
    "See you at the meeting later.",
    "Exclusive deal for you only",
    "Can you call me tonight?",
    "Earn free money",
    "HMV BONUS SPECIAL 500 pounds of genuine HMV vouchers to be won. Just answer 4 easy questions."

]
print("\n--- Test Predictions ---")
for msg in test_msgs:
    result = predict(msg, vocab, w, b)
    print(f"{msg} => {'SPAM' if result else 'NOT SPAM'}")
