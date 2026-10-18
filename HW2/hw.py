
import os
import numpy as np
import pandas as pd


INPUT_DIM = 784               # 28 x 28 pixeli
OUTPUT_DIM = 10               # cifrele de la 0 la 9
LEARNING_RATE = 0.041          # am observat ca mai mic nu merge asa bine, nici mai mare
NUM_EPOCHS = 50               # de cate ori trecem prin tot setul de antrenare
BATCH_SIZE = 32               # cate imagini procesam odata inainte sa actualizam greutatile
VALIDATION_FRACTION = 0.2     # 20% din train merge la validare (regula 80-20)
RANDOM_SEED = 1               # seed fix -> rezultate reproductibile

# Daca suntem pe Kaggle luam datele de acolo, altfel (la mine pe PC) din folderul curent.
KAGGLE_DATA_PATH = "/kaggle/input/competitions/nn-2026-fii-homework-1"
DATA_DIRECTORY = KAGGLE_DATA_PATH if os.path.exists(KAGGLE_DATA_PATH) else "."


OUTPUT_DIRECTORY = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
MODEL_WEIGHTS_PATH = os.path.join(OUTPUT_DIRECTORY, "weights.npy")
MODEL_BIAS_PATH = os.path.join(OUTPUT_DIRECTORY, "bias.npy")
SUBMISSION_PATH = os.path.join(OUTPUT_DIRECTORY, "submission.csv")


# PERCEPTRON MODEL
# Un singur strat: 784 intrari -> 10 iesiri + softmax
# probabilitati = softmax(X @ W + b)

class PerceptronModel:
    def __init__(self, input_dim=INPUT_DIM, num_classes=OUTPUT_DIM,
                 rng_seed=RANDOM_SEED, learning_rate=LEARNING_RATE):
        rng = np.random.RandomState(rng_seed)

        # Initializare Xavier: greutatile de start sunt random in [-limit, +limit].
        # Limita depinde de nr. de intrari/iesiri ca sa nu pornim cu valori prea mari/mici.
        limit = np.sqrt(6 / (input_dim + num_classes))
        self.weights = rng.uniform(-limit, limit, (input_dim, num_classes))   # (784, 10)
        self.bias = np.zeros(num_classes)                                     # (10,)
        self.learning_rate = learning_rate

    def forward_propagation(self, input_data):
        # scorurile brute pentru fiecare clasa: (N,784) @ (784,10) + (10,) = (N,10)
        z = input_data @ self.weights + self.bias

        # softmax: scorurile devin probabilitati (pozitive, suma 1 pe fiecare rand).
        # Scadem maximul de pe rand inainte de exp() ca sa evitam overflow;
        # rezultatul final ramane acelasi.
        z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return z_exp / np.sum(z_exp, axis=1, keepdims=True)

    def compute_loss(self, y_pred, y_true):
        # cross-entropy: practic -log(probabilitatea data clasei corecte), mediata pe batch.
        # +1e-8 ca sa nu facem log(0).
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

    def step_gradient(self, X, y_true):
        # un pas de gradient descent pe un batch
        m = X.shape[0]
        y_pred = self.forward_propagation(X)

        # pentru softmax + cross-entropy gradientul fata de scoruri e simplu: pred - real
        error = y_pred - y_true

        delta_W = X.T @ error / m              # cat a contribuit fiecare pixel la eroare
        delta_b = np.sum(error, axis=0) / m    # eroarea medie pe fiecare clasa

        # scadem gradientul (mergem in directia in care loss-ul scade)
        self.weights -= self.learning_rate * delta_W
        self.bias -= self.learning_rate * delta_b

    def predict(self, input_data):
        # clasa prezisa = cea cu probabilitatea cea mai mare
        return np.argmax(self.forward_propagation(input_data), axis=1)

    def get_parameters_copy(self):
        # .copy() ca sa salvam valorile de acum, nu o referinta care se schimba la antrenare
        return self.weights.copy(), self.bias.copy()

    def set_parameters(self, parameters):
        self.weights, self.bias = parameters

    def save(self):
        np.save(MODEL_WEIGHTS_PATH, self.weights)
        np.save(MODEL_BIAS_PATH, self.bias)

    def load(self):
        self.weights = np.load(MODEL_WEIGHTS_PATH)
        self.bias = np.load(MODEL_BIAS_PATH)


# DATA HELPERS

def load_data(train=True):
    file_name = "train.npz" if train else "test.npz"
    data_file = np.load(os.path.join(DATA_DIRECTORY, file_name))

    # intindem imaginile 28x28 in vectori de 784 si aducem pixelii din 0..255 in 0..1
    X = data_file["data"].astype(np.float32).reshape(-1, INPUT_DIM) / 255.0

    # setul de test nu are etichete, doar cel de train
    y = data_file["labels"].astype(int) if train else None

    print(f"Loaded {'train' if train else 'test'} set: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y


def to_one_hot(y, num_classes=OUTPUT_DIM):
    # 3 -> [0,0,0,1,0,0,0,0,0,0]
    return np.eye(num_classes)[y]


def split_train_validation(X, y, validation_fraction=VALIDATION_FRACTION, rng_seed=RANDOM_SEED):
    # amestecam indicii, apoi primele 20% -> validare, restul -> antrenare
    perm = np.random.RandomState(rng_seed).permutation(len(X))
    num_val = int(validation_fraction * len(X))
    val_idx, train_idx = perm[:num_val], perm[num_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def accuracy(y_true, y_pred):
    # y_true si y_pred sunt etichete (numere 0-9), nu one-hot
    return np.mean(y_true == y_pred)


# TRAINER

class PerceptronTrainer:
    def __init__(self):
        self.model = PerceptronModel()

        # incarcam train-ul si il impartim in antrenare + validare
        X_all, y_all = load_data(train=True)
        self.X_train, self.y_train, self.X_val, self.y_val = split_train_validation(X_all, y_all)

        # etichetele one-hot le calculam o singura data
        self.Y_train = to_one_hot(self.y_train)
        self.Y_val = to_one_hot(self.y_val)

    def compute(self):
        m = self.X_train.shape[0]
        rng = np.random.RandomState(RANDOM_SEED)    # generator pentru amestecarea din fiecare epoca

        # tinem minte cele mai bune greutati (dupa acuratetea pe validare)
        best_val_acc, best_parameters = -1.0, self.model.get_parameters_copy()

        for epoch in range(NUM_EPOCHS):
            # amestecam datele la fiecare epoca (aceeasi ordine pt imagini si etichete!)
            perm = rng.permutation(m)
            X_shuf = self.X_train[perm]
            Y_shuf = self.Y_train[perm]
            epoch_loss = 0.0

            # parcurgem datele in batch-uri (ultimul poate fi mai mic)
            for i in range(0, m, BATCH_SIZE):
                X_batch = X_shuf[i:i + BATCH_SIZE]
                Y_batch = Y_shuf[i:i + BATCH_SIZE]

                y_pred = self.model.forward_propagation(X_batch)
                loss = self.model.compute_loss(y_pred, Y_batch)
                epoch_loss += loss * X_batch.shape[0]      # inmultim cu marimea batch-ului pt media corecta

                self.model.step_gradient(X_batch, Y_batch)

            # la finalul epocii vedem cum sta modelul
            avg_loss = epoch_loss / m
            train_acc = accuracy(self.y_train, self.model.predict(self.X_train))
            val_acc = accuracy(self.y_val, self.model.predict(self.X_val))
            val_loss = self.model.compute_loss(self.model.forward_propagation(self.X_val), self.Y_val)

            print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} | "
                  f"val loss: {val_loss:.4f} | train acc: {train_acc:.4f} | val acc: {val_acc:.4f}")

            # daca e cea mai buna epoca pana acum, salvam greutatile ei
            if val_acc > best_val_acc:
                best_val_acc, best_parameters = val_acc, self.model.get_parameters_copy()

        # revenim la cea mai buna epoca (nu neaparat ultima) si salvam pe disc
        self.model.set_parameters(best_parameters)
        print(f"Best val acc: {best_val_acc:.4f}")
        self.model.save()


# TESTER

class PerceptronTester:
    def __init__(self):
        # modelul se incarca din fisierele salvate de Trainer
        self.model = PerceptronModel()
        self.model.load()
        self.X_test, _ = load_data(train=False)    # nu avem etichete la test, de-asta "_"

    def compute(self):
        # prezicem clasele pentru imaginile de test
        preds = self.model.predict(self.X_test)
        print(f"Predicted {len(preds)} test samples.")
        return preds

    def generate_submission(self, preds):
        # formatul cerut de Kaggle: coloanele ID si target
        df = pd.DataFrame({"ID": np.arange(len(preds)), "target": preds})
        df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission file saved to {SUBMISSION_PATH}")


# MAIN

def main():
    # Antrenam perceptronul (si salvam weights.npy + bias.npy)
    PerceptronTrainer().compute()

    # Testam perceptronul (incarcam ce a salvat Trainer-ul)
    tester = PerceptronTester()
    preds = tester.compute()

    # Generam submission.csv
    tester.generate_submission(preds)


if __name__ == "__main__":
    main()