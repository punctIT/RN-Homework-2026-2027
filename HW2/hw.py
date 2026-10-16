#
import os
import numpy as np
import pandas as pd


NUMBER_OF_INPUT_FEATURES = 784        # 28 x 28 pixeli
NUMBER_OF_CLASSES = 10
LEARNING_RATE = 0.05
NUMBER_OF_EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_FRACTION = 0.2             # 20% din train merge la validare
RANDOM_SEED = 1

# Pe Kaggle: calea de mai jos. Local: foloseste kagglehub (vezi mai jos).
DATA_DIRECTORY = "/kaggle/input/competitions/nn-2026-fii-homework-1"
DATA_DIRECTORY = "."
if not os.path.exists(DATA_DIRECTORY):
    import kagglehub
    DATA_DIRECTORY = kagglehub.competition_download("nn-2026-fii-homework-1")

OUTPUT_DIRECTORY = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
SUBMISSION_FILE_PATH = os.path.join(OUTPUT_DIRECTORY, "submission.csv")



class PerceptronModel:
    def __init__(self, number_of_input_features=NUMBER_OF_INPUT_FEATURES,
                 number_of_classes=NUMBER_OF_CLASSES,
                 learning_rate=LEARNING_RATE, random_seed=RANDOM_SEED):
        random_generator = np.random.RandomState(random_seed)
        xavier_initialization_limit = np.sqrt(6 / (number_of_input_features + number_of_classes))
        self.weights = random_generator.uniform(
            -xavier_initialization_limit, xavier_initialization_limit,
            (number_of_input_features, number_of_classes))
        self.bias = np.zeros(number_of_classes)
        self.learning_rate = learning_rate

    def forward_propagation(self, input_images):
        weighted_sums = input_images @ self.weights + self.bias
        exponentials = np.exp(weighted_sums - weighted_sums.max(axis=1, keepdims=True))
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def cross_entropy_loss(self, predicted_probabilities, one_hot_targets):
        return -np.sum(one_hot_targets * np.log(predicted_probabilities + 1e-8)) / one_hot_targets.shape[0]

    def gradient_descent_step(self, input_images, one_hot_targets):
        batch_size = input_images.shape[0]
        prediction_error = self.forward_propagation(input_images) - one_hot_targets
        self.weights -= self.learning_rate * (input_images.T @ prediction_error) / batch_size
        self.bias -= self.learning_rate * prediction_error.sum(axis=0) / batch_size

    def predict_classes(self, input_images):
        return np.argmax(self.forward_propagation(input_images), axis=1)

    def get_parameters_copy(self):
        return self.weights.copy(), self.bias.copy()

    def set_parameters(self, saved_parameters):
        self.weights, self.bias = saved_parameters


def load_data():
    training_file = np.load(os.path.join(DATA_DIRECTORY, "train.npz"))
    test_file = np.load(os.path.join(DATA_DIRECTORY, "test.npz"))
    training_images = training_file["data"].astype(np.float32).reshape(-1, NUMBER_OF_INPUT_FEATURES) / 255.0
    training_labels = training_file["labels"].astype(int)
    test_images = test_file["data"].astype(np.float32).reshape(-1, NUMBER_OF_INPUT_FEATURES) / 255.0
    return training_images, training_labels, test_images


def convert_to_one_hot(labels, number_of_classes=NUMBER_OF_CLASSES):
    return np.eye(number_of_classes)[labels]


def split_train_validation(images, labels, validation_fraction=VALIDATION_FRACTION, random_seed=RANDOM_SEED):
    shuffled_indices = np.random.RandomState(random_seed).permutation(len(images))
    number_of_validation_examples = int(validation_fraction * len(images))
    validation_indices = shuffled_indices[:number_of_validation_examples]
    training_indices = shuffled_indices[number_of_validation_examples:]
    return (images[training_indices], labels[training_indices],
            images[validation_indices], labels[validation_indices])



def train_model(model, training_images, training_labels, validation_images, validation_labels):
    one_hot_training_labels = convert_to_one_hot(training_labels)
    one_hot_validation_labels = convert_to_one_hot(validation_labels)
    number_of_training_examples = training_images.shape[0]
    random_generator = np.random.RandomState(RANDOM_SEED)
    best_validation_accuracy, best_parameters = -1.0, model.get_parameters_copy()

    for epoch in range(NUMBER_OF_EPOCHS):
        shuffled_order = random_generator.permutation(number_of_training_examples)
        shuffled_images = training_images[shuffled_order]
        shuffled_targets = one_hot_training_labels[shuffled_order]
        total_epoch_loss = 0.0

        for batch_start in range(0, number_of_training_examples, BATCH_SIZE):
            image_batch = shuffled_images[batch_start:batch_start + BATCH_SIZE]
            target_batch = shuffled_targets[batch_start:batch_start + BATCH_SIZE]
            total_epoch_loss += model.cross_entropy_loss(
                model.forward_propagation(image_batch), target_batch) * image_batch.shape[0]
            model.gradient_descent_step(image_batch, target_batch)

        training_accuracy = np.mean(model.predict_classes(training_images) == training_labels)
        validation_accuracy = np.mean(model.predict_classes(validation_images) == validation_labels)
        validation_loss = model.cross_entropy_loss(
            model.forward_propagation(validation_images), one_hot_validation_labels)
        print(f"Epoch {epoch + 1:3d}/{NUMBER_OF_EPOCHS} | loss {total_epoch_loss / number_of_training_examples:.4f} | "
              f"val loss {validation_loss:.4f} | train acc {training_accuracy:.4f} | val acc {validation_accuracy:.4f}")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy, best_parameters = validation_accuracy, model.get_parameters_copy()

    model.set_parameters(best_parameters)       # revenim la cea mai buna epoca
    print(f"Best val acc: {best_validation_accuracy:.4f}")


# ---------------- MAIN ----------------
def main():
    all_training_images, all_training_labels, test_images = load_data()
    print("Train:", all_training_images.shape, "| Test:", test_images.shape)

    training_images, training_labels, validation_images, validation_labels = \
        split_train_validation(all_training_images, all_training_labels)
    model = PerceptronModel()
    train_model(model, training_images, training_labels, validation_images, validation_labels)

    test_predictions = model.predict_classes(test_images)
    pd.DataFrame({"ID": np.arange(len(test_predictions)), "target": test_predictions}).to_csv(
        SUBMISSION_FILE_PATH, index=False)
    print(f"Submission salvat in {SUBMISSION_FILE_PATH}")


if __name__ == "__main__":
    main()