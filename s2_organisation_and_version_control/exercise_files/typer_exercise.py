import typer
from typing import Annotated
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

app = typer.Typer()
train_app = typer.Typer()
app.add_typer(train_app, name="train")

def load_dataset():
    # Load the dataset
    data = load_breast_cancer()
    x = data.data
    y = data.target

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test

@train_app.command()
def svm(kernel: str = "linear", output_file: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt") -> None:
    """Train a SVM model."""
    train(output_file, model = "SVM", kernel = kernel)


@train_app.command()
def knn(n_neighbors: int = 5, output_file: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt") -> None:
    """Train a KNN model."""
    train(output_file, model = "knn", n_neighbors = n_neighbors)

@app.command()
def train(output: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt", model: str = "SVM", **kwargs):
    """Train and evaluate the model."""

    x_train, _, y_train, _ = load_dataset()

    if model == "SVM":
        # Train a Support Vector Machine (SVM) model
        model = SVC(kernel=kwargs["kernel"], random_state=42)
        model.fit(x_train, y_train)
    elif model == "knn":
        model = KNeighborsClassifier(n_neighbors=kwargs["n_neighbors"])
        model.fit(x_train, y_train)
    else:
        raise typer.BadParameter(f"Unknown model '{model}'")

    # save the model
    with open(output, "wb") as f:
        pickle.dump(model, f)
    print("Model successfully trained")

@app.command()
def evaluate(model_path: Annotated[str, typer.Argument()]):

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    _, x_test, _, y_test = load_dataset()
    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report


# this "if"-block is added to enable the script to be run from the command line
if __name__ == "__main__":
    app()
