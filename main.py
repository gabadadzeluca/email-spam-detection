from src.explore_data import read_data
from src.train import train_model

if __name__ == "__main__":
    train_data, test_data = read_data()

    # Evaluate the model on the test set
    if train_data:
        model, vectorizer = train_model(train_data)
        print("Success! Model is ready.")
        
        test_labels = [item[0] for item in test_data]
        test_texts  = [item[1] for item in test_data]

        X_test_features = vectorizer.transform(test_texts)
        predictions = model.predict(X_test_features)
        print(f"Predictions for samples: {predictions}")
    
        accuracy = sum([pred == actual for pred, actual in zip(predictions, test_labels)]) / len(test_labels)
        print("Test accuracy:", accuracy)
    
    else:
        print("No data found to train on.")