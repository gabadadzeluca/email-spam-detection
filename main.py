from src.explore_data import read_data
from src.train import train_model

if __name__ == "__main__":
    train_data, test_data = read_data()
    
    if train_data:
        model, vectorizer = train_model(train_data)
        print("Success! Model is ready.")
        
        # 3. Quick Test
        sample_text = ["URGENT! You have won a 1 week FREE membership to our prize jackpot!"]
        sample_features = vectorizer.transform(sample_text)
        prediction = model.predict(sample_features)
        print(f"Prediction for sample: {prediction[0]}")
    else:
        print("No data found to train on.")