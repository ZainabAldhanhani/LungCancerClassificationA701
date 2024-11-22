import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix

# Training and evaluation function with a training loop
def train_and_evaluate(model_name, model, train_loader, val_loader, accuracies, f1_scores, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    # Evaluation on validation set
    model.eval()
    y_true, y_pred = [], []
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    accuracies.append(accuracy)
    f1_scores.append(f1)
    #save the model weights
    torch.save(model.state_dict(), f'{model_name}_weights.pth')
    print(f"Model saved to {model_name}_weights.pth")
