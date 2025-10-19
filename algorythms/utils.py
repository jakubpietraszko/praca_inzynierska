import torch
import pathlib
import os

def trainStep(model, loader, optimizer, criterion, device):
    model.train()
    totalLoss, correct = 0.0, 0
    for xBatch, yBatch in loader:
        xBatch, yBatch = xBatch.to(device), yBatch.to(device)
        optimizer.zero_grad()
        outputs = model(xBatch)
        loss = criterion(outputs, yBatch)
        loss.backward()
        optimizer.step()
        totalLoss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == yBatch).sum().item()


    avgLoss = totalLoss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avgLoss, accuracy


def testStep(model, loader, criterion, device):
    model.eval()
    totalLoss, correct = 0.0, 0
    with torch.no_grad():
        for xBatch, yBatch in loader:
            xBatch, yBatch = xBatch.to(device), yBatch.to(device)
            outputs = model(xBatch)
            loss = criterion(outputs, yBatch)
            totalLoss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == yBatch).sum().item()
    avgLoss = totalLoss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avgLoss, accuracy


def train(model, trainLoader, testLoader, optimizer, criterion, device, epochs, dir):
    model = model.to(device)
    bestTestLoss = float('inf')
    bestModelPath = pathlib.Path(os.path.join("models", dir, "best_model.pth"))
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_epoch = 0
    for epoch in range(epochs):
        trainLoss, trainAccuracy = trainStep(model, trainLoader, optimizer, criterion, device)
        testLoss, testAccuracy = testStep(model, testLoader, criterion, device)
        train_losses.append(trainLoss)
        test_losses.append(testLoss)
        train_accuracies.append(trainAccuracy)
        test_accuracies.append(testAccuracy)

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {trainLoss:.4f}, Train Accuracy: {trainAccuracy:.4f} - "
              f"Test Loss: {testLoss:.4f}, Test Accuracy: {testAccuracy:.4f}")

        if testLoss < bestTestLoss:
            best_epoch = epoch + 1
            bestTestLoss = testLoss
            torch.save(model.state_dict(), bestModelPath)

    model.load_state_dict(torch.load(bestModelPath))

    return model, train_losses, test_losses, train_accuracies, test_accuracies, best_epoch
