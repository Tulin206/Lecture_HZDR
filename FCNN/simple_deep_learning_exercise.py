#%% md
# # Simple Deep Learning with PyTorch (Exercise Version)
# 
# This notebook implements a basic deep learning model using PyTorch to classify MNIST digits. You'll learn by completing the exercises:
# - Building a simple neural network
# - Training and validation process
# - Performance visualization
# 
# ## Step 1: Import Required Libraries
# 
# First, import all necessary libraries for our deep learning project.
#%%
# TODO: Import the required PyTorch libraries and other dependencies
# Hint: We need torch, torch.nn, torch.optim, torchvision, and matplotlib




#%% md
# ## Step 2: Set Random Seed
# 
# Setting a random seed ensures reproducible results across different runs.
#%%
# TODO: Set a random seed using torch.manual_seed()
# Use seed value: 42


#%% md
# ## Step 3: Define Neural Network Architecture
# 
# Create a simple neural network for MNIST classification:
# 1. Input layer: 784 neurons (28x28 MNIST images flattened)
# 2. Hidden layer: 128 neurons with ReLU activation
# 3. Output layer: 10 neurons (one for each digit)
#%%
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()  # Flatten the 28x28 image to 784 pixels

        # TODO: Complete the neural network architecture using nn.Sequential
        # Requirements:
        # 1. Linear layer from 784 -> 128
        # 2. ReLU activation
        # 3. Linear layer from 128 -> 10
        self.layers = nn.Sequential(
            # Your code here

        )

    def forward(self, x):
        # TODO: Implement the forward pass
        # 1. Flatten the input
        # 2. Pass through layers
        # Your code here


#%% md
# ## Step 4: Define Training Function
# 
# The training function is provided. Study it carefully to understand:
# - How the training loop works
# - How metrics are calculated
# - The difference between training and validation phases
#%%
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # TODO: Implement the training step
            # 1. Forward pass
            # 2. Calculate loss
            # 3. Zero gradients
            # 4. Backward pass
            # 5. Optimizer step
            # Your code here


            # Calculate training statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate and store training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        print('-' * 60)

    return train_losses, val_losses, train_accuracies, val_accuracies

#%% md
# ## Step 5: Define Visualization Function
# 
# The visualization function is partially provided. Complete the missing parts to create informative plots.
#%%
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Create a figure with two subplots
    plt.figure(figsize=(12, 5))

    # TODO: Complete the loss plot
    plt.subplot(1, 2, 1)
    # Your code here: Plot training and validation losses

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # TODO: Complete the accuracy plot
    plt.subplot(1, 2, 2)
    # Your code here: Plot training and validation accuracies

    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

#%% md
# ## Step 6: Main Execution Function
# 
# Complete the main function to orchestrate the training process.
#%%
def main():
    # TODO: Set the hyperparameters
    # Your code here: Define batch_size, learning_rate, and num_epochs


    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # TODO: Split the dataset into training and validation sets (80% train, 20% validation)
    # Your code here


    # TODO: Create data loaders
    # Your code here: Create train_loader and val_loader


    # TODO: Initialize the model, loss function, and optimizer
    # Your code here


    # Train the model and get metrics
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs
    )

    # Plot the results
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

#%% md
# ## Step 7: Execute Training
# 
# Run the main function when the script is executed directly.
#%%
if __name__ == '__main__':
    main()
