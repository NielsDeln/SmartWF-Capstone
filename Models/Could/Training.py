from Could_LSTM import *
from Could_functions import *
from datetime import date


if __name__ == "__main__":
    # Load the data
    train_data = []
    train_labels = []
    validation_data = []    
    validation_labels = [] 
    test_data = []
    test_labels = []

    # Create the dataset
    datasets = {
        'train': WindTurbineDataset(train_data, train_labels),
        'validation': WindTurbineDataset(validation_data, validation_labels),
        'test': WindTurbineDataset(test_data, test_labels)
    }

    # Create the dataloader
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=32, shuffle=True),
        'validation': DataLoader(datasets['validation'], batch_size=32, shuffle=False),
        'test': DataLoader(datasets['test'], batch_size=32, shuffle=False)
    }


    # Create model
    input_channels = 3          # Number of channels in 2D wind speed field
    input_hight = 33            # Height of the 2D input field
    input_width = 41            # Width of the 2D input field
    feature_dim = 64            # Feature dimension from CNN
    hidden_dim = 128            # Hidden dimension of LSTM
    output_dim = 2              # Output dimension (load value)
    num_layers = 2              # Number of LSTM layers
    seq_len = 1500              # Length of the input sequence

    batch_size = 8              # Batch size
    learning_rate = 0.001       # Learning rate optimizer
    num_epochs = 15             # Amount of epochs

    model = WindTurbineLoadPredictor(input_channels=input_channels, 
                                     feature_dim=feature_dim, 
                                     hidden_dim=hidden_dim, 
                                     output_dim=output_dim, 
                                     num_layers=num_layers)
    
    
    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train the model
    trained_model, train_losses, validation_losses = train(model, 
                                                           dataloaders['train'], 
                                                           loss_fn, 
                                                           optimizer, 
                                                           num_epochs, 
                                                           device=device, 
                                                           early_stopping=-1, 
                                                           print_freq=10,
                                                           )

    # Plot the training and validation losses
    plot_losses(train_losses, validation_losses)

    # Evaluate the model and plot inference
    test_loss = evaluate(trained_model, dataloaders['test'], loss_fn, device=device)
    plot_inference(trained_model, dataloaders['test'], device=device) # Needs to be altered to only plot inference for a couple of datapoints
