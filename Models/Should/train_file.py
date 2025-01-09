from _utils import *
from Dataset_class import *
from Should_LSTM import *

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
        'train': Should_Dataset(train_data, train_labels),
        'validation': Should_Dataset(validation_data, validation_labels),
        'test': Should_Dataset(test_data, test_labels)
    }

    # Create the dataloader
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=32, shuffle=True),
        'validation': DataLoader(datasets['validation'], batch_size=32, shuffle=False),
        'test': DataLoader(datasets['test'], batch_size=32, shuffle=False)
    }

    # Create the model
    input_size = 1
    hidden_size = 128
    num_layers = 2
    output_size = 1
    model = Should_model(input_size=input_size, 
                        hidden_size=hidden_size, 
                        num_layers=num_layers, 
                        output_size=output_size,
                        )
    
    
    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train the model
    n_epochs = 500
    trained_model, train_losses, validation_losses = train(model, 
                                                           dataloaders['train'], 
                                                           loss_fn, optimizer, 
                                                           n_epochs, 
                                                           device=device, 
                                                           early_stopping=10, 
                                                           print_freq=10,
                                                           )

    # Plot the training and validation losses
    plot_losses(train_losses, validation_losses)

    # Evaluate the model and plot inference
    test_loss = evaluate(trained_model, dataloaders['test'], loss_fn, device=device)
    plot_inference(trained_model, dataloaders['test'], device=device) # Needs to be altered to only plot inference for a couple of datapoints
