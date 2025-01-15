from Models.Should._utils import *
from Models.Should.Dataset_processing import *
from Models.Should.Should_LSTM import *

if __name__ == "__main__":
    # Load the data
    dataset_path = 'data/should_data/' # Replace in Kaggle with actual dataset path
    output_list = os.listdir(os.path.join(dataset_path, 'outputs'))
    input_list = os.listdir(os.path.join(dataset_path, 'inputs'))

    # Split the data
    data_list = pd.DataFrame([input_list, output_list]).T
    train_data, test_data, validation_data = split_dataset(data_list, test_size=0.15, validation_size=0.15)

    # Split features and labels
    train_features, train_labels = train_data.iloc[:, 0], train_data.iloc[:, 1]
    validation_features, validation_labels = validation_data.iloc[:, 0], validation_data.iloc[:, 1]
    test_features, test_labels = test_data.iloc[:, 0], test_data.iloc[:, 1]

    # Create the dataset
    datasets = {
        'train': Should_Dataset(train_features, train_labels),
        'validation': Should_Dataset(validation_features, validation_labels),
        'test': Should_Dataset(test_features, test_labels)
    }

    # Create the dataloader
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=32, shuffle=True, num_workers=4),
        'validation': DataLoader(datasets['validation'], batch_size=32, shuffle=True, num_workers=4),
        'test': DataLoader(datasets['test'], batch_size=32, shuffle=True, num_workers=4)
    }

    # Create the model
    input_size = 1500
    hidden_size = 1500
    num_layers = 2
    dropout = 0.2
    model = Should_model(input_size=input_size, 
                        hidden_size=hidden_size, 
                        num_layers=num_layers, 
                        dropout=dropout
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

    plot_inference(trained_model, dataloaders['test'], num_inf=1, device=device) # Needs to be altered to only plot inference for a couple of datapoints
