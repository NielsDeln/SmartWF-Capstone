from Models.Should._utils import *
from Models.Should.Dataset_processing import *
from Models.Should.Should_LSTM import *

if __name__ == "__main__":
    # Load the data
    dataset_path = '/Users/niels/Desktop/TU Delft/Dataset/chunks' # Replace in Kaggle with actual dataset path
    file_list = os.listdir(dataset_path)

    # Split the data
    train_data, test_data, validation_data = split_dataset(file_list, test_size=0.15, validation_size=0.15)

    # Specify the load axis used
    load_axis = 'Mxb1'

    # Create the dataset
    datasets = {
        'train': Should_Dataset(dataset_path, train_data, load_axis),
        'validation': Should_Dataset(dataset_path, validation_data, load_axis),
        'test': Should_Dataset(dataset_path, test_data, load_axis)
    }

    # Create the dataloader
    batch_size = 7

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'validation': DataLoader(datasets['validation'], batch_size=batch_size, shuffle=True, num_workers=4),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=True, num_workers=4)
    }

    # Create the model
    input_size = 2
    hidden_size = 64
    num_layers = 3
    dropout = 0.2
    proj_size = 1
    model = Should_model(input_size=input_size, 
                        hidden_size=hidden_size, 
                        num_layers=num_layers, 
                        dropout=dropout,
                        proj_size=proj_size,
                        batch_size=batch_size
                        )


    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'The specified device is: {device}')
    print(f'The model architecture is:\n{model}')

    # Train the model
    n_epochs = 50
    model, train_losses, validation_losses = train(model, 
                                                        dataloaders['train'], 
                                                        dataloaders['validation'],
                                                        loss_fn, 
                                                        optimizer, 
                                                        n_epochs, 
                                                        device=device, 
                                                        early_stopping=10, 
                                                        print_freq=1,
                                                        )

    # Plot the training and validation losses
    plot_losses(train_losses, validation_losses)

    # Evaluate the model and plot inference
    test_loss = evaluate(model, dataloaders['test'], loss_fn, device=device)

    plot_inference(model, dataloaders['test'], num_inf=2, device=device)
