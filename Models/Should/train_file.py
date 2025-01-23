from Models.Should._utils import *
from Models.Should.Dataset_creation import *
from Models.Should.Should_LSTM import *

if __name__ == "__main__":
    # Load the data
    dataset_path = '/kaggle/input/should-azimuth-complete-simulations/Must_Should_processed'
    file_list = os.listdir(dataset_path)

    # Split the data
    train_data, test_data, validation_data = split_dataset(file_list, test_size=0.15, validation_size=0.15)

    # Specify the load axis used
    load_axis = 'Mxb1'
    # load_axis = 'Myb1'

    # Calculate average and standard deviation of training set labels
    train_labels_mean, train_labels_stdev = calculate_average_and_std(dataset_path, train_data, load_axis)

    # Create the dataset
    datasets = {
        'train': Should_Dataset(dataset_path, train_data, load_axis, train_labels_mean, train_labels_stdev),
        'validation': Should_Dataset(dataset_path, validation_data, load_axis, train_labels_mean, train_labels_stdev),
        'test': Should_Dataset(dataset_path, test_data, load_axis, train_labels_mean, train_labels_stdev)
    }

    # Create the dataloader
    batch_size = 64

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'validation': DataLoader(datasets['validation'], batch_size=batch_size, shuffle=True, num_workers=4),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=True, num_workers=4)
    }

    print(f'The length of the train dataset is: {len(datasets["train"])}')
    print(f'The length of the validation dataset is: {len(datasets["validation"])}')
    print(f'The length of the test dataset is: {len(datasets["test"])}')

    # Create the model
    input_size = 2
    hidden_size = 128
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'The specified device is: {device}')
    print(f'The model architecture is:\n{model}')
    
    # Train the model
    save_directory = '/kaggle/working'
    n_epochs = 50
    model, train_losses, validation_losses, validation_dels = train(
        model,                                      
        dataloaders,
        loss_fn, 
        optimizer, 
        n_epochs,
        save_directory,
        device=device, 
        early_stopping=5, 
        print_freq=1,
    )

    # Plot the training and validation losses
    plot_losses(train_losses, validation_losses, validation_dels)

    ### FOR IF YOU WANT TO LOAD A DIFFERENT MODEL
    ## USE CORRESPONDING LOAD AXIS
    model.load_state_dict(torch.load(f'/kaggle/working/model_{date.today()}_Mxb1.pt', weights_only=True))

    # Evaluate the model and plot inference
    test_loss, test_del_error = evaluate(model, dataloaders['test'], loss_fn, device=device)
    print(f'The loss over the test set is: {test_loss}\nThe test DEL error is: {test_del_error}')
    plot_inference(model, dataloaders['test'], num_inf=3, device=device)