import sys
import os


# Construct the path
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# print(path)
# Add the path to sys.path
sys.path.append(path)
# Change the working directory
os.chdir(path)

# print(os.path.join('Models', 'Must', 'DEL_must_model.csv'))
from Models.Must.Neural_Network._utils import *
from Models.Must.Neural_Network.Must_Dataset_processing import *
from Models.Must.Neural_Network.Must_FNN import *

if __name__ == "__main__":
    must_df = pd.read_csv(r"Models/Must/DEL_must_model.csv", sep='\t', header=0)

    # Load the data
    train_data = must_df.iloc[:200][['Windspeed', 'STDeV']]
    train_labels = must_df.iloc[:200]['Leq_x']
    validation_data = must_df.iloc[200:400][['Windspeed', 'STDeV']]
    validation_labels = must_df.iloc[200:400]['Leq_x']
    test_data = must_df.iloc[400:][['Windspeed', 'STDeV']]
    test_labels = must_df.iloc[400:]['Leq_x']

    # Create the dataset
    datasets = {
        'train': Must_Dataset(train_data, train_labels),
        'validation': Must_Dataset(validation_data, validation_labels),
        'test': Must_Dataset(test_data, test_labels)
    }

    # Create the dataloader
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=32, shuffle=True),
        'validation': DataLoader(datasets['validation'], batch_size=32, shuffle=False),
        'test': DataLoader(datasets['test'], batch_size=32, shuffle=False)
    }

    # Create the model
    input_size = 2
    hidden_size = 128
    # num_layers = 2, THis is already defined in Must_FNN.py
    output_size = 1
    model = Must_model(input_size=input_size, 
                        hidden_size=hidden_size, 
                        # num_layers=num_layers, 
                        output_size=output_size,
                        )
    
    
    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    early_stopping = EarlyStopping(patience=10)
    # Train the model
    n_epochs = 500
    trained_model, train_losses, validation_losses = train(model, 
                                                           dataloaders, 
                                                           loss_fn, optimizer, 
                                                           n_epochs, 
                                                           device=device, 
                                                           early_stopping=early_stopping, 
                                                           print_freq=10,
                                                           )

    # Plot the training and validation losses
    plot_losses(train_losses, validation_losses)

    # Evaluate the model and plot inference
    test_loss = evaluate(trained_model, dataloaders['test'], loss_fn, device=device)
    plot_inference(trained_model, dataloaders['test'], device=device) # Needs to be altered to only plot inference for a couple of datapoints
