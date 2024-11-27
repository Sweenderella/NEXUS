import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import co2_prediction_withXAI
from co2_prediction_withXAI import rule_based_reasoning
import regulations_QnA
from regulations_QnA import symbolic_reasoning_regulations


# Define the neural network
class EmissionsNet(nn.Module):
    def __init__(self):
        super(EmissionsNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.fc(x)

# Train the model
def train_model(csv_file):
    data = pd.read_csv(csv_file)

    # Handle missing values
    data = data.dropna()

    # Feature engineering
    data['Fuel Consumption Combined'] = (
        data['Fuel Consumption City (L/100 km)'] + data['Fuel Consumption Hwy (L/100 km)']
    ) / 2

    # Selecting features and target
    X = data[['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
              'Fuel Consumption Hwy (L/100 km)', 'mileage (km/l)', 'Fuel Type'
              ]]
    y = data['CO2 Emissions(g/km)'].values

    # Encoding categorical variables
    le = LabelEncoder()
    X['Fuel Type'] = le.fit_transform(X['Fuel Type'])

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataLoader for batch training
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define the model, loss function, and optimizer
    model = EmissionsNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 10

    for epoch in range(1000):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                val_outputs = model(batch_X)
                val_loss += criterion(val_outputs, batch_y).item()

        val_loss /= len(test_loader)
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/1000], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    print(f'Final Validation Loss: {best_loss:.4f}')
    return model, scaler, le

# Predict CO2 emissions
def predict_emissions(model, scaler, car_details):
    car_details = scaler.transform([car_details])
    car_details = torch.tensor(car_details, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return model(car_details).item()

# Get car details from the CSV
def get_car_details(csv_file, make, model, fuel_type, label_encoder):
    data = pd.read_csv(csv_file)
    car_row = data[(data['Make'] == make) & (data['Model'] == model) & (data['Fuel Type'] == fuel_type)]
    if car_row.empty:
        raise ValueError("Car details not found in the dataset.")
    car_features = car_row.iloc[0][['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
                                    'Fuel Consumption Hwy (L/100 km)', 'mileage (km/l)', 'Fuel Type']].values
    car_features[5] = label_encoder.transform([car_features[5]])[0]  # Encode fuel type
    print (car_features)
    return car_features

if __name__ == "__main__":
    csv_file_path = 'CO2 Emissions_Canada.csv'
    model, scaler, label_encoder = train_model(csv_file_path)

#     # Get user input
#     make = input("Enter car Make: ")
#     model_name = input("Enter car Model: ")
#     fuel_type = input("Enter car Fuel Type: ")
     
    #Sample User Input 1, 
    make = "VOLVO"
    model_name = "V60 T5"
    fuel_type = "X"
    
    #you can test with any [Input 2: make : TOYOTA, model = TACOMA , fuel type = X]
    #input 3 : make = FORD, model = ESCAPE, Fuel_type = X

    try:
        car_details = get_car_details(csv_file_path, make, model_name, fuel_type, label_encoder)
        predicted_co2 = predict_emissions(model, scaler, car_details)
        print(f"Predicted CO2 Emissions (g/km): {predicted_co2:.2f}")
        
        #output will be predicted carbon emissions : [Predicted CO2 Emissions (g/km) between 193 to 195] for VOlVO example
    except ValueError as e:
        print(e)
        
    print('Now moving towards the Compliance Report and Explanations')
    reasoning_result = rule_based_reasoning(predicted_co2, country='USA')
    print(reasoning_result)     #here is the compliance report is printed and explanation is given
    
    print('Referring to regulations now')
    #moving further , we have the regulations to check based on this data
    query = "What the emission limits for VOlVO light duty vehicle for the year 2027?"     #See output 1 image in gmail
    #moving on we are going to refer to environment regulations here now
    symbolic_reasoning_regulations (query)
    
    query = "What are the health impact due to exceeded carbon limits?"      #see output 2 image in gmail
    #moving on we are going to refer to environment regulations here now
    symbolic_reasoning_regulations (query)
    
