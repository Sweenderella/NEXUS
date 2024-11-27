import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sympy import symbols
from torch.optim.lr_scheduler import ReduceLROnPlateau
import regulations_QnA
from regulations_QnA import  symbolic_reasoning_regulations

# Define symbolic variables for neurosymbolic reasoning
CO2, NOx, PM25, VOC, SO2 = symbols('CO2 NOx PM25 VOC SO2')

#---model perfection


import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class EmissionsNet(nn.Module):
    def __init__(self):
        super(EmissionsNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.fc(x)

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
              'Fuel Consumption Hwy (L/100 km)', 'mileage (km/l)', 'Fuel Type',
              'Fuel Consumption Combined']]
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
    return model, scaler

#-----









# Neural network for emissions prediction
# class EmissionsNet(nn.Module):
#     def __init__(self):
#         super(EmissionsNet, self).__init__()
#         self.fc1 = nn.Linear(6, 128)  # Input: 6 features
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 1)  # Output: CO2 Emissions (g/km)
#         self.dropout = nn.Dropout(0.3)  # Added dropout for regularization
# 
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(torch.relu(self.fc2(x)))
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

# Rule-based reasoning for emissions limits
def rule_based_reasoning(co2_emissions, country='USA'):
    limits = {
        'USA': {
            'CO2': 120,
            'NOx': 0.07,
            'PM25': 0.03,
            'VOC': 0.07,
            'SO2': 0.02,
            'source': "https://www.epa.gov"
        },
        'UK': {
            'CO2': 95,
            'NOx': 0.08,
            'PM25': 0.02,
            'VOC': 0.06,
            'SO2': 0.015,
            'source': "https://www.gov.uk"
        },
        'India': {
            'CO2': 100,
            'NOx': 0.1,
            'PM25': 0.05,
            'VOC': 0.08,
            'SO2': 0.03,
            'source': "https://www.cpcb.nic.in"
        }
    }

    limit = limits[country]
    result = {}
    if co2_emissions > limit['CO2']:
        exceedance = co2_emissions - limit['CO2']
        result['status'] = 'Exceeds Limit'
        result['exceedance'] = exceedance
        result['recommendation'] = (
            f"The predicted CO2 emissions of {co2_emissions:.2f} g/km exceed the limit of {limit['CO2']} g/km by "
            f"{exceedance:.2f} g/km. Consider more fuel-efficient vehicles or alternative fuel options."
        )
    else:
        result['status'] = 'Within Limit'
        result['exceedance'] = 0
        result['recommendation'] = (
            f"The predicted CO2 emissions of {co2_emissions:.2f} g/km are within the permissible limit of {limit['CO2']} g/km."
        )
    result['source'] = limit['source']
    return result

# Vehicle selection prompt for multiple matches
def interactive_vehicle_selection(matches):
    print("Multiple matches found. Please choose the transmission type:")
    for idx, match in enumerate(matches):
        print(f"{idx + 1}: {match}")
    choice = int(input("Enter your choice (number): ")) - 1
    return matches[choice]

# Train the model with enhancements
# def train_model(csv_file):
#     data = pd.read_csv(csv_file)
# 
#     # Data preprocessing
#     X = data[['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
#               'Fuel Consumption Hwy (L/100 km)', 'mileage (km/l)', 'Fuel Type']]
#     y = data['CO2 Emissions(g/km)'].values
# 
#     le = LabelEncoder()
#     X['Fuel Type'] = le.fit_transform(X['Fuel Type'])
# 
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
# 
#     X = torch.tensor(X, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
# 
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
#     model = EmissionsNet()
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
# 
#     best_loss = float('inf')
#     patience_counter = 0
#     patience_limit = 10
# 
#     for epoch in range(1000):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)
#         loss.backward()
#         optimizer.step()
# 
#         model.eval()
#         with torch.no_grad():
#             val_outputs = model(X_test)
#             val_loss = criterion(val_outputs, y_test)
# 
#         scheduler.step(val_loss)
# 
#         if val_loss < best_loss:
#             best_loss = val_loss
#             patience_counter = 0
#         else:
#             patience_counter += 1
# 
#         if patience_counter >= patience_limit:
#             print(f"Early stopping triggered at epoch {epoch}")
#             break
# 
#         if epoch % 100 == 0:
#             print(f'Epoch [{epoch}/1000], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
# 
#     print(f'Final Validation Loss: {best_loss:.4f}')
#     return model, scaler

# Main function
if __name__ == "__main__":
    csv_file_path = 'CO2 Emissions_Canada.csv'
    model, scaler = train_model(csv_file_path)

    example_input = [[2.5, 4, 9.5, 7.0, 11.0, 0, 3.0]]
    example_input = scaler.transform(example_input)
    example_input = torch.tensor(example_input, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predicted_co2 = model(example_input).item()
        print(f"Predicted CO2 Emissions (g/km): {predicted_co2:.2f}")

    reasoning_result = rule_based_reasoning(predicted_co2, country='USA')
    print(reasoning_result)
    
    print('moving ahead with symbolic reasoning  ----> ')
    print('referring regulations  --------------------------------->')
    compliance_query = 'how does carbon affect the human health?'
    symbolic_reasoning_regulations(compliance_query)
