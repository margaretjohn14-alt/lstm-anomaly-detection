import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#Load the data
train_data = np.load("C:/Users/john_mg/telemanom/archive/data/data/train/P-1.npy")
test_data = np.load("C:/Users/john_mg/telemanom/archive/data/data/test/P-1.npy")

'''
2872 timesteps, 3 features (sensor readings)
25 sensors, 1 label (anomaly or not)
Data length: 2872
seq_len: 30
Number of windows created: 2872 - 30 = 2842 windows
'''
#Use only the first feature (channel 0) to keep it simple
train = torch.FloatTensor(train_data[:, 0]).unsqueeze(-1) #shape(2872, 1)
test = torch.FloatTensor(test_data[:, 0]).unsqueeze(-1)

#Create Sliding Windows
def create_sequences(data, seq_len=30): #30 timesteps per sequence
    sequences= [] #list to hold the sequences
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len]) #append the window of data to the list
        #cut out 30 timesteps starting at position i, and add that chunk to my list
    return torch.stack(sequences)#convert list of sequences to a tensor
    
SEQ_LEN = 30
X_train = create_sequences(train, SEQ_LEN) #shape (2842, 30, 1)
X_test = create_sequences(test, SEQ_LEN)#shape (2842, 30, 1)

print("X_train shape:", X_train.shape)
print("X_test shape", X_test.shape)
train_loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)
#test_loader = Dataloader(TensorDataset(X_test), batch_size=64, shuffle=False)

#Create LSTM Autoencoder Model
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)#encode the input sequence and get the hidden state
        repeated = h.permute(1, 0, 2).repeat(1, x.size(1),1) #repeat the hidden state for each timestep in the sequence
        out, _ = self.decoder(repeated) #decode the repeated hidden state to reconstruct the input sequence
        return out
        
#Train the model
model = LSTMAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for (batch,) in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch) #Compute the reconstruction loss
        loss.backward() #Backpropagate the loss
        optimizer.step() #Update the model parameters
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.6f}")

#detect anomalies
model.eval()
with torch.no_grad():
    reconstructed = model(X_test)
    errors = ((X_test - reconstructed) ** 2).mean(dim=(1,2)).numpy()#Calculate the mean squared error for each sequence
threshold = np.percentile(errors, 95)
anomalies = errors > threshold #Identify sequences with reconstruction error above the threshold as anomalies

#Plot the reconstruction error
#1.plot figure size
plt.figure(figsize=(14, 4))
#2.set labels
plt.plot(errors, label='Reconstruction Error')
#3.plot threshold
plt.axhline(threshold, color='r', linestyle='--', label= 'Threshold(95th percentile)')
#Fills the area between the error line and the threshold line with red, but only where anomalies occur.
plt.fill_between(range(len(errors)), errors, threshold, where=anomalies, color='red', alpha=0.3, label='Anomalies' )
'''
plt.fill_between(x, y1, y2, where=...) — fills the space between two horizontal values at each x position.

range(len(errors)) — the x-axis

Just the window indices: 0, 1, 2, 3... up to the last window
Essentially the timeline


errors — the top boundary (y1)

The reconstruction error line
This is the "roof" of the filled area


threshold — the bottom boundary (y2)

The red dashed line
This is the "floor" of the filled area
'''
plt.legend()
plt.title('LSTM Autoencoder - Anomoly Detection on NASA SMAP P-1')
plt.xlabel('Timesteps')
plt.ylabel('Reconstruction Error')
plt.tight_layout()
plt.savefig('anomoly_results.png')
plt.show()
print(f"\nDone. Detected {anomalies.sum()} anomalies out of {len(anomalies)} windows.")