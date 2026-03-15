import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. Configuration
CSV_FILE = 'full_triplet_data_transfer.csv'
INPUT_DIM = 12  # Number of system metrics (CPU, Memory, IOPS, etc.)

class Config:
    def __init__(self, embedding_dim, batch_size, learning_rate, epochs, margin, input_dim=INPUT_DIM):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.margin = margin
        self.input_dim = input_dim
        

# 2. Dataset Definition
class TripletDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values.astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # The CSV is structured as: [anchor(12), pos(12), neg(12)]
        row = self.data[idx]
        anchor = row[0:12]
        pos = row[12:24]
        neg = row[24:36]
        return torch.tensor(anchor), torch.tensor(pos), torch.tensor(neg)

# 3. Model Architecture (Embedding Network)
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, x):
        # We normalize the output to the unit sphere for more stable triplet loss
        output = self.net(x)
        return nn.functional.normalize(output, p=2, dim=1)
    

def train_model_function(df, config, validation=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if validation:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_loader = DataLoader(TripletDataset(train_df), batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(TripletDataset(val_df), batch_size=config.batch_size)
    else:
        train_loader = DataLoader(TripletDataset(df), batch_size=config.batch_size, shuffle=True)
    model = EmbeddingNet(config.input_dim, config.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.TripletMarginLoss(margin=config.margin, p=2)

    # 5. Training Loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for anchor, pos, neg in train_loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            
            optimizer.zero_grad()
            
            # Get embeddings
            emb_a = model(anchor)
            emb_p = model(pos)
            emb_n = model(neg)
            
            # Calculate loss
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if validation:
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for anchor, pos, neg in val_loader:
                    anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
                    emb_a, emb_p, emb_n = model(anchor), model(pos), model(neg)
                    val_loss += criterion(emb_a, emb_p, emb_n).item()

            print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        else:
            print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {total_loss/len(train_loader):.4f}")
    
    return model

    
def main(validation=False):
    # Load and split data
    df = pd.read_csv(CSV_FILE)
    df = df.drop(columns=['anchor_id', 'pos_id', 'neg_id'])
    config = Config(
        embedding_dim=16,
        batch_size=64,
        learning_rate=0.001,
        epochs=50,
        margin=1.0
    )
    
    model = train_model_function(df, config, validation=validation)
    
    # 6. Save the trained embedding model
    torch.save(model.state_dict(), 'context_model.pth')
    print("Model saved to context_model.pth")


if __name__ == "__main__":
    main()
    