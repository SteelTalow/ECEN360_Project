print("Starting Neural Net Training Python File")
print("Double Single Output Neural Nets Version")
print("Importing Modules (This may take a while)")

#Time for imports
import numpy as np, pandas as pd
#from matplotlib.pyplot import subplots

print("Import sklearn stuff")
from sklearn.linear_model import \
(LinearRegression,LogisticRegression,Lasso)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

print("Import ISLP stuff")
from ISLP import load_data
from ISLP.models import ModelSpec as MS

from sklearn.model_selection import \
(train_test_split, GridSearchCV)

print("Import torch stuff")
#Pytorch stuff
import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset

print("Import auxilary torch stuff")
from torchmetrics import (MeanAbsoluteError, R2Score)
from torchinfo import summary
from torchvision.io import read_image

print("Import lightning stuff")
#Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

print("importing extra stuff")
#from torchvision.datasets Don't need these data sets I don't think
from torchvision.models import (resnet50,ResNet50_Weights)
from torchvision.transforms import (Resize, Normalize, CenterCrop, ToTensor)

from ISLP.torch import (SimpleDataModule, SimpleModule, ErrorTracker, rec_num_workers)
from ISLP.torch.imdb import (load_lookup, load_tensor, load_sparse, load_sequential)

from torch.utils.data import DataLoader, TensorDataset

#Encoders, we are swapping from label encoder to ordinal encoder now
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

print("Imports Successful")

print("Now managing the data")

columns_of_interest = ["camis","dba","cuisine_description","boro","score","violation_code"]

df = pd.read_csv(r"example.csv")

print("Columns in file: (A Subset of which will be used)")
print(df.columns)


#Columns of interest
df = df[['camis','dba','cuisine_description','boro','score','violation_code']]
#We need to encode the violation_code
violation_le = LabelEncoder()
df["violation_code"] = df["violation_code"].astype(str)
df["violation_code"] = violation_le.fit_transform(df['violation_code'])

#Predictors
X = ["camis","dba","cuisine_description","boro"]

df['camis'] = df['camis'].fillna("unknown")
df['dba'] = df['dba'].fillna("unknown")
df['cuisine_description'] = df['cuisine_description'].fillna("unknown")
df['boro'] = df['boro'].fillna("unknown")
df['score'] = df['score'].fillna(0)
df['violation_code'] = df['violation_code'].fillna("unknown")

#Train test split it
df_train, df_test = train_test_split(df, test_size=0.2, random_state=298845)

print("First 5 samples in the test set scores:")
for i, row in df_test.head(5).iterrows():
    print(f"{i}: DBA = {row['dba']}, Score = {row['score']}")

#Function to encode the categorical variables
def encode_categorical(df_subset, ordinal_encoder = None, fit = True):
    if fit:
        ordinal_encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
        encoded_array = ordinal_encoder.fit_transform(df_subset[X].astype(str))

    else:
        encoded_array = ordinal_encoder.transform(df_subset[X].astype(str))

    #Reserving 0 as unknowns. We 
    encoded_array = encoded_array + 1

    x_cat = torch.tensor(encoded_array, dtype=torch.long)
    return x_cat, ordinal_encoder

print("Running Encode on Predictors")
x_cat_train, ordinal_encoder = encode_categorical(df_train, fit=True)
x_cat_test, _ = encode_categorical(df_test, ordinal_encoder, fit=False)

print("Running Encode for Outcomes")
# Regression target
y_score_train = torch.tensor(df_train["score"].values, dtype=torch.float32).unsqueeze(1)
y_score_test = torch.tensor(df_test["score"].values, dtype=torch.float32).unsqueeze(1)

# Classification target
y_violation_train = torch.tensor(df_train["violation_code"].values, dtype=torch.long)
y_violation_test = torch.tensor(df_test["violation_code"].values, dtype=torch.long)


print(f"x_cat_train shape: {x_cat_train.shape}")
print(f"y_score_train shape: {y_score_train.shape}")
print(f"y_score_test shape: {y_score_test.shape}")
print(f"y_violation_train shape: {y_violation_train.shape}")
print(f"y_violation_test shape: {y_violation_test.shape}")
#We are going to leave these two seperate and not combine them.

print("Putting the data into tensor dataset format")
train_dataset_SCORE = TensorDataset(x_cat_train, y_score_train)
test_dataset_SCORE = TensorDataset(x_cat_test, y_score_test)

train_dataset_VIO = TensorDataset(x_cat_train, y_violation_train)
test_dataset_VIO = TensorDataset(x_cat_test, y_violation_test)

print("Working on Embedding Sizes")

embedding_sizes = []

#Get the embedding sizes
for i, col in enumerate(X):
    num_categories = len(ordinal_encoder.categories_[i])
    embedding_dim = min (50, (num_categories + 1)//2)
    embedding_sizes.append((num_categories + 1, embedding_dim))

print("Embedding Sizes:")
print(embedding_sizes)

print("Creating the two data modules using SimpleDataModule")

score_dm = SimpleDataModule(
    train_dataset_SCORE,
    test_dataset_SCORE,
    batch_size = 32,
    num_workers = min(5, rec_num_workers()),
    validation = 0.2
)

vio_dm = SimpleDataModule(
    train_dataset_VIO,
    test_dataset_VIO,
    batch_size = 32,
    num_workers = min(5, rec_num_workers()),
    validation = 0.2
)

print("Now defining the model")

class RestModel(nn.Module):

    def __init__(self, embedding_sizes, num_violation_classes):
        #Don't actually know what this line here does
        super(RestModel, self).__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim)
            for num_categories, emb_dim in embedding_sizes
        ])

        #Dropout layer?
        self.embedding_dropout = nn.Dropout(0.2)

        total_embedding_dim = sum(e[1] for e in embedding_sizes)

        # Shared layers
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        #-1 is the magic number to say we are doing score, a num_violation_classes above 0 is code for we are doing the violation version
        #Self.GO is different depending on if we are doing regression for out output or classification with the violation codes.
        if num_violation_classes == -1:
            self.GO = nn.Linear(16, 1)
        else:
            self.GO = nn.Linear(16, num_violation_classes)

    def forward(self, x_cat):

        #Do the embedding
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]

        x = torch.cat(embedded, dim=1)
        x = self.embedding_dropout(x)

        out = self.shared(x)

        output = self.GO(out)

        return output

print("Instantiating the model score model")

model_SCORE = RestModel(embedding_sizes, -1)

SCORE_module = SimpleModule.regression(model_SCORE)
SCORE_logger = CSVLogger('logs', name='SCORE')

print("Defining the trainer")

SCORE_trainer = Trainer(
    deterministic = True,
    max_epochs = 30,
    logger = SCORE_logger,
    callbacks = [ErrorTracker()]
)


if __name__ == "__main__":

    SCORE_trainer.fit(SCORE_module, datamodule = score_dm)

    print("Model training for score successful.")

    print("Testing one sample:")

    sample_input, true_score = test_dataset_SCORE[0]
    sample_input = sample_input.unsqueeze(0)

    print("True score:", true_score.item())

    sample_dataset = TensorDataset(sample_input, torch.tensor([[0.0]]))
    sample_loader = DataLoader(sample_dataset, batch_size=1)

    prediction = SCORE_trainer.predict(SCORE_module, dataloaders=sample_loader)

    true_y, y_hat = prediction[0]

    print("Predicted score:", y_hat.item())
    print("True score:", true_score.item())
    #Now we should make some output/graph of our model's predictions

    print("Now testing multiple samples")

    n = 5

    samples = [test_dataset_SCORE[i] for i in range(n)]

    sample_inputs, true_scores = zip(*samples)

    sample_inputs_tensor = torch.stack(sample_inputs)
    true_scores_tensor = torch.stack(true_scores)

    sample_dataset = TensorDataset(sample_inputs_tensor, true_scores_tensor)
    sample_loader = DataLoader(sample_dataset, batch_size=n)

    predictions = SCORE_trainer.predict(SCORE_module, dataloaders=sample_loader)

    #Outputs

    true_y, y_hat = predictions[0]

    for i in range(n):
        print(f"Sample {i + 1}:")
        print("  True Score     :", true_y[i].item())
        print("  Predicted Score:", y_hat[i].item())
