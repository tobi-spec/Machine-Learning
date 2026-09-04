import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Runs on device:", device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 0.0003
PATCH_SIZE = 4
CHANNELS = 3
NUMBER_OF_CLASSES = 10
IMAGE_SIZE = 32
EMBEDDING_DIMENSION = 256
NUMBER_OF_HEADS = 8
DEPTH = 6
MLP_DIMENSION = 512
DROP_RATE = 0.1

class PatchEmbedding(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 in_channels,
                 embedding_dimensions):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels=in_channels,
                                    out_channels=embedding_dimensions,
                                    kernel_size=patch_size,
                                    stride=patch_size) # No overlap, just patches :)
        number_of_patches = (image_size // patch_size)**2
        # represents the image as a whole
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dimensions))
        # represents the position of the patch in the image
        self.positional_embedding = nn.Parameter(torch.randn(1, 1+number_of_patches, embedding_dimensions))

    # x: (32, 3, 224, 224)
    #     B   C   H    W
    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = self.projection(x) # before: (32, 3, 224, 224) -> after:  (32, 768,  14,  14)
        x = x.flatten(2).transpose(1, 2) # Grid into sequence (32, 768, 14, 14) -> (32, 196, 768)
        cls_token = self.cls_token.expand(batch_size, -1, -1) # Create a CLS token for every image
        x = torch.cat((cls_token, x), dim=1) # Put CLS in front of the patch sequence -> (32, 197, 768)
        x = x + self.positional_embedding
        return x

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 drop_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features,
                             out_features=in_features)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.drop_out(F.gelu(self.fc1(x)))
        x = self.drop_out(F.gelu(self.fc2(x)))
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size,
                 patch_size,
                 in_channels,
                 number_of_classes,
                 embedding_dimensions,
                 depth,
                 number_of_heads,
                 mlp_dimension,
                 drop_rate):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              in_channels=in_channels,
                                              embedding_dimensions=embedding_dimensions)
        layers = [TransformerEncoderLayer(embedding_dimensions, number_of_heads, mlp_dimension, drop_rate) for _ in range(depth)]
        self.encoder = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embedding_dimensions)
        self.head = nn.Linear(embedding_dimensions, number_of_classes)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer_function = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.005)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

    def backward(self, train_loader, epoch, num_epochs):
        self.train()
        cumulative_loss = 0

        for x_values, y_values in train_loader:
            images = x_values.to(device)
            labels = y_values.to(device)
            prediction = self.forward(images)
            loss = self.loss_function(prediction, labels)
            loss.backward()
            self.optimizer_function.step()
            self.optimizer_function.zero_grad()
            cumulative_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {cumulative_loss / len(train_loader):.4f}")



all_transforms = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=all_transforms, download=False)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=all_transforms, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = VisionTransformer(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_channels=CHANNELS, number_of_classes=NUMBER_OF_CLASSES,
                          embedding_dimensions=EMBEDDING_DIMENSION, depth=DEPTH, number_of_heads=NUMBER_OF_HEADS,
                          mlp_dimension=MLP_DIMENSION, drop_rate=DROP_RATE).to(device)

for epoch in range(EPOCHS):
    model.backward(train_loader, epoch, EPOCHS)


with torch.no_grad():
    predictions = []
    targets = []
    for x_values, y_values in test_loader:
        images = x_values.to(device)
        labels = y_values.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        targets.extend(labels.tolist())

    accuracy = MulticlassAccuracy(num_classes=10)
    precision = MulticlassPrecision(num_classes=10, average=None)
    recall = MulticlassRecall(num_classes=10, average=None)
    f1 = MulticlassF1Score(num_classes=10, average=None)
    print("Accuracy:", accuracy(torch.tensor(predictions), torch.tensor(targets)))
    print("Precision:", precision(torch.tensor(predictions), torch.tensor(targets)))
    print("Recall:", recall(torch.tensor(predictions), torch.tensor(targets)))
    print("F1:", f1(torch.tensor(predictions), torch.tensor(targets)))

with open("results.txt", 'a' ) as file:
    file.write("\nNext Run")
    file.write("\nAccuracy:" + str(accuracy(torch.tensor(predictions), torch.tensor(targets))))
    file.write("\nPrecision:" + str(precision(torch.tensor(predictions), torch.tensor(targets))))
    file.write("\nRecall:" + str(recall(torch.tensor(predictions), torch.tensor(targets))))
    file.write("\nF1:" + str(f1(torch.tensor(predictions), torch.tensor(targets))))
    file.write("\n")
