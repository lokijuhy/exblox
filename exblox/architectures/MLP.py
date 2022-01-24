from torch import nn


class MLP(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_size=24, dropout_p=0.1):
        super().__init__()

        self.config = {
            'input_layer_size': input_layer_size,
            'hidden_layer_size': hidden_layer_size,
            'dropout_p': dropout_p,
        }

        self.fc1 = nn.Linear(input_layer_size, hidden_layer_size)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_layer_size, int(hidden_layer_size/2))
        self.fc3 = nn.Linear(hidden_layer_size, 1)
        self.out = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc3(x)
        output = self.out(x)
        return output

    def get_params(self):
        return self.config
