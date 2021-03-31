import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import sys
from matplotlib import pyplot
#from torchvision import transforms


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

def visualiseModel(accuracy, loss):
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(loss, label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(accuracy, label='train')
    #pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()
    return

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, process=False):
        try:
            self.data = pd.read_csv(csv_file)
        except:
            self.data = pd.read_csv(root_dir + csv_file)

        #self.transform = transforms.Compose([transforms.ToTensor()])

        self.root_dir = root_dir
        self.process = process

        if process:
            self.processed_data = self.processData()

    def __len__(self):
        if self.process:
            return len(self.processed_data)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = idx.tolist()

        if self.process:
            x = np.array(self.processed_data.iloc[index][:13])
            y = np.array(self.processed_data.iloc[index][-1])
            return torch.from_numpy(x), torch.from_numpy(y)
        else:
            return self.data.iloc[index]

    def processData(self):
        data_to_process = self.data.copy()
        date = data_to_process.pop('Date')
        seasons = data_to_process.pop('Seasons')
        holiday = data_to_process.pop('Holiday')
        func_day = data_to_process.pop('Functioning Day')

        new_dates = []
        for row in date:
            row = row.replace("/","-")
            new_dates.append(row)

        new_seasons = []
        season_type = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Autumn': 4}
        for row in seasons:
            new_season = season_type[str(row)]
            new_seasons.append(new_season)

        new_holidays = []
        holiday_type = {'Holiday': 1, 'No Holiday': 2}
        for row in holiday:
            new_holiday = holiday_type[str(row)]
            new_holidays.append(new_holiday)

        new_func_days = []
        func_day_type = {'Yes': 1, 'No': 2}
        for row in func_day:
            new_day = func_day_type[str(row)]
            new_func_days.append(new_day)

        test = data_to_process.pop('Rented Bike Count')
        data_to_process.insert(0, 'Date', new_dates)
        data_to_process.insert(1, 'Seasons', new_seasons)
        data_to_process.insert(2, 'Holidays', new_holidays)
        data_to_process.insert(3, 'Functioning Day', new_func_days)
        data_to_process.insert(13, 'Rented Bike Count', test)

        data_to_process.columns = ['Date', 'Seasons', 'Holidays', 'FuncDays',
                                    'Hour', 'Temp', 'WindSpeed', 'Humidity',
                                    'Visibility', 'DewPoint', 'SolarRad',
                                    'Rainfall', 'Snowfall', 'Rented']

        numeric_attr = ['Temp', 'WindSpeed', 'Humidity', 'Visibility', 'DewPoint',
                        'SolarRad', 'Rainfall', 'Snowfall', 'Rented']
        for col in numeric_attr:
            data_to_process[col] = pd.to_numeric(data_to_process[col], errors='coerce')


        categorical_attr = ['Seasons', 'Holidays', 'FuncDays', 'Hour']
        for col in categorical_attr:
            data_to_process[col] = data_to_process[col].astype("category")

        data_to_process['Date'] = pd.to_datetime(data_to_process['Date'], errors='coerce')

        data_to_process.pop('Date')

        print (data_to_process)

        return data_to_process

    def returnDF(self):
        if self.process:
            return self.processed_data
        else:
            return self.data

class FeedForwardNN(nn.Module):
  def __init__(self, emb_dims, no_of_cont, lin_layer_sizes,
               output_size, emb_dropout, lin_layer_dropouts):
    super().__init__()

    # Embedding layers
    self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                     for x, y in emb_dims])

    no_of_embs = sum([y for x, y in emb_dims])
    self.no_of_embs = no_of_embs
    self.no_of_cont = no_of_cont

    # Linear Layers
    first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont,
                                lin_layer_sizes[0])

    self.lin_layers =\
     nn.ModuleList([first_lin_layer] +\
          [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
           for i in range(len(lin_layer_sizes) - 1)])

    for lin_layer in self.lin_layers:
      nn.init.kaiming_normal_(lin_layer.weight.data)

    # Output Layer
    self.output_layer = nn.Linear(lin_layer_sizes[-1],
                                  output_size)
    nn.init.kaiming_normal_(self.output_layer.weight.data)

    # Batch Norm Layers
    self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
    self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size)
                                    for size in lin_layer_sizes])

    # Dropout Layers
    self.emb_dropout_layer = nn.Dropout(emb_dropout)
    self.droput_layers = nn.ModuleList([nn.Dropout(size)
                                  for size in lin_layer_dropouts])

  def forward(self, cont_data, cat_data):

    if self.no_of_embs != 0:
      x = [emb_layer(cat_data[:, i])
           for i,emb_layer in enumerate(self.emb_layers)]
      x = torch.cat(x, 1)
      x = self.emb_dropout_layer(x)

    if self.no_of_cont != 0:
      normalized_cont_data = self.first_bn_layer(cont_data)

      if self.no_of_embs != 0:
        x = torch.cat([x, normalized_cont_data], 1)
      else:
        x = normalized_cont_data

    for lin_layer, dropout_layer, bn_layer in\
        zip(self.lin_layers, self.droput_layers, self.bn_layers):

      x = F.relu(lin_layer(x))
      x = bn_layer(x)
      x = dropout_layer(x)

    x = self.output_layer(x)

    return x


def model():
    if sys.platform == "linux" or sys.platform == "linux2":
        current_dir = os.path.dirname(__file__)
        fname = '/SeoulBikeData.csv'
    elif sys.platform == "win32":
        current_dir = os.path.dirname(__file__)
        fname = '\SeoulBikeData.csv'

    ds = Dataset(csv_file=fname, root_dir=current_dir, process=True)

    categorical_attr = ['Seasons', 'Holidays', 'FuncDays', 'Hour']
    output_attr = 'Rented'

    cat_dims = [int(ds[col].nunique()) for col in categorical_attr]
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

    model = FeedForwardNN(emb_dims, no_of_cont=8, lin_layer_sizes=[50, 100],
                          output_size=1, emb_dropout=0.04,
                          lin_layer_dropouts=[0.001,0.01]).to(device)

    # Hyperparam
    BATCH_SIZE = 64
    EPOCH = 10

    # Load Dataframe

    loader = torch.utils.data.DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Learning Rate
    learning_rate = 1e-3

    # OPT - model.parameters() tells RMSProp what tensors should be updated
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # MSE Error Loss
    loss_fn = torch.nn.MSELoss()

    accuracy = []
    loss_list = []

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            b_x = torch.autograd.Variable(batch_x)
            b_y = torch.autograd.Variable(batch_y)

            input = b_x.float()
            target = b_y.float()

            prediction = model(input)     # input x and predict based on x

            loss = loss_fn(prediction, target)     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

        threshold = 0.1
        errors = (prediction - target) ** 2  # Squared error
        acc = (errors < threshold).float().mean()
        error = errors.mean()
        print (acc)
        print (error)

    #visualiseModel(accuracy, loss)

if __name__ == '__main__':
    model()
