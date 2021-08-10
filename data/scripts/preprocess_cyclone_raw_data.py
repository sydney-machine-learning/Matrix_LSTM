import pandas as pd

#This piece of code sets the starting coordinates of all cyclones to be (0,0)

train_df = pd.read_csv(r'./data/rawtrain1985-2001.txt', delimiter = "\t", header = None, names =['id', 'date', 'longitude', 'latitude', 'speed']);
test_df = pd.read_csv(r'./data/rawtest2006-2013.txt', delimiter = "  ", header = None, names =['id', 'date', 'longitude', 'latitude', 'speed']);

id = train_df['id'][0]
x0 = train_df['longitude'][0]
y0 = train_df['latitude'][0]
train_df['longitude'][0] = 0
train_df['latitude'][0] = 0
for i in range(1, train_df.shape[0]):
  if train_df['id'][i] == id :
    train_df['longitude'][i] = train_df['longitude'][i] - x0
    train_df['latitude'][i] = train_df['latitude'][i] - y0
  else:
    x0 = train_df['longitude'][i]
    y0 = train_df['latitude'][i]
    train_df['longitude'][i] = 0
    train_df['latitude'][i] = 0
    id = train_df['id'][i]
    
id = test_df['id'][0]
x0 = test_df['longitude'][0]
y0 = test_df['latitude'][0]
test_df['longitude'][0] = 0
test_df['latitude'][0] = 0
for i in range(1, test_df.shape[0]):
  if test_df['id'][i] == id :
    test_df['longitude'][i] = test_df['longitude'][i] - x0
    test_df['latitude'][i] = test_df['latitude'][i] - y0
  else:
    x0 = test_df['longitude'][i]
    y0 = test_df['latitude'][i]
    test_df['longitude'][i] = 0
    test_df['latitude'][i] = 0
    id = test_df['id'][i]
    
train_df.to_csv('./data/south_indian_ocean_processedtrain1985-2001.txt', index=False, header=False)
test_df.to_csv('./data/south_indian_ocean_processedtest2006-2013.txt', index=False, header=False)