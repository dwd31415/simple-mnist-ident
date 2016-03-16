require 'torch'
require 'cutorch'
local mnist = require 'mnist'

local train_data = mnist.traindataset()
local train_data_gpu = {}

-- number of samples to load
N = 5000

for i = 1,N do
	train_data_gpu[i] = {train_data[i].y,train_data[i].x:cuda()}
end

print ('=> Loaded MNIST Database to GPU')

return train_data_gpu
