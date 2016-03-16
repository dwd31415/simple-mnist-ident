require 'torch'
require 'cutorch'
local samples = require 'load_mnist_cuda.lua'

-- the learning rate
local learning_rate = 0.01
-- the number of epochs to train for
local epochs = 10
-- weights decay
local decay = 0.1

local weights = {}
for i = 1,10 do
	weights[i] = torch.rand(28,28):cuda()
end
print ("=> Initialized weights")

local memory = {}
local kernel = function (x) return torch.exp(-torch.pow(x,2)) end
local inverse_kernel = function (x) return -torch.exp(-torch.pow(x,2))+1 end

for e = 1,epochs do
	for i=1,(N*0.5) do
		local target = samples[i][1]
		local input = samples[i][2]
		if memory[target] == nil then
			memory[target] = input
		else
			weights[target+1] = torch.mul(inverse_kernel(torch.mul(input-memory[target],1)),learning_rate) + weights[target+1]
			-- in the second epoch all classes should have been shown
			if e > 1 then
				weights[target+1] = -torch.mul(kernel(torch.mul(input-memory[math.random(0,9)],1)),learning_rate) + weights[target+1]
			end			
			memory[target] = input
		end
	end
	print ("=> Trained for one epoch")
end

torch.save('weights.dat',weights)
print("=> Saved weights to disk")

function identify(image)
	local highest_score = -1
	local ident = -1
	for i = 1,10 do
		local res = torch.sum(torch.cmul(image,weights[i]))
		if res > highest_score then
			highest_score = res
			ident = i-1 	
		end
	end
	return ident
end

print("=> Start testing")
local accuracy = 0
local images = 0
for i=(N*0.5),N do
	images = images + 1
	local target = samples[i][1]
	local input = samples[i][2]
	if target == identify(input) then
		accuracy = accuracy + 1 	
	end
end
print ("=> Finished testing with " .. (100 * accuracy/images) .. "% accuracy")
