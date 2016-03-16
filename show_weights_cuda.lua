require 'gnuplot'
require 'cutorch'
require 'torch'
require 'io'

w = torch.load("weights.dat")

for i = 1,10 do
	gnuplot.imagesc(w[i])
	io.read()
end
