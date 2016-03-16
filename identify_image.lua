require 'image'
require 'torch'
require 'cutorch'
require 'gnuplot'

local img_name = arg[1]
local img = -((image.scale(image.load(img_name,1),28,28))) + 1
img = img:cuda()
if arg[3] == "gnuplot" then
	gnuplot.imagesc(img,'grey');
end

weights = torch.load("weights.dat")

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

print (identify(img))
