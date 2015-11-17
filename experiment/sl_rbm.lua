codeFolder = '../code/'

require('torch')
require(codeFolder..'rbm')
require(codeFolder..'dataset-mnist')
require(codeFolder..'rbm-visualisation.lua')
ProFi = require(codeFolder..'ProFi')
require('paths')
require('string')



function save_rbm_image(rbm, sample)
	if sample == 1 or sample % 100 == 0 then
		-- Output the rbm weights
		local image_name = string.format("%s/rbm.W.epoch_%02d.sample_%d.png",
			rbm.output_folder, rbm.currentepoch, sample)
		local geometry = {32,32}

		create_weight_image(rbm, geometry, image_name)
	end
end


-- create the options
if not opts then
	cmd = torch.CmdLine()
	cmd:option('-folder', 'output', 'the folder to be used for output')
	cmd:option('-n_hidden', 1024, 'number of hidden units')
	cmd:option('-datasetsize', 'full', 'small|full size of dataset')
	cmd:option('-image_name', 'demo.png', 'the filename with which the generated image should be saved')
	opts = cmd:parse(arg or {})
end


torch.setdefaulttensortype('torch.FloatTensor')


-- The supplied MNIST images are 32x32 pixels in size.
geometry = {32,32}

-- Only load the small dataset to start with.
if opts.datasetsize == 'full' then
    trainData,valData = mnist.loadTrainAndValSet(geometry,'none')
    testData = mnist.loadTestSet(nbTestingPatches, geometry)
else
	dataSize = 2000
	trainData = mnist.loadTrainSet(dataSize, geometry,'none')
	testData = mnist.loadTestSet(dataSize/2, geometry)
	valData = mnist.loadTestSet(dataSize/2, geometry)
end

-- The datasets need to be converted probabilities
trainData:toProbability()
valData:toProbability()
testData:toProbability()

-- Create the output folder if it does not exist
os.execute('mkdir -p ' .. opts.folder)

-- Create the rbm
rbm = rbmsetup(opts, trainData)

-- Set the output for the rbm
rbm.output = save_rbm_image

-- Train the rbm
rbm = rbmtrain(rbm, trainData, valData)




