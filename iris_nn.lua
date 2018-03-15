require 'torch'
require 'nn'
function read_data(f)
	local dataset = {}
	while true do
		line = f:read()
		if line == nil then break end
	--	if i%2 == 0 then print("yes   ",line) end
		local a1, a2, a3, a4, label =
line:match("([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)") 
		local x = torch.Tensor(4)	
		x[1] = a1; x[2] = a2; x[3] = a3; x[4] = a4;
		local y = torch.Tensor(1)
		y[1] = label
		dataset[i] = {x,y}
		i = i+1
		--print (line)
		--print (line,inp) --a1,a2,a3,a4,label)
	end
	
	function dataset:size() return (i - 1) end
	return dataset

end

-- read()

-- USER INPUT INFO-----------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('th hello.lua --inputsize 50 --classes 5')
cmd:text('Options:')
cmd:option('-trainfile', 'hw1-train-split.txt', 'Training File')
cmd:option('-testfile', 'hw1-test-split.txt', 'Test File')
cmd:option('-inputsize', 4, 'Number of Inputs')
cmd:option('-hiddensize', 40, 'Number of Hidden units')
cmd:option('-classes', 3, 'number of Outputs')
cmd:option('-maxIter', 30, 'number of Iterations')
cmd:text()
opt = cmd:parse(arg or {})

-- INITIALIZING---------------------
trainfile = opt.trainfile
testfile = opt.testfile
inputSize = opt.inputsize
hiddenLayer1Size = opt.hiddensize
nclasses = opt.classes
iter = opt.maxIter
--LOADING INPUT DATA-----------------
i = 0
fh = io.open(trainfile)
train_dataset = read_data(fh)

-- PRINT INFO---------------------
print('\nThis NN classifies iris data into 3 different classes.')
print('\nThese are the user-specified options and hyper-parameters:')
print('\t Training File: \t', trainfile)
print('\t Test File: \t', testfile)
print('\t Number of Inputs: \t',inputSize)
print('\t Number of Outputs: \t', nclasses)
print('\t Number of Hunits: \t', hiddenLayer1Size)
print('\t Learning rate: \t 0.01')
print('\t Max Iterations: \t',iter)
print('\nThis is what the dataset looks like:')
print('\t trainSize:', #train_dataset+1)
print('\t testSize: ', 15)


-- NEURAL NETWORK-------------------
print('\nThis is what the NN looks like:')
mlp = nn.Sequential()
mlp:add(nn.Linear(inputSize, hiddenLayer1Size))
mlp:add(nn.Sigmoid())
--mlp:add(nn.Linear(hiddenLayer1Size, hiddenLayer2Size))
--mlp:add(nn.Sigmoid())

mlp:add(nn.Linear(hiddenLayer1Size, nclasses))
mlp:add(nn.LogSoftMax())

print (mlp)


--TRAINING----------------------------
print('\nTraining the network.')
criterion = nn.ClassNLLCriterion()
criterion.sizeAverage = false
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = iter
trainer:train(train_dataset)


--LOADING TEST DATA--------------------
i=0
th = io.open("hw1-test-split.txt")
test_dataset = read_data(th)

--TESTING--------------------------

function argmax(v)
	local maxvalue = torch.max(v)
	for i=1,v:size(1) do
		if v[i] == maxvalue then
			return i
		end
	end
end

tot =0
pos =0

print('\nTesting the network.')
for i=0, #test_dataset do
--	print (test_dataset[i][1])
	result = argmax(mlp:forward(test_dataset[i][1]))
	actual_value = test_dataset[i][2][1]

	if result == actual_value then
		pos=pos+1
	end
	tot=tot+1
	print('Predicted: '..result..' Desired: '..actual_value..' Correct: '..pos..'/'..tot)
end
print ('Percent correct during test: '..pos/tot)

torch.save('hw1.torchModel',mlp)




