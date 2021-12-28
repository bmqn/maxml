# About

Machine learning library written in pure C++.

I decided to write this library as a way to exmplore machine learning and improve my C++ knowledge + skills.
I am particularly interested in image processing with convolutional neural networks (CNNs).

# Example

Say we want to fit a sequential neural network model to the following function

![Function](https://latex.codecogs.com/png.image?\dpi{300}&space;\bg_white&space;y=2^{\sin(5x^3)}-x^2)

First we create the description of the desired sequential network to create a model

```C++
maxml::InputLayerDesc inpLayerDesc;
inpLayerDesc.Channels = 1;
inpLayerDesc.Rows = 1;
inpLayerDesc.Cols = 1;

maxml::FullyConnectedLayerDesc fc1LayerDesc;
fc1LayerDesc.NumOutputs = 16;
fc1LayerDesc.ActivFunc = maxml::ActivationFunc::Tanh;

maxml::FullyConnectedLayerDesc fc2LayerDesc;
fc2LayerDesc.NumOutputs = 8;
fc2LayerDesc.ActivFunc = maxml::ActivationFunc::Tanh;

maxml::FullyConnectedLayerDesc fc3LayerDesc;
fc3LayerDesc.NumOutputs = 1;
fc3LayerDesc.ActivFunc = maxml::ActivationFunc::None;

maxml::SequentialDesc seqDesc;
seqDesc.ObjectiveFunc = maxml::LossFunc::MSE;
seqDesc.LearningRate = 0.1;
seqDesc.LayerDescs = { inpLayerDesc, fc1LayerDesc, fc2LayerDesc, fc3LayerDesc };

maxml::Sequential seq(seqDesc);
```

Next we train the model

```C++
for (int i = 0; i < numIterations; i++)
{
  	int choice = rand() % data.size();

	const maxml::DTensor& inp = data[choice].first;
	const maxml::DTensor& exp = data[choice].second;

	const maxml::DTensor& out = seq.feedForward(inp);
	double err = seq.feedBackward(exp);
}
```

After 500,000 iterations we arrive at the following fit

![Regression](/regression.png)
