# Maxml

Machine learning library written in pure C++.

# About

I decided to write this library as a way to exmplore machine learning and improve my C++ knowledge + skills.
I am particularly interested in image processing with convolutional neural networks (CNNs).

# Example

Say we want to fit a sequential neural network model to the following function

![Function](https://latex.codecogs.com/png.image?\dpi{300}&space;\bg_white&space;y=2^{\sin(5x^3)}-x^2)

First we create the description of the desired sequential network to create a model

```C++
maxml::SequentialDesc seqDesc;
seqDesc.ObjectiveFunc = maxml::LossFunc::MSE;
seqDesc.LearningRate = 0.1f;
seqDesc.LayerDescs = {
	maxml::makeInput(1, 1, 1),
	maxml::makeFullCon(16, maxml::ActivationFunc::Tanh),
	maxml::makeFullCon(8, maxml::ActivationFunc::Tanh),
	maxml::makeFullCon(1, maxml::ActivationFunc::None)
};

maxml::Sequential seq(seqDesc);
```

Next we train the model

```C++
for (int i = 0; i < kNumIterations; i++)
{
  	int choice = rand() % data.size();

	const maxml::DTensor &inp = data[choice].first;
	const maxml::DTensor &exp = data[choice].second;

	const maxml::DTensor &out = seq.feedForward(inp);
	double err = seq.feedBackward(exp);
}
```

After 500,000 iterations we arrive at the following fit

![Regression](/regression.png)
