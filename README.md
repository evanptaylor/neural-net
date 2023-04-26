# neural-net

### “What I cannot create, I do not understand” - Richard Feynman

neural-net.py is my implementation of a simple neural network in Python using NumPy. For now, it uses a custom dataset with three classes and trains the network using stochastic gradient descent. The current code uses 2 hidden layers, each containing 32 neurons. The hidden layers use a rectified linear activation function. The model achieves ~95% accuracy on this simple dataset.

This specific 'sprial' dataset is from nnfs.io--a good resource to learn more about neural networks

Below is a visual representation of the current implementation

<p align="center" text-align="center">
  <img src="https://user-images.githubusercontent.com/36122439/234676555-cee15993-a8d6-4d6e-b3ee-8b98079f25d4.png" width=80% height=60%>
</p>

#### Dataset visualized with matplotlib:

<img src="https://user-images.githubusercontent.com/36122439/234508366-74f5fcf1-9d98-4fee-aacb-b7ecdada94c1.png" width=50% height=50%>


#### Example output after 10,000 cycles (cycle number, accuracy, loss, learning rate):
<img src="https://user-images.githubusercontent.com/36122439/234675149-719dfeab-ed2b-461a-a3a6-1cf601d42e1e.png" width=50% height=50%>

#### Model's predictions visualized (where green is a correct prediction):
<img src="https://user-images.githubusercontent.com/36122439/234675335-8716ff46-e338-4bc4-9eb1-e106901cb8ae.png" width=50% height=50%>
 
