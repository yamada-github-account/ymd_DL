#include <iostream>

#include "MNIST.hh"
#include "DL_core.hh"
#include "DL_util.hh"

int main(int argc,char* argv[]){
  // Arguments:
  // ----------
  // argc[1]: MNIST directory
  // argc[2]: Size to read MNIST data for training

  auto MNIST_DIR = std::string{(argc > 1) ? argv[1]: "."};
  const auto read_size = ((argc > 2) ? std::strtoul(argv[2],nullptr,0): 0ul);

  auto train_data = ymd::read_MNIST(MNIST_DIR + "/train-images-idx3-ubyte",read_size);
  auto train_label= ymd::read_MNIST(MNIST_DIR + "/train-labels-idx1-ubyte",read_size);
  auto test_data  = ymd::read_MNIST(MNIST_DIR + "/t10k-images-idx3-ubyte" ,read_size);
  auto test_label = ymd::read_MNIST(MNIST_DIR + "/t10k-labels-idx1-ubyte" ,read_size);

  // Create Network
  auto digit = ymd::InputLayer<double>{28*28} >> ymd::SoftMax_CrossEntropy{10};

  auto eps = 0.01;
  auto L1 = 0.0;
  auto L2 = 0.0;

  for(auto n = 0ul; n < 100; ++n){
    // Stochastic Gradient Descent
    for(auto&& [D,L]: ymd::zip(train_data,train_label)){
      digit << std::make_tuple(D,L); // Back Propagation
      digit.Update(eps/(n+1),L1,L2);
    }

    std::cout << n << "th train loss: " << digit.Loss(train_data,train_label)
	      << " test loss: " << digit.Loss(test_data,test_label)
	      << " train error: " << ymd::error_rate(digit,train_data,train_label)
	      << " test error: " << ymd::error_rate(digit,test_data,test_label)
	      << std::endl;
  }

  return 0;
}
