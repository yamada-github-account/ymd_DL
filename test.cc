#include <iostream>
#include <random>
#include <cmath>

#include "MNIST.hh"
#include "DL_core.hh"

int main(){

  auto DL =ymd::InputLayer<double>{5} >> ymd::ReLU{3} >> ymd::Identity_SquareError{5};

  DL.DebugShow();

  DL.Apply([g=std::mt19937{std::random_device{}()},
	    d=std::normal_distribution{}](auto& v) mutable { v = d(g); });

  DL.DebugShow();

  for(auto&& v: std::vector<double>{1,2,3,4,5} >> DL){
    std::cout << v << std::endl;
  }


  DL.DebugShow();


  std::cout << "Check calculation" << std::endl;
  auto d = ymd::InputLayer<double>{1} >> ymd::ReLU{2} >> ymd::Identity_SquareError{1};
  d.DebugShow();

  d.Apply([g=std::mt19937{std::random_device{}()},
	   d=std::normal_distribution{}](auto& v) mutable { v = d(g); });

  d.DebugShow();

  for(auto&& v: std::vector<double>{1} >> d){
    std::cout << v << std::endl;
  }
  d.DebugShow();

  auto error_rate = [](auto& dl,const auto& data,const auto& label){
		      auto error = 0ul;
		      for(auto&& [d,l]: ymd::zip(data,label)){
			auto p = d >> dl;
			if(std::distance(p.begin(),
					 std::max_element(p.begin(),p.end())) !=
			   std::distance(l.begin(),
					 std::max_element(l.begin(),l.end()))){
			  ++error;
			}
		      }
		      return error/(1.0*data.size());
		    };


  auto simple = ymd::InputLayer<double>{3} >> ymd::ReLU{10}
  >> ymd::SoftMax_CrossEntropy{3};

  simple.Apply([g=std::mt19937{std::random_device{}()},
		d=std::normal_distribution{}](auto& v) mutable { v = d(g); });


  auto simple_data = std::vector<std::vector<double>>{{1.0,0.0,0.0},
						      {0.0,1.0,0.0},
						      {0.0,0.0,1.0}};
  auto simple_test = std::vector<std::vector<double>>{{0.0,0.0,1.0},
						      {0.0,1.0,0.0},
						      {1.0,0.0,0.0}};



  for(auto n = 0ul; n < 10000ul; ++n){
    for(auto&& [D,L]: ymd::zip(simple_data,simple_data)){
      simple << std::make_tuple(D,L);
    }
    simple.Update(0.1/(n+1),0.0,0.0);

    auto loss = 0.0;
    for(auto&& [D,L]: ymd::zip(simple_data,simple_data)){
      loss += simple.Loss(D,L);
    }

    std::cout << n << "th loss: " << loss
	      << " train error: " << error_rate(simple,simple_data,simple_data)
	      << " test error: " << error_rate(simple,simple_test,simple_test)
	      << std::endl;
  }

  std::exit(1);


  constexpr const auto read_size = 1000;

  auto data = ymd::read_MNIST("/Users/yamada/ymd_data/MNIST/train-images-idx3-ubyte",
			      read_size);
  auto label= ymd::read_MNIST("/Users/yamada/ymd_data/MNIST/train-labels-idx1-ubyte",
			      read_size);

  auto data_size = data.size();

  auto train_size = std::size_t(data_size * 0.7);

  decltype(data) train_data{},test_data{};
  decltype(label) train_label{},test_label{};

  train_data.reserve(train_size);
  train_label.reserve(train_size);

  std::copy_n(data.begin(),train_size,std::back_inserter(train_data));
  std::copy_n(label.begin(),train_size,std::back_inserter(train_label));

  test_data.reserve(data_size-train_size);
  test_label.reserve(data_size-train_size);
  std::copy(data.begin()+train_size,data.end(),std::back_inserter(test_data));
  std::copy(label.begin()+train_size,label.end(),std::back_inserter(test_label));

  auto digit = ymd::InputLayer<double>{28*28} >> ymd::ReLU{100} >> ymd::ReLU{50}
  >> ymd::SoftMax_CrossEntropy{10};

  digit.Apply([g=std::mt19937{std::random_device{}()},
	       d=std::normal_distribution{0.0,std::sqrt(1.0/700)}]
	      (auto& v) mutable { v = d(g); });

  auto eps = 0.01;
  auto L1 = 0.0;
  auto L2 = 0.0;

  for(auto n = 0ul; n < 100; ++n){
    for(auto&& [D,L]: ymd::zip(train_data,train_label)){
      digit << std::make_tuple(D,L);
      digit.Update(eps/(n+1),L1,L2);
      digit.Apply([](auto& v){
		    if(std::isnan(v) || std::isinf(v)){ v = 0.0; }
		  });
    }


    auto loss = 0.0;
    for(auto&& [D,L]: ymd::zip(train_data,train_label)){
      auto tmp = digit.Loss(D,L);
      if(std::isnan(tmp) || std::isinf(tmp)){
	for(auto& d: D >> digit){ std::cout << d << " "; }
	std::cout << std::endl;
	for(auto& l: L){ std::cout << l << " "; }
	std::cout << std::endl;
	digit.DebugShow();
	std::cout << "n = " << n << std::endl;
	std::exit(1);
      }
      loss += tmp;
    }
    std::cout << n << "th loss: " << loss
	      << " train error: " << error_rate(digit,train_data,train_label)
	      << " test error: " << error_rate(digit,test_data,test_label)
	      << std::endl;
  }

  return 0;
}
