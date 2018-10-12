#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#include "DL_core.hh"
#include "DL_util.hh"

#include "zip.hh"
#include "index_iterator.hh"
#include "sliding_window.hh"
#include "Adam.hh"

int main(int argc,char* argv[]){
  const auto Ndivision = ((argc > 1) ? std::strtoul(argv[1],nullptr,0): 100ul);

  constexpr const auto two_pi = std::acos(-1.0) * 2.0;

  std::vector<std::vector<double>> data{};
  std::vector<std::vector<double>> label{};

  data.reserve(Ndivision);
  label.reserve(Ndivision);

  std::generate_n(std::back_inserter(data),Ndivision,
		  [dx=two_pi/Ndivision,i=0ul]() mutable {
		    return std::vector{dx*(i++)};
		  });
  std::transform(data.begin(),data.end(),std::back_inserter(label),
		 [=](auto& v){
		   return std::vector{std::sin(two_pi * v[0])};
		 });


  const auto batch_size = 32ul;
  const auto epoch_size = 1000ul;
  const auto train_rate = 0.7;
  const auto L1 = 0.01;
  const auto L2 = 0.01;

  auto DL = ymd::InputLayer<double>{1}
  >> ymd::ReLU{100} >> ymd::ReLU{100}
  >> ymd::Identity_SquareError{1};

  auto [train_data,train_label,test_data,test_label]
    = ymd::split_train_test(data,label,train_rate);

  auto adam = DL.MakeAdaptive(ymd::Adam<double>{});
  for(auto epoch = 0ul; epoch < epoch_size; ++epoch){
    for(auto&& Batch:
	  ymd::zip(train_data,train_label) |
	  ymd::adaptor::shuffle_view{} |
	  ymd::adaptor::sliding_window{batch_size,batch_size}){
      for(auto&& DataLabel: Batch){ DL << DataLabel; }
      DL.Update(adam,L1,L2);

    }
    std::cout << "epoch: " << epoch
	      << " train loss: " << DL.Loss(train_data,train_label)
	      << " test loss: " << DL.Loss(test_data,test_label)
	      << std::endl;
  }

  return 0;
}
