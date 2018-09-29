#ifndef YMD_DL_UTIL_HH
#define YMD_DL_UTIL_HH 1

#include <algorithm>
#include <tuple>
#include <type_traits>

#include "zip.hh"

namespace ymd {
  template<typename DL,typename Data,typename Label>
  inline auto error_rate(DL& dl,const Data& data,const Label& label){
    auto error = 0ul;
    auto norm = 1.0/(data.size());

    for(auto&& [d,l]: ymd::zip(data,label)){
      auto p = d >> dl;

      const auto predict_class = std::distance(p.begin(),
					       std::max_element(p.begin(),p.end()));
      const auto label_class   = std::distance(l.begin(),
					       std::max_element(l.begin(),l.end()));
      if(predict_class != label_class) { ++error; }
    }

    return error * norm;
  }

  template<typename Data,typename Label>
  inline auto split_train_test(const Data& data,
			       const Label& label,
			       double train_rate = 0.7){
    Data train_data{},test_data{};
    Label train_label{},test_label{};

    const auto data_size = data.size();
    const auto train_size = std::size_t(data.size() * train_rate);

    train_data.reserve(train_size);
    train_label.reserve(train_size);
    std::copy_n(data.begin(),train_size,std::back_inserter(train_data));
    std::copy_n(label.begin(),train_size,std::back_inserter(train_label));

    test_data.reserve(data_size-train_size);
    test_label.reserve(data_size-train_size);
    std::copy(data.begin()+train_size,data.end(),std::back_inserter(test_data));
    std::copy(label.begin()+train_size,label.end(),std::back_inserter(test_label));

    return std::make_tuple(train_data,train_label,test_data,test_label);
  }
} // namespace ymd
#endif // YMD_DL_UTIL
