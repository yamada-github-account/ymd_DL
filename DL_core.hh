#ifndef YMD_DL_CORE_HH
#define YMD_DL_CORE_HH 1

#include <vector>
#include <cmath>
#include <type_traits>
#include <algorithm>
#include <functional>
#include <utility>
#include <iterator>
#include <tuple>
#include <limits>

#include "zip.hh"

namespace ymd {

  template<typename ValueType>
  struct Layer {
    using value_type = ValueType;
    using layer_type = std::vector<value_type>;
    using function_type = std::function<layer_type(layer_type)>;

    std::size_t size;
    layer_type value;
    function_type activate;
    function_type differentiate;

    Layer() = default;
    Layer(std::size_t size,function_type&& activate,function_type&& differentiate)
      : size(size),value(size),activate{activate},differentiate{differentiate} {}
    Layer(const Layer&) = default;
    Layer(Layer&&) = default;
    Layer& operator=(const Layer&) = default;
    Layer& operator=(Layer&&) = default;
    ~Layer() = default;

    friend inline layer_type operator>>(layer_type u,Layer<ValueType>& l){
      l.value = l.activate(u);
      return l.value;
    }

    layer_type operator<<(layer_type df_dz){
      for(auto&& [delta_z,dz_du] : ymd::zip(df_dz,differentiate(value))){
	delta_z *= dz_du;
      }
      return df_dz;
    }
    friend inline std::ostream& operator<<(std::ostream& os,
					   const Layer<ValueType>& l){
      os << "layer z\n";
      for(auto& ze : l.value){ os << ze << " "; }
      os << "\n";
      return os;
    }
  };


  template<typename ValueType> class WeightBias {
  public:
    using value_type = ValueType;
    using layer_type = std::vector<value_type>;

  private:
    std::size_t prev_size;
    std::size_t next_size;
    layer_type value;
    layer_type diff_value;

  public:
    WeightBias() = default;
    WeightBias(std::size_t prev_size,std::size_t next_size)
      : prev_size(prev_size), next_size(next_size),
	value((prev_size+1)*next_size),
	diff_value((prev_size+1)*next_size){}
    WeightBias(const WeightBias&) = default;
    WeightBias(WeightBias&&) = default;
    WeightBias& operator=(const WeightBias&) = default;
    WeightBias& operator=(WeightBias&&) = default;
    ~WeightBias() = default;

    inline auto Windex(std::size_t prev_i,std::size_t next_i) const {
      return next_size * prev_i + next_i;
    }
    inline auto Bindex(std::size_t next_i) const {
      return next_size*prev_size + next_i;
    }
    inline auto Wbegin(){ return value.begin(); }
    inline auto Wend  (){ return value.end() - next_size; }
    inline auto Bbegin(){ return value.end() - next_size; }
    inline auto Bend  (){ return value.end(); }

    template<typename F> inline auto for_each_prev_next(F&& f){
      for(auto next_i = 0ul; next_i < next_size; ++next_i)
	{ for(auto prev_i = 0ul; prev_i < prev_size; ++prev_i){ f(prev_i,next_i); } }
    }

    template<typename F> inline auto for_each_next(F&& f){
      for(auto next_i = 0ul; next_i < next_size; ++next_i){ f(next_i); }
    }

    auto operator*(const layer_type& prev){
      layer_type next(Bbegin(),Bend());

      for_each_prev_next([&](auto prev_i,auto next_i){
			   next[next_i] += prev[prev_i]*value[Windex(prev_i,next_i)];
			 });
      return next;
    }

    auto BackPropagate(const layer_type& prev, const layer_type& df_du){
      for_each_next([&](auto next_i){ diff_value[Bindex(next_i)] += df_du[next_i]; });

      for_each_prev_next([&](auto prev_i,auto next_i)
		{ diff_value[Windex(prev_i,next_i)] += prev[prev_i]*df_du[next_i]; });

      std::vector<value_type> df_dprev_z(prev_size,value_type{0});
      for_each_prev_next([&](auto prev_i,auto next_i)
		{ df_dprev_z[prev_i] += df_du[next_i]*value[Windex(prev_i,next_i)];});

      return df_dprev_z;
    }

    auto Update(value_type eps,value_type L1,value_type L2){
      for(auto&& [wb,diff_wb]: ymd::zip(value,diff_value)){
	wb -= eps*(std::exchange(diff_wb,value_type{0}) +
		   std::copysign(L1,wb) + L2*wb);
      }
    }

    friend inline std::ostream& operator<<(std::ostream& os,
					   const WeightBias<ValueType>& wb){
      os << "weight and bias\n";
      for(auto next_i = 0ul; next_i < wb.next_size; ++next_i){
	for(auto prev_i = 0ul; prev_i < wb.prev_size; ++prev_i){
	  os << wb.value[wb.Windex(prev_i,next_i)] << " ";
	}
	os << "; " << wb.value[wb.Bindex(next_i)] << "\n";
      }
      os << "\n";

      os << "diff_value\n";
      for(auto next_i = 0ul; next_i < wb.next_size; ++next_i){
	for(auto prev_i = 0ul; prev_i < wb.prev_size; ++prev_i){
	  os << wb.diff_value[wb.Windex(prev_i,next_i)] << " ";
	}
	os << "; " << wb.diff_value[wb.Bindex(next_i)] << "\n";
      }
      os << "\n";

      return os;
    }

    template<typename F> std::size_t apply(F& f){
      for(auto& v: value){ f(v); }
      return value.size();
    }
  };

  template<typename ValueType>
  class NonOutputLayer {
  public:
    using value_type = ValueType;
    using layer_type = std::vector<value_type>;

  private:
    Layer<value_type> layer;
    WeightBias<value_type> weight_bias;

  public:
    NonOutputLayer() = default;
    NonOutputLayer(const NonOutputLayer&) = default;
    NonOutputLayer(NonOutputLayer&&) = default;
    NonOutputLayer(Layer<value_type> l): layer{l} {}
    NonOutputLayer& operator=(const NonOutputLayer&) = default;
    NonOutputLayer& operator=(NonOutputLayer&&) = default;
    ~NonOutputLayer() = default;

    void SetNextLayer(std::size_t next_size){
      weight_bias = WeightBias<value_type>{layer.size,next_size};
    }

    friend inline auto operator>>(const layer_type& u,NonOutputLayer<ValueType>& l){
      return l.weight_bias * (u >> l.layer);
    }

    inline auto operator<<(const layer_type& df_du){
      return layer << weight_bias.BackPropagate(layer.value,df_du);
    }

    friend inline std::ostream& operator<<(std::ostream& os,
					   const NonOutputLayer<ValueType>& l){
      os << l.layer << "\n" << l.weight_bias << "\n";
      return os;
    }

    template<typename F> std::size_t apply(F& f){ return weight_bias.apply(f); }
    std::size_t update(value_type eps,value_type L1,value_type L2){
      weight_bias.Update(eps,L1,L2);
      return 1ul;
    }
  };

  template<typename ValueType>
  class OutputLayer {
  public:
    using value_type = ValueType;
    using layer_type = std::vector<value_type>;
    using activation_type = std::function<layer_type(layer_type)>;
    using loss_type = std::function<value_type(layer_type,layer_type)>;
    using diff_type = std::function<layer_type(layer_type,layer_type)>;

  private:
    layer_type value;

    activation_type activate;
    loss_type loss;
    diff_type differentiate;

  public:
    OutputLayer() = default;
    OutputLayer(const OutputLayer&) = default;
    OutputLayer(OutputLayer&&) = default;
    OutputLayer(std::size_t size,activation_type a,loss_type l,diff_type d)
      : value(size), activate{a}, loss{l}, differentiate{d} {}
    OutputLayer& operator=(const OutputLayer&) = default;
    OutputLayer& operator=(OutputLayer&&) = default;
    ~OutputLayer() = default;


    friend inline auto operator>>(const layer_type& u,OutputLayer<ValueType>& l){
      l.value = l.activate(u);
      return l.value;
    }

    inline auto Loss(const layer_type& real){
      return loss(value,real);
    }

    inline auto operator<<(const std::vector<value_type>& real){
      return differentiate(value,real);
    }

    friend inline std::ostream& operator<<(std::ostream& os,
					   const OutputLayer<ValueType>& l){
      os << "layer z\n";
      for(auto& ze: l.value){ os << ze << " "; }
      os << "\n";

      return os;
    }

    template<typename F> std::size_t apply(F& f){ return 0ul; }
    std::size_t update(value_type,value_type,value_type){ return 0ul; }
  };


  template<typename...Layers> class NeuralNet {
  public:
    using value_type = std::common_type_t<typename Layers::value_type...>;
    using layer_type = std::vector<value_type>;

  private:
    std::tuple<Layers...> layers;

  public:
    NeuralNet() = delete;
    NeuralNet(const std::tuple<Layers...>& layers): layers{layers} {}
    NeuralNet(std::tuple<Layers...>&& layers): layers{layers} {}
    NeuralNet(const NeuralNet&) = default;
    NeuralNet(NeuralNet&&) = default;
    NeuralNet& operator=(const NeuralNet&) = default;
    NeuralNet& operator=(NeuralNet&&) = default;
    ~NeuralNet() = default;

    friend auto operator>>(layer_type input,NeuralNet<Layers...>& nn){
      return std::apply([&](auto&&...l){ return (input >> ... >> l); },nn.layers);
    }

    auto operator<<(layer_type real){
      std::apply([&](auto&&...l){ (l << ... << real); },layers);
    }

    auto operator<<(std::tuple<layer_type,layer_type> input_real){
      std::get<0>(input_real) >> (*this);
      (*this) << std::get<1>(input_real);
    }

    auto Loss(const layer_type& real){
      return std::get<sizeof...(Layers)-1>(layers).Loss(real);
    }
    auto Loss(layer_type input, const layer_type& real){
      input >> (*this);
      return Loss(real);
    }

    auto AddLayer(std::size_t next_size,
		  std::function<value_type(value_type)> activate,
		  std::function<value_type(value_type)> differentiate){
      std::get<sizeof...(Layers)-1>(layers).SetNextLayer(next_size);

      auto new_layer = Layer<value_type>{next_size,
					 [=](auto u){
					   for(auto& ue: u){ ue = activate(ue); }
					   return u;
					 },
					 [=](auto z){
					   for(auto& ze: z){ ze = differentiate(ze); }
					   return z;
					 }};

      using NN = NeuralNet<Layers...,decltype(NonOutputLayer{new_layer})>;
      return NN{std::tuple_cat(layers,std::make_tuple(NonOutputLayer{new_layer}))};
    }

    auto AddOutputLayer(std::size_t next_size,
			std::function<layer_type(layer_type)> activate,
			std::function<value_type(layer_type,layer_type)> loss,
			std::function<layer_type(layer_type,layer_type)> diff){
      std::get<sizeof...(Layers)-1>(layers).SetNextLayer(next_size);
      auto new_layer = OutputLayer<value_type>{next_size,activate,loss,diff};

      using NN = NeuralNet<Layers...,decltype(new_layer)>;
      return NN(std::tuple_cat(layers,std::make_tuple(new_layer)));
    }

    template<typename F> void Apply(F f){
      std::apply([&](auto&...l){ (... + l.apply(f)); },layers);
    }

    void Update(value_type eps,value_type L1,value_type L2){
      std::apply([=](auto&...l){ (... + l.update(eps,L1,L2)); },layers);
    }

    void DebugShow() const {
      std::cout << "DEBUG: show (saved) layer-z and weight and bias" << std::endl;
      std::apply([](auto&&...l){ (std::cout << ... << l) << std::endl; },layers);
    }
  };

  template<typename ValueType=double> struct InputLayer {
    using value_type = ValueType;
    std::size_t size;

    template<typename A,typename D> auto AddLayer(std::size_t next_size,
						  A activate,D differentiate){
      auto layer1 = Layer<value_type>{size,
				      [=](auto u){ return u; },
				      [=](auto z){
					for(auto& ze: z){ ze = value_type{1}; }
					return z;
				      }};

      auto layer2 = Layer<value_type>{next_size,
				      [=](auto u){
					for(auto& ue: u){ ue = activate(ue); }
					return u;
				      },
				      [=](auto z){
					for(auto& ze: z){ ze = differentiate(ze); }
					return z;
				      }};

      auto layers = std::make_tuple(NonOutputLayer{layer1},
				    NonOutputLayer{layer2});
      std::get<0>(layers).SetNextLayer(next_size);

      return NeuralNet{layers};
    }
  };

  struct ReLU { std::size_t size; };
  template<typename NN> inline auto operator>>(NN nn,ReLU relu){
    using value_type = typename NN::value_type;
    constexpr const auto zero = value_type{0};
    constexpr const auto one = value_type{1};
    return nn.AddLayer(relu.size,
		       [=](value_type u){ return std::max(zero,u); },
		       [=](value_type z){ return std::signbit(z) ? zero : one; });
  }

  struct Sigmoid { std::size_t size; };
  template<typename NN> inline auto operator>>(NN nn,Sigmoid sigmoid){
    using value_type = typename NN::value_type;
    constexpr const auto one = value_type{1};
    return nn.AddLayer(sigmoid.size,
		       [=](value_type u){ return one/(one + std::exp(-u)); },
		       [=](value_type z){ return z*(one - z); });
  }

  struct Identity_SquareError { std::size_t size; };
  template<typename NN>inline auto operator>>(NN nn,Identity_SquareError identity){
    using value_type = typename NN::value_type;
    using layer_type = std::vector<value_type>;
    return nn.AddOutputLayer(identity.size,
			     [ ](layer_type u){ return u; },
			     [ ](layer_type z,layer_type real){
			       auto loss = value_type{0};
			       for(auto&& [ze,r]: ymd::zip(z,real)){
				 loss += std::pow(ze-r,2);
			       }
			       return 0.5*loss;
			     },
			     [](layer_type z,layer_type real){
			       for(auto&& [ze,r]: ymd::zip(z,real)){ ze -= r; }
			       return z;
			     });
  }

  struct SoftMax_CrossEntropy { std::size_t size; };
  template<typename NN>inline auto operator>>(NN nn,SoftMax_CrossEntropy softmax){
    using value_type = typename NN::value_type;
    using layer_type = std::vector<value_type>;
    const auto eps = value_type{1e-10};

    return nn.AddOutputLayer(softmax.size,
			     [](layer_type u){
			       auto max = *std::max_element(u.begin(),u.end());

			       auto sum = value_type{0};
			       for(auto& ue: u){
				 ue = std::exp(ue-max);
				 sum += ue;
			       }

			       for(auto& ue: u){ ue /= sum; }

			       return u;
			     },
			     [=](layer_type z, layer_type real){
			       auto loss = value_type{0};

			       for(auto&& [ze,r]: ymd::zip(z,real)){
				 loss -= r * std::log(ze + eps);
			       }

			       return loss;
			     },
			     [](layer_type z,layer_type real){
			       for(auto&& [ze,r]: ymd::zip(z,real)){ ze -= r; }
			       return z;
			     });
  }

} // namespace ymd
#endif // YMD_DL_CORE
