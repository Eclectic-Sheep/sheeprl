#include "torch/torch.h"
namespace bar {

struct Bar {
	Bar(torch::Tensor tensor);
	void set_tensor(torch::Tensor tensor);
	torch::Tensor get_tensor();
	torch::Tensor tensor;
};

}  // namespace bar