
#include "torch/torch.h"
namespace foo {

struct Foo {
	Foo(torch::Tensor tensor);
	void set_tensor(torch::Tensor tensor);
	torch::Tensor get_tensor();
	torch::Tensor tensor;
};

}  // namespace foo