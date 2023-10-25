
#include "bar.h"

#include <iostream>

#include "torch/extension.h"
#include "torch/torch.h"

namespace bar {
bar::Bar::Bar(torch::Tensor tensor) : tensor(tensor) {}
torch::Tensor bar::Bar::get_tensor() { return this->tensor; }
void bar::Bar::set_tensor(torch::Tensor tensor) { this->tensor = tensor; }
};	// namespace bar