
#include "foo.h"

#include <iostream>

#include "torch/extension.h"
#include "torch/torch.h"

foo::Foo::Foo(torch::Tensor tensor) : tensor(tensor) {}
torch::Tensor foo::Foo::get_tensor() { return this->tensor; }
void foo::Foo::set_tensor(torch::Tensor tensor) { this->tensor = tensor; }