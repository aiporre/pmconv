#include <torch/torch.h>
#include <iostream>


using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

// y = x * w + b
// define the function to compute the output
class LinearRegression : public torch::autograd::Function<LinearRegression> {
public:
    static variable_list forward(AutogradContext *context, Variable input, Variable weight, Variable bias) {
    // classic linear regression needs the x and w for backward, we save in the context
    context->save_for_backward({input, weight});
    auto output = input.mm(weight) + bias;
    return {output};
    }

    static variable_list backward(AutogradContext *context, variable_list grad_output) {
    auto grad = grad_output[0];
    auto saved = context->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    // compute the gradients
    // dL/dx
    auto grad_input = grad.mm(weight.t()); // dL/dx = dL/dy * dy/dx = grad * w^T
    // dL/dw
    auto grad_weight = input.t().mm(grad); // dL/dw = dL/dy * dy/dw = x^T * grad
    // dL/db
    auto grad_bias = grad.sum(0); // dL/db = dL/dy * dy/db = grad.sum(0)
    return {grad_input, grad_weight, grad_bias};
    }
};

// define a linear regression function
static torch::Tensor linear_regression(Variable input, Variable weight, Variable bias) {
    return LinearRegression::apply(input, weight, bias)[0];
}

int main(){
   std::cout << "Hello, Linear Regression!" << std::endl;
     torch::manual_seed(0);
    // create some random data
    auto x = torch::rand({10, 3});
    auto w = torch::rand({3, 2});
    auto b = torch::rand({2});
    // compute the output
    auto y = linear_regression(x, w, b);
    std::cout << "Output: " << y << std::endl;
     // compute the gradients
    auto grad_output = torch::rand({10, 2});
    auto grad_input = linear_regression::backward(nullptr, {grad_output})[0];
    std::cout << "Gradient w.r.t input: " << grad_input << std::endl;
     auto grad_weight = linear_regression::backward(nullptr, {grad_output})[1];
    std::cout << "Gradient w.r.t weight: " << grad_weight << std::endl;
     auto grad_bias = linear_regression::backward(nullptr, {grad_output})[2];
    std::cout << "Gradient w.r.t bias: " << grad_bias << std::endl;
     return 0;
}
