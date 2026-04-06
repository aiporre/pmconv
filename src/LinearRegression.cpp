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
   const int64_t N = 10;
   auto x = torch::rand({N, 2});
   auto w_true = torch::tensor({{2.0, 3.0}, {4.0, 5.0}});
   auto b_true = torch::tensor({1.0, 2.0});
   // fake data y = x * w_true + b_true
   auto y_true = x.mm(w_true) + b_true;

   // create parameters for linear regression
//   auto w = torch::zeros({2, 2}, torch::TensorOptions().requires_grad(true));
//   auto b = torch::zeros({2}, torch::TensorOptions().requires_grad(true));
   auto w = torch::randn({2, 2}, torch::TensorOptions().requires_grad(true));
   auto b = torch::randn({2}, torch::TensorOptions().requires_grad(true));

   // training loop
   //   const double lr = 0.1;
   // AdamW optimizer
   torch::optim::AdamW optimizer(
       {w, b},
       torch::optim::AdamWOptions(/*lr=*/1e-1).weight_decay(1e-2)
   );

   for (int epoch = 0; epoch < 3000; ++epoch) {
       // foward pass
       auto y_pred = linear_regression(x, w, b);
       auto loss = torch::mse_loss(y_pred, y_true);
       // backward pass
       optimizer.zero_grad();
       loss.backward();
       optimizer.step();


       //       // update parameters
       //       {
       //           torch::NoGradGuard no_grad;
       //           // update w and b using gradient descent
       //           w -= lr * w.grad();
       //           b -= lr * b.grad();
       //       }
       //
       //       // zero gradients
       //       w.grad().zero_();
       //       b.grad().zero_();
       if (epoch % 50 == 0) {
            std::cout << "Epoch [" << epoch << "/100], Loss: " << loss.item<double>() << std::endl;
       }

   }
   // print the learned parameters
   std::cout << "Learned weight:\n" << w << std::endl;
   std::cout << "Learned bias:\n" << b << std::endl;

   // print the true parameters
   std::cout << "True weight:\n" << w_true << std::endl;
   std::cout << "True bias:\n" << b_true << std::endl;

   return 0;
}
