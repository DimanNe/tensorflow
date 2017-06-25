#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

namespace tf = tensorflow;
namespace to = tf::ops;

int main() {
   tf::Scope r     = tf::Scope::NewRootScope();
   tf::Scope Root  = r.ExitOnError();
   auto      x     = to::Placeholder(Root, tf::DT_DOUBLE);
   auto      W     = to::RandomUniform(Root, 10, tf::DT_DOUBLE);
   auto      b     = to::RandomUniform(Root, 10, tf::DT_DOUBLE);
   auto      Model = to::Sum(Root, b, to::MatMul(Root, W, x, to::MatMul::TransposeB(true)));

   std::vector<tf::Tensor> Outputs;
   tf::ClientSession       Session(Root);
   TF_CHECK_OK(Session.Run({Model}, &Outputs));
   LOG(INFO) << Outputs[0].matrix<double>();

   return 0;


   // tf::Scope root = tf::Scope::NewRootScope();
   // // Matrix A = [3 2; -1 0]
   // auto A = to::Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
   // // Vector b = [3 5]
   // auto b = to::Const(root, {{3.f, 5.f}});
   // // v = Ab^T
   // auto v = to::MatMul(root.WithOpName("v"), A, b, to::MatMul::TransposeB(true));

   // std::vector<tf::Tensor> outputs;
   // tf::ClientSession       session(root);
   // // Run and fetch v
   // TF_CHECK_OK(session.Run({v}, &outputs));
   // // Expect outputs[0] == [19; -3]
   // LOG(INFO) << outputs[0].matrix<float>();
   // return 0;
}
