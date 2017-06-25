#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

namespace tf = tensorflow;
namespace to = tf::ops;

int main() {
   tf::Scope         r = tf::Scope::NewRootScope();
   tf::Scope         s = r.ExitOnError();
   tf::ClientSession Session(s);


   tf::Output RandW     = to::RandomUniform(s, {10, 1}, tf::DT_DOUBLE);
   tf::Output W = to::Variable(s.WithOpName("W"), {10, 1}, tf::DT_DOUBLE);
   tf::Output AssignToW = to::Assign(s, W, RandW);

   tf::Output RandB     = to::RandomUniform(s, {10, 1}, tf::DT_DOUBLE);
   tf::Output b = to::Variable(s.WithOpName("b"), {10, 1}, tf::DT_DOUBLE);
   tf::Output AssignToB = to::Assign(s, b, RandB);

   {
      std::vector<tf::Tensor> OutputsOfAssigning;
      TF_CHECK_OK(Session.Run({AssignToW}, &OutputsOfAssigning));
      LOG(INFO) << OutputsOfAssigning[0].matrix<double>();
   }

   {
      std::vector<tf::Tensor> OutputsOfAssigning;
      TF_CHECK_OK(Session.Run({AssignToB}, &OutputsOfAssigning));
      LOG(INFO) << OutputsOfAssigning[0].matrix<double>();
   }


   tf::Output x     = to::Placeholder(s.WithOpName("xInput"), tf::DT_DOUBLE);
   tf::Output Wx    = to::MatMul(s.WithOpName("Wx"), W, x, to::MatMul::TransposeB(true));
   tf::Output Model = to::AddN(s.WithOpName("Wxb"), {b, Wx});

   std::vector<tf::Tensor> Outputs;
   TF_CHECK_OK(Session.Run({{x, {{1., 2., 3., 4., 5., 6., 7., 8., 9., 0.}, {}}}}, {Model}, &Outputs));
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
