#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/cc/DimanNe/tensorflow_utils/variable_initializer.h"

namespace tf = tensorflow;
namespace to = tf::ops;

tf::Output AddLayer(tf::Scope                  s,
                    tfu::TVariableInitializer &InitV,
                    const tf::int64            NumOfInputs,
                    const tf::int64            LayerWidth,
                    const tf::Output &         PrevOutput) {
    using TRand = to::ParameterizedTruncatedNormal;

    // tf::Output b = InitV.Create<TRand>(s.WithOpName("b"), {1, LayerWidth}, tf::DT_DOUBLE, {}, 1., 1., 0., 3.);
    tf::Output W = InitV.Create<TRand>(s.WithOpName("W"), {NumOfInputs, LayerWidth}, tf::DT_DOUBLE, {}, 1., 2., -2., 3.);

    tf::Output xW = to::MatMul(s, PrevOutput, W);
    // tf::Output xWb   = to::Add(s, xW, b);
    tf::Output Layer = to::Relu(s, xW);
    return Layer;
}

tf::Output CreateMLP(tf::Scope &s, tfu::TVariableInitializer &InitV, const tf::Output &x) {
    const tf::PartialTensorShape ShapeOfInput    = tfu::GetShapeOfOutput(x);
    const tf::int64              NumOfElemInItem = ShapeOfInput.dim_sizes().back();

    tf::Output Result = x;
    Result            = AddLayer(s.WithOpName("Layer1"), InitV, NumOfElemInItem, 3, Result);
    Result            = AddLayer(s.WithOpName("Layer2"), InitV, 3, 1, Result);

    return Result;
}

tf::Output AddLossFunction(tf::Scope &s, const tf::Output &ActualValue, const tf::Input &ExpectedValue) {
    tf::Output Result = to::SquaredDifference(s.WithOpName("SqDiff"), ActualValue, ExpectedValue);
    return Result;
}

int main() {
    tf::Scope                 r = tf::Scope::NewRootScope();
    tf::Scope                 s = r.ExitOnError();
    tfu::TVariableInitializer InitV;

    const tf::Output x     = to::Placeholder(s.WithOpName("xInput"), tf::DT_DOUBLE, to::Placeholder::Attrs().Shape({-1, 2}));
    const tf::Output Expec = to::Placeholder(s.WithOpName("ExpVal"), tf::DT_DOUBLE, to::Placeholder::Attrs().Shape({-1, 1}));
    const tf::Output Model = CreateMLP(s, InitV, x);
    const tf::Output Loss  = AddLossFunction(s, Model, Expec);



    std::vector<tf::Tensor> Outputs;
    // clang-format off
    tf::ClientSession::FeedType Feed = {
        { x,     {{1., 1.}/*, {0.01, 0.01}*/}     },
        { Expec, {{1.}                      }     }
    };
    // clang-format on
    tf::ClientSession Session(s);
    InitV(Session);
    TF_CHECK_OK(Session.Run(Feed, {Model, Loss}, &Outputs));
    LOG(INFO) << "Prediction: " << Outputs[0].matrix<double>();
    LOG(INFO) << "Loss:       " << Outputs[1].matrix<double>();

    return 0;
}

// gradients_test.cc


// training_ops_test.cc
// static void SGD(int32 n, Graph** init_g, Graph** train_g) {
//   {
//     Graph* g = new Graph(OpRegistry::Global());
//     auto var = Var(g, n);
//     test::graph::Assign(g, var, Zeros(g, n));
//     *init_g = g;
//   }
//   {
//     Graph* g = new Graph(OpRegistry::Global());
//     auto var = Var(g, n);
//     auto lr = Scalar(g, 0.01);
//     auto grad = Random(g, n);
//     test::graph::Multi(g, "ApplyGradientDescent", {var, lr, grad});
//     *train_g = g;
//   }
// }


// training_ops.cc
// template <typename Device, typename T>
// class ApplyGradientDescentOp : public OpKernel {
//  public:
//   explicit ApplyGradientDescentOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
//     OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
//   }
//
//   void Compute(OpKernelContext* ctx) override {
//     auto locks =
//         MaybeLockVariableInputMutexesInOrder(ctx, use_exclusive_lock_, {0});
//     Tensor var;
//     OP_REQUIRES_OK(
//         ctx, GetInputTensorFromVariable(ctx, 0, use_exclusive_lock_, &var));
//
//     OP_REQUIRES(
//         ctx, var.IsInitialized(),
//         errors::FailedPrecondition(
//             "Attempting to use uninitialized variables: ", def().input(0)));
//     const Tensor& alpha = ctx->input(1);
//     OP_REQUIRES(ctx, IsLegacyScalar(alpha.shape()),
//                 errors::InvalidArgument("alpha is not a scalar: ",
//                                         alpha.shape().DebugString()));
//     const Tensor& delta = ctx->input(2);
//     OP_REQUIRES(
//         ctx, var.shape().IsSameSize(delta.shape()),
//         errors::InvalidArgument("var and delta do not have the same shape",
//                                 var.shape().DebugString(), " ",
//                                 delta.shape().DebugString()));
//
//     const Device& device = ctx->template eigen_device<Device>();
//     functor::ApplyGradientDescent<Device, T>()(
//         device, var.flat<T>(), alpha.scalar<T>(), delta.flat<T>());
//
//     MaybeForwardRefInputToRefOutput(ctx, 0, 0);
//   }
//
//  private:
//   bool use_exclusive_lock_;
// };
