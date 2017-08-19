#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/cc/DimanNe/tensorflow_utils/variable_initializer.h"

namespace tf = tensorflow;
namespace to = tf::ops;

tf::Output AddLayer(tf::Scope                  s,
                    tfu::TVariableInitializer &InitV,
                    const tf::int64            NumOfInputs,
                    const tf::int64            LayerWidth,
                    const tf::Output &         PrevOutput,
                    tf::Output *               Weights) {
    using TRand = to::ParameterizedTruncatedNormal;

    // tf::Output b = InitV.Create<TRand>(s.WithOpName("b"), {1, LayerWidth}, tf::DT_DOUBLE, {}, 1., 1., 0., 3.);
    tf::Output W = InitV.Create<TRand>(s.WithOpName("W"), {NumOfInputs, LayerWidth}, tf::DT_DOUBLE, {}, 1., 2., -2., 3.);
    if(Weights)
        *Weights = W;

    tf::Output xW = to::MatMul(s, PrevOutput, W);
    // tf::Output xWb   = to::Add(s, xW, b);
    // tf::Output Layer = to::Relu(s, xW);
    return xW;
}

// tf::Output CreateMLP(tf::Scope &s, tfu::TVariableInitializer &InitV, const tf::Output &x) {
//     const tf::PartialTensorShape ShapeOfInput    = tfu::GetShapeOfOutput(x);
//     const tf::int64              NumOfElemInItem = ShapeOfInput.dim_sizes().back();
//
//     tf::Output Result = x;
//     Result            = AddLayer(s.WithOpName("Layer1"), InitV, NumOfElemInItem, 3, Result);
//     Result            = AddLayer(s.WithOpName("Layer2"), InitV, 3, 1, Result);
//
//     return Result;
// }


tf::Output CreateMLP(tf::Scope &s, tfu::TVariableInitializer &InitV, const tf::Output &x, tf::Output &Weights) {
    const tf::PartialTensorShape ShapeOfInput    = tfu::GetShapeOfOutput(x);
    const tf::int64              NumOfElemInItem = ShapeOfInput.dim_sizes().back();

    tf::Output Result = x;
    Result            = AddLayer(s.WithOpName("Layer1"), InitV, NumOfElemInItem, 1, Result, &Weights);

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
    // const tf::Output Model = CreateMLP(s, InitV, x);
    tf::Output       Weights;
    const tf::Output Model = CreateMLP(s, InitV, x, Weights);
    const tf::Output Loss  = AddLossFunction(s, Model, Expec);

    tf::ClientSession Session(s);
    InitV(Session, s);


    tf::Output     dLossdLoss = to::Const(s, {1.});
    tf::OutputList dLossdWeights;
    TF_CHECK_OK(tf::AddSymbolicGradients(s, {Loss}, {Weights}, {dLossdLoss}, &dLossdWeights));

    // clang-format off
    tf::ClientSession::FeedType Feed = {
        { x,     {{1., 1.}/*, {0.01, 0.01}*/}     },
        { Expec, {{1.}                      }     }
    };
    // clang-format on

    const tf::Output ApplySGD =
        to::ApplyGradientDescent(s, Weights, to::Const(s.WithOpName("learning_rate"), 0.1), dLossdWeights.front());


    for(tf::uint32 i = 0; i < 10; ++i) {
        {
            std::vector<tf::Tensor> Outputs;
            TF_CHECK_OK(Session.Run(Feed, {Model, Loss}, &Outputs));
            LOG(INFO) << "Prediction: " << Outputs[0].matrix<double>() << ", Loss: " << Outputs[1].matrix<double>();
        }
        {
            std::vector<tf::Tensor> Outputs;
            TF_CHECK_OK(Session.Run(Feed, {ApplySGD}, &Outputs));
        }
    }

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




// Compiler run:
// /usr/lib/gcc/x86_64-linux-gnu/6/cc1plus -quiet -imultiarch x86_64-linux-gnu -MD
// bazel-out/local-opt/bin/tensorflow/cc/_objs/math_grad/tensorflow/cc/gradients/math_grad.d -MF
// bazel-out/local-opt/bin/tensorflow/cc/_objs/math_grad/tensorflow/cc/gradients/math_grad.d -MQ
// bazel-out/local-opt/bin/tensorflow/cc/_objs/math_grad/tensorflow/cc/gradients/math_grad.o -D_GNU_SOURCE -U _FORTIFY_SOURCE
// -D _FORTIFY_SOURCE=1 -D NDEBUG -D EIGEN_MPL2_ONLY -D TENSORFLOW_USE_JEMALLOC -D SNAPPY -D __DATE__="redacted" -D
// __TIMESTAMP__="redacted" -D __TIME__="redacted" -iquote . -iquote bazel-out/local-opt/genfiles -iquote external/jemalloc
// -iquote bazel-out/local-opt/genfiles/external/jemalloc -iquote external/bazel_tools -iquote
// bazel-out/local-opt/genfiles/external/bazel_tools -iquote external/protobuf -iquote
// bazel-out/local-opt/genfiles/external/protobuf -iquote external/eigen_archive -iquote
// bazel-out/local-opt/genfiles/external/eigen_archive -iquote external/local_config_sycl -iquote
// bazel-out/local-opt/genfiles/external/local_config_sycl -iquote external/gif_archive -iquote
// bazel-out/local-opt/genfiles/external/gif_archive -iquote external/jpeg -iquote bazel-out/local-opt/genfiles/external/jpeg
// -iquote external/com_googlesource_code_re2 -iquote bazel-out/local-opt/genfiles/external/com_googlesource_code_re2 -iquote
// external/farmhash_archive -iquote bazel-out/local-opt/genfiles/external/farmhash_archive -iquote external/fft2d -iquote
// bazel-out/local-opt/genfiles/external/fft2d -iquote external/highwayhash -iquote
// bazel-out/local-opt/genfiles/external/highwayhash -iquote external/png_archive -iquote
// bazel-out/local-opt/genfiles/external/png_archive -iquote external/zlib_archive -iquote
// bazel-out/local-opt/genfiles/external/zlib_archive -iquote external/snappy -iquote
// bazel-out/local-opt/genfiles/external/snappy -iquote external/local_config_cuda -iquote
// bazel-out/local-opt/genfiles/external/local_config_cuda -iquote external/curl -iquote
// bazel-out/local-opt/genfiles/external/curl -iquote external/boringssl -iquote
// bazel-out/local-opt/genfiles/external/boringssl -iquote external/jsoncpp_git -iquote
// bazel-out/local-opt/genfiles/external/jsoncpp_git -isystem external/jemalloc/include -isystem
// bazel-out/local-opt/genfiles/external/jemalloc/include -isystem external/bazel_tools/tools/cpp/gcc3 -isystem
// external/protobuf/src -isystem bazel-out/local-opt/genfiles/external/protobuf/src -isystem external/eigen_archive -isystem
// bazel-out/local-opt/genfiles/external/eigen_archive -isystem external/gif_archive/lib -isystem
// bazel-out/local-opt/genfiles/external/gif_archive/lib -isystem external/farmhash_archive/src -isystem
// bazel-out/local-opt/genfiles/external/farmhash_archive/src -isystem external/png_archive -isystem
// bazel-out/local-opt/genfiles/external/png_archive -isystem external/zlib_archive -isystem
// bazel-out/local-opt/genfiles/external/zlib_archive -isystem external/local_config_cuda/cuda -isystem
// bazel-out/local-opt/genfiles/external/local_config_cuda/cuda -isystem external/local_config_cuda/cuda/include -isystem
// bazel-out/local-opt/genfiles/external/local_config_cuda/cuda/include -isystem external/curl/include -isystem
// bazel-out/local-opt/genfiles/external/curl/include -isystem external/boringssl/src/include -isystem
// bazel-out/local-opt/genfiles/external/boringssl/src/include -isystem external/jsoncpp_git/include -isystem
// bazel-out/local-opt/genfiles/external/jsoncpp_git/include tensorflow/cc/gradients/math_grad.cc -quiet -dumpbase
// math_grad.cc -mavx -msse4.2 -mtune=generic -march=x86-64 -auxbase-strip
// bazel-out/local-opt/bin/tensorflow/cc/_objs/math_grad/tensorflow/cc/gradients/math_grad.o -g0 -g -O2 -Wall
// -Wunused-but-set-parameter -Wno-free-nonheap-object -Wno-builtin-macro-redefined -std=c++11 -fstack-protector
// -fno-omit-frame-pointer -ffunction-sections -fdata-sections
// -frandom-seed=bazel-out/local-opt/bin/tensorflow/cc/_objs/math_grad/tensorflow/cc/gradients/math_grad.o
// -fno-canonical-system-headers -Wformat-security -o /tmp/cctrFkMC.s
