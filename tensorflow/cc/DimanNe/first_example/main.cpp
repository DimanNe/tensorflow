#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/cc/DimanNe/tensorflow_utils/variable_initializer.h"


namespace tf = tensorflow;
namespace to = tf::ops;


int main() {
    tf::Scope                 r = tf::Scope::NewRootScope();
    tf::Scope                 s = r.ExitOnError();
    tfu::TVariableInitializer InitV;

    tf::Output W = InitV.Create<to::RandomUniform>(s.WithOpName("W"), {1, 10}, tf::DT_DOUBLE, {}, tf::DT_DOUBLE);
    tf::Output b =
        InitV.Create<to::ParameterizedTruncatedNormal>(s.WithOpName("b"), {1, 1}, tf::DT_DOUBLE, {}, 10., 1., 0., 20.);

    tf::Output x     = to::Placeholder(s.WithOpName("xInput"), tf::DataTypeToEnum<double>::v());
    tf::Output Wx    = to::MatMul(s.WithOpName("Wx"), W, x, to::MatMul::TransposeB(true));
    tf::Output Model = to::AddN(s.WithOpName("Wxb"), {b, Wx});


    std::vector<tf::Tensor>     Outputs;
    tf::ClientSession::FeedType Feed = {
        {x, {{1., 2., 3., 4., 5., 6., 7., 8., 9., 0.}}} // 10x1 matrix
    };
    tf::ClientSession Session(s);
    InitV(Session);
    TF_CHECK_OK(Session.Run(Feed, {Model}, &Outputs));
    LOG(INFO) << Outputs[0].matrix<double>();

    return 0;
}






///
/// Real operations
/// https://www.tensorflow.org/extend/adding_an_op
///
/// 1  tensorflow::Tensor::CheckTypeAndIsAligned tensor.cc                            487  0x55555700bbca
/// 2  tensorflow::Tensor::shaped<int, 1ul> tensor.h                             612  0x55555577d930
/// 3  tensorflow::Tensor::flat<int> tensor.h                             364  0x55555661a73a
/// 4  tensorflow::(anonymous namespace)::ParameterizedTruncatedNormalOp<Eigen::ThreadPoolDevice, double>::Compute
/// parameterized_truncated_normal_op.cc 263  0x55555661a73a
/// 5  tensorflow::ThreadPoolDevice::Compute threadpool_device.cc                 59   0x555556e5a434
/// 6  tensorflow::(anonymous namespace)::ExecutorState::Process executor.cc                          1653 0x555556e27751
/// 7  tensorflow::(anonymous namespace)::ExecutorState::<lambda()>::operator()
/// executor.cc                          2056 0x555556e2797f
/// 8  std::_Function_handler<void(), tensorflow::(anonymous namespace)::ExecutorState::ScheduleReady(const TaggedNodeSeq&,
/// tensorflow::(anonymous namespace)::ExecutorState::TaggedNodeReadyQueue *)::<lambda()>>::_M_invoke(const std::_Any_data &)
/// functional                           1731 0x555556e2797f
/// 9  std::function<void ()>::operator()() const functional                           2127 0x555557071fd1
/// 10 tensorflow::thread::EigenEnvironment::ExecuteTask threadpool.cc                        81   0x555557071fd1
/// 11 Eigen::NonBlockingThreadPoolTempl<tensorflow::thread::EigenEnvironment>::WorkerLoop
/// NonBlockingThreadPool.h              232  0x555557071fd1
/// 12 std::function<void ()>::operator()() const functional                           2127 0x555557070137
/// 13 tensorflow::thread::EigenEnvironment::CreateThread(std::function<void ()> const&)::{lambda()#1}::operator()() const
/// threadpool.cc                        56   0x555557070137
/// 14 std::_Function_handler<void (), tensorflow::thread::EigenEnvironment::CreateThread(std::function<void ()>
/// const&)::{lambda()#1}>::_M_invoke(std::_Any_data const&) functional                           1731 0x555557070137
/// 15 ?? 0x7ffff71d783f
/// 16 start_thread pthread_create.c                     456  0x7ffff74ab6da
/// 17 clone clone.S                              105  0x7ffff6c46d7f
