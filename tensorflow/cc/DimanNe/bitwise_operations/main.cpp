#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/cc/DimanNe/tensorflow_utils/shape_helpers.h"
#include "tensorflow/cc/DimanNe/tensorflow_utils/variable_initializer.h"

namespace tf = tensorflow;
namespace to = tf::ops;

tf::Output AddLayer(tf::Scope                  s,
                    tfu::TVariableInitializer &InitV,
                    const tf::int64            NumOfInputs,
                    const tf::int64            LayerWidth,
                    const tf::Output &         PrevOutput) {
    using TRand = to::ParameterizedTruncatedNormal;

    tf::Output b = InitV.Create<TRand>(s.WithOpName("b"), {1, LayerWidth}, tf::DT_DOUBLE, {}, 1., 1., 0., 3.);
    tf::Output W = InitV.Create<TRand>(s.WithOpName("W"), {NumOfInputs, LayerWidth}, tf::DT_DOUBLE, {}, 1., 2., -2., 3.);

    tf::Output xW    = to::MatMul(s, PrevOutput, W);
    tf::Output xWb   = to::Add(s, xW, b);
    tf::Output Layer = to::Relu(s, xWb);
    return Layer;
}

tf::Output CreateMLP(tf::Scope &s, tfu::TVariableInitializer &InitV, const tf::Output &x) {
    const tf::PartialTensorShape ShapeOfInput    = tfu::GetShapeOfOutput(x);
    const tf::int64              NumOfElemInItem = ShapeOfInput.dim_sizes().back();

    tf::Output Result = x;

    // for(tf::int64 NumOfHiddenLayers = 0; NumOfHiddenLayers < 10; ++NumOfHiddenLayers)
    //     Result = AddLayer(s, InitV, NumOfElemInItem, 20, Result);

    Result = AddLayer(s.WithOpName("Layer1"), InitV, NumOfElemInItem, 10, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);
    Result = AddLayer(s.WithOpName("Layer2"), InitV, 20, 20, Result);

    return Result;
}

int main() {
    tf::Scope                 r = tf::Scope::NewRootScope();
    tf::Scope                 s = r.ExitOnError();
    tfu::TVariableInitializer InitV;

    tf::Output x     = to::Placeholder(s.WithOpName("xInput"), tf::DT_DOUBLE, to::Placeholder::Attrs().Shape({-1, 2}));
    tf::Output Model = CreateMLP(s, InitV, x);

    std::vector<tf::Tensor> Outputs;
    // clang-format off
    tf::ClientSession::FeedType Feed = {
        {x, {{1., 1.}/*, {0.01, 0.01}*/}}
    };
    // clang-format on
    tf::ClientSession Session(s);
    InitV(Session);
    TF_CHECK_OK(Session.Run(Feed, {Model}, &Outputs));
    LOG(INFO) << Outputs[0].matrix<double>();

    return 0;
}
