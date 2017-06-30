#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/lib/gtl/array_slice.h"


namespace tf = tensorflow;
namespace to = tf::ops;

namespace tfu {

    tensorflow::Input InputFromTensorShape(const tensorflow::TensorShape &Shape) {
        tf::Tensor Tensor(tf::DT_INT64, {Shape.dims()});
        tf::int16  i = 0;
        for(const tf::TensorShapeDim Dimenstion : Shape) {
            Tensor.flat<tf::int64>()(i) = Dimenstion.size;
            // LOG(INFO) << "Assigning to ith: " << i << " element in tensor value: " << Dimenstion.size;
            ++i;
        }
        tensorflow::Input Result(Tensor);
        return Result;
    }

    class TVariableInitializer {
    public:
        template <class TDistr, class... TDistrArgs>
        tensorflow::ops::Variable Create(const tensorflow::Scope &               Scope,
                                         const tensorflow::TensorShape &         Shape,
                                         const tensorflow::DataType              Dtype,
                                         const tensorflow::ops::Variable::Attrs &VarAttrs,
                                         TDistrArgs &&... DistrArgs) {
            namespace tf = tensorflow;
            namespace to = tf::ops;

            const tf::Input ShapeAsInput = InputFromTensorShape(Shape);
            TDistr          Distribution = TDistr(Scope, ShapeAsInput, std::forward<TDistrArgs>(DistrArgs)...);
            to::Variable    Variable     = {Scope, Shape, Dtype, VarAttrs};
            to::Assign      Assign       = to::Assign(Scope, Variable, Distribution);
            Assigns.push_back(Assign);
            return Variable;
        }

        void operator()(tensorflow::ClientSession &Session) const {
            for(const tensorflow::ops::Assign &Assign : Assigns) {
                std::vector<tf::Tensor> OutputsOfAssigning;
                TF_CHECK_OK(Session.Run({Assign}, &OutputsOfAssigning));
                // LOG(INFO) << OutputsOfAssigning[0].DebugString();
                // LOG(INFO) << OutputsOfAssigning[0].matrix<double>();
                // LOG(INFO) << OutputsOfAssigning[0].vec<double>();
            }
        }

    private:
        std::vector<tensorflow::ops::Assign> Assigns;
    };
}

int main() {
    tf::Scope                 r = tf::Scope::NewRootScope();
    tf::Scope                 s = r.ExitOnError();
    tf::ClientSession         Session(s);
    tfu::TVariableInitializer InitVars;

    tf::Output W = InitVars.Create<to::RandomUniform>(s.WithOpName("W"), {1, 10}, tf::DT_DOUBLE, {}, tf::DT_DOUBLE);
    tf::Output b = InitVars.Create<to::RandomUniform>(s.WithOpName("b"), {1, 1}, tf::DT_DOUBLE, {}, tf::DT_DOUBLE);



    // tf::Output RandW     = to::RandomUniform(s, {1, 10}, tf::DT_DOUBLE);
    // tf::Output W         = to::Variable(s.WithOpName("W"), {1, 10}, tf::DataTypeToEnum<double>::v());
    // tf::Output AssignToW = to::Assign(s, W, RandW);

    // tf::Output RandB     = to::RandomUniform(s, {1, 1}, tf::DT_DOUBLE);
    // tf::Output b         = to::Variable(s.WithOpName("b"), {1, 1}, tf::DataTypeToEnum<double>::v());
    // tf::Output AssignToB = to::Assign(s, b, RandB);

    InitVars(Session);
    // {
    //     std::vector<tf::Tensor> OutputsOfAssigning;
    //     TF_CHECK_OK(Session.Run({AssignToW}, &OutputsOfAssigning));
    //     LOG(INFO) << OutputsOfAssigning[0].matrix<double>();
    // }

    // {
    //     std::vector<tf::Tensor> OutputsOfAssigning;
    //     TF_CHECK_OK(Session.Run({AssignToB}, &OutputsOfAssigning));
    //     LOG(INFO) << OutputsOfAssigning[0].matrix<double>();
    // }


    tf::Output x     = to::Placeholder(s.WithOpName("xInput"), tf::DataTypeToEnum<double>::v());
    tf::Output Wx    = to::MatMul(s.WithOpName("Wx"), W, x, to::MatMul::TransposeB(true));
    tf::Output Model = to::AddN(s.WithOpName("Wxb"), {b, Wx});

    std::vector<tf::Tensor>     Outputs;
    tf::ClientSession::FeedType Feed = {
        {x, {{1., 2., 3., 4., 5., 6., 7., 8., 9., 0.}}} // 10x1 matrix
    };
    TF_CHECK_OK(Session.Run(Feed, {Model}, &Outputs));
    LOG(INFO) << Outputs[0].matrix<double>();

    return 0;
}
