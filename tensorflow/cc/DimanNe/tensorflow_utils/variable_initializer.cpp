#include "variable_initializer.h"


namespace tfu {
    namespace tf = tensorflow;

    tensorflow::Input InputFromTensorShape(const tensorflow::TensorShape &Shape) {
        using Type = tf::int32;

        tf::Tensor Tensor(tf::DataTypeToEnum<Type>::v(), {Shape.dims()});
        tf::int64  i = 0;
        for(const tf::TensorShapeDim Dimenstion : Shape) {
            Tensor.flat<Type>()(i) = Dimenstion.size;
            // LOG(INFO) << "Assigning to ith: " << i << " element in tensor value: " << Dimenstion.size;
            ++i;
        }
        tensorflow::Input Result(Tensor);
        return Result;
    }

    void TVariableInitializer::operator()(tensorflow::ClientSession &Session) const {
        for(const tensorflow::ops::Assign &Assign : Assigns) {
            std::vector<tf::Tensor> OutputsOfAssigning;
            TF_CHECK_OK(Session.Run({Assign}, &OutputsOfAssigning));
            LOG(INFO) << OutputsOfAssigning[0].DebugString();
            // LOG(INFO) << OutputsOfAssigning[0].matrix<double>();
            // LOG(INFO) << OutputsOfAssigning[0].vec<double>();
        }
    }
}
