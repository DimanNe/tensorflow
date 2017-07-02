#include "shape_helpers.h"


namespace tfu {
    namespace tf = tensorflow;

    tf::Input InputFromTensorShape(const tensorflow::TensorShape &Shape) {
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

    namespace {
        const char * const ATTR_NAME_SHAPE = "shape";
    }

    // tf::TensorShape GetShapeOfIthInput(const tensorflow::Output &Output, size_t i) {
    //     auto It = Output.op().node()->in_nodes().begin();
    //     const auto End = Output.op().node()->in_nodes().end();
    //     /// We can't use std::advance/std::next because they failed to implement an iterator:
    //     /// error: no type named 'difference_type' in 'std::iterator_traits<tensorflow::NeighborIter>'
    //     /// so, use for:
    //     for(size_t n = 0; n < i; ++n) {
    //         ++It;
    //         CHECK(It != End);
    //     }
    //
    //     const tf::Node *Node = *It;
    //     auto ShapeIt = Node->def().attr().find(ATTR_NAME_SHAPE);
    //     if(ShapeIt == Node->def().attr().end())
    //         return {};
    //
    //     CHECK(ShapeIt->second.has_shape());
    //     const tf::TensorShapeProto ProtoShape = ShapeIt->second.shape();
    //
    //     tf::TensorShape Result(ProtoShape);
    //     return Result;
    // }

    tensorflow::PartialTensorShape GetShapeOfOutput(const tensorflow::Output &Output) {
        const tf::Node *Node = Output.op().node();
        auto ShapeIt = Node->def().attr().find(ATTR_NAME_SHAPE);
        if(ShapeIt == Node->def().attr().end())
            LOG(INFO) << "asdf";

        CHECK(ShapeIt->second.has_shape());
        const tf::TensorShapeProto ProtoShape = ShapeIt->second.shape();

        tf::PartialTensorShape Result(ProtoShape);

        return Result;
    }
}
