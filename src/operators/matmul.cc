#include "operators/matmul.h"
#include "utils/operator_utils.h"
#include <numeric>

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto shapeA_ = inputs[0]->getDims();
        auto shapeB_ = inputs[1]->getDims();
        auto rankA_ = inputs[0]->getRank();
        auto rankB_ = inputs[1]->getRank();

        Shape shapeA1_ = Shape(shapeA_.begin(), shapeA_.begin() + rankA_ - 2);
        Shape shapeB1_ = Shape(shapeB_.begin(), shapeB_.begin() + rankB_ - 2);
        Shape broadcast_shape_ = infer_broadcast(shapeA1_, shapeB1_);
        
        m = *(transA ? shapeA_.rbegin() : shapeA_.rbegin() + 1);
        n = *(transB ? shapeB_.rbegin() + 1 : shapeB_.rbegin());

        broadcast_shape_.push_back(m);
        broadcast_shape_.push_back(n);
        return {{broadcast_shape_}};
    }

} // namespace infini