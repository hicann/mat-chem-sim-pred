#include "register/op_def_registry.h"

namespace ops {

class MeshGraphNet : public OpDef {
public:
    explicit MeshGraphNet(const char* name) : OpDef(name) {
        this->Input("node_features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("edge_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("edge_features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("node_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("edge_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("num_nodes").AttrType(REQUIRED).Int();
        this->Attr("num_edges").AttrType(REQUIRED).Int();
        this->Attr("node_dim").AttrType(REQUIRED).Int();
        this->Attr("edge_dim").AttrType(REQUIRED).Int();
        this->Attr("hidden_dim").AttrType(REQUIRED).Int();
        this->Attr("output_dim").AttrType(REQUIRED).Int();
        this->Attr("max_neighbors").AttrType(REQUIRED).Int();

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MeshGraphNet);

}  // namespace ops
