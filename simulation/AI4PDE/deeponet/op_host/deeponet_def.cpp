#include "register/op_def_registry.h"

namespace ops {

class DeepOnetForward : public OpDef {
public:
    explicit DeepOnetForward(const char* name) : OpDef(name) {
        this->Input("branch_input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("trunk_input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("branch_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("trunk_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("batch_size").AttrType(REQUIRED).Int();
        this->Attr("branch_dim").AttrType(REQUIRED).Int();
        this->Attr("trunk_dim").AttrType(REQUIRED).Int();
        this->Attr("latent_dim").AttrType(REQUIRED).Int();
        this->Attr("query_size").AttrType(REQUIRED).Int();

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(DeepOnetForward);

}  // namespace ops
