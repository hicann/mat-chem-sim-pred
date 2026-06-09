#include "register/op_def_registry.h"

namespace ops {

class FNOForward : public OpDef {
public:
    explicit FNOForward(const char* name) : OpDef(name) {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("weights")
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
        this->Attr("grid_dim").AttrType(REQUIRED).Int();
        this->Attr("in_channels").AttrType(REQUIRED).Int();
        this->Attr("out_channels").AttrType(REQUIRED).Int();
        this->Attr("hidden_channels").AttrType(REQUIRED).Int();
        this->Attr("modes").AttrType(REQUIRED).Int();

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FNOForward);

}  // namespace ops
