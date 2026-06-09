#include "register/op_def_registry.h"

namespace ops {

class PinnFC : public OpDef {
public:
    explicit PinnFC(const char* name) : OpDef(name) {
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

        this->Output("gradient")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("batch_size").AttrType(REQUIRED).Int();
        this->Attr("input_dim").AttrType(REQUIRED).Int();
        this->Attr("hidden_dim").AttrType(REQUIRED).Int();
        this->Attr("output_dim").AttrType(REQUIRED).Int();
        this->Attr("num_layers").AttrType(REQUIRED).Int();
        this->Attr("activation_type").AttrType(REQUIRED).Int();

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(PinnFC);

}  // namespace ops
