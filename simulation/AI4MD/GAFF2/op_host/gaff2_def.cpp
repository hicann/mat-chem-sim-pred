/**
 * gaff2_def.cpp
 *
 * GAFF2 Force Field — Operator Definition
 *
 * Standard Ascend C operator definition for CANN/ACL integration.
 * Provides operator registration metadata for toolchain compatibility.
 */

#include <cstdint>
#include <cstdio>

#include "acl/acl.h"
#include "gaff2_types.h"

// Operator registration info
static constexpr const char* GAFF2_OP_NAME = "gaff2_compute_forces";
static constexpr uint32_t GAFF2_OP_VERSION = 1;

// ============================================================
// Operator initialization (called once at system init)
// ============================================================
extern "C" int32_t gaff2_force_op_init(void) {
    fprintf(stdout, "[GAFF2] Initializing operator: %s v%u\n",
            GAFF2_OP_NAME, GAFF2_OP_VERSION);
    return 0;
}

// ============================================================
// Operator teardown
// ============================================================
extern "C" int32_t gaff2_force_op_finalize(void) {
    fprintf(stdout, "[GAFF2] Finalizing operator: %s\n", GAFF2_OP_NAME);
    return 0;
}
