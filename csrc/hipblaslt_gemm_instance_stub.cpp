/* Provides hipblaslt_ext::GemmInstance::setMaxWorkspaceBytes(size_t) so
 * jax-rocm7-pjrt 0.9.2+ can be used on a ROCm 7.0.2 base.
 *
 * THE MISSING SYMBOL
 *
 * Exactly one of the 33 hipblaslt_ext symbols the 0.10/0.11 PJRT plugin
 * references is absent from ROCm 7.0.2: setMaxWorkspaceBytes is declared on
 * GemmPreference at 7.0.2 but only gained on GemmInstance in ROCm 7.1.
 *
 * Whether that is fatal depends on the plugin's link flags, not on ROCm:
 *
 *   0.9.1.post5/post6   BIND_NOW, does not reference it   -> fine
 *   0.9.2 - 0.10.1      BIND_NOW, references it           -> fails at load
 *   0.10.2, 0.11.0      LAZY,     references it           -> loads and runs
 *
 * So 0.11.0 works on 7.0.2 until something calls this method, at which point
 * the process dies with a symbol lookup error -- possibly hours in. A jax
 * 0.11.0 image on this base therefore needs the symbol to exist.
 *
 * A STUB RATHER THAN THE REAL LIBRARY
 *
 * ROCm 7.1's libhipblaslt is not droppable in as a leaf upgrade the way RCCL is:
 * it needs 7.1's librocroller too (7 unresolved rocRoller::Expression /
 * CommandKernel / Command symbols against the 7.0.2 rocRoller), and with both
 * libraries swapped in a jax program that takes 90 s on the stock image does not
 * produce output in 20 minutes.
 *
 * THE COST
 *
 * The symbol is a no-op here, so the workspace cap XLA asks for is NOT applied
 * and hipBLASLt keeps its own default. That is a behavioural difference, not a
 * numerical one: it can only affect algorithm selection and how much scratch
 * hipBLASLt allocates.
 *
 * No probed workload reaches this symbol (dot, +bias, +gelu, +relu, batched
 * einsum, mixed-shape; f32/bf16/f16; autotuning on). If the warning ever
 * appears, that evidence is wrong and the fix is a ROCm 7.1+ base, where the
 * symbol exists natively, rather than a stub. Do not delete the warning to
 * quiet a log.
 */

#include <cstdio>
#include <cstdlib>

namespace {
bool warned = false;
}

extern "C" void _ZN13hipblaslt_ext12GemmInstance20setMaxWorkspaceBytesEm(void *self,
                                                                        unsigned long bytes) {
    (void)self;
    if (!warned) {
        warned = true;
        std::fprintf(stderr,
                     "[lumi-jax] WARNING: hipblaslt_ext::GemmInstance::setMaxWorkspaceBytes(%lu) "
                     "was called and is a NO-OP stub (ROCm 7.0.2 lacks the real symbol; see "
                     "csrc/hipblaslt_gemm_instance_stub.cpp). The workspace cap was NOT applied. "
                     "This path was believed unreachable -- please report it.\n",
                     bytes);
        std::fflush(stderr);
    }
}
