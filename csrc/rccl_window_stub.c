/* Provides ncclCommWindowRegister and ncclCommWindowDeregister so JAX 0.8+
 * wheels load against RCCL versions that predate NCCL/RCCL 2.27 (the LAIF
 * container ships RCCL 2.26.6, which is missing the symbols). LD_PRELOADed
 * via activate.sh.
 *
 * Background: NCCL/RCCL 2.27 added "symmetric memory windows" — an opt-in
 * registration where the same virtual address holds the same buffer on every
 * rank, enabling specialized low-latency kernels. Standard collectives
 * (ncclAllReduce, ncclAllGather, ncclReduceScatter, ncclBroadcast) do NOT
 * call WindowRegister themselves; the application has to opt in. JAX 0.9.1
 * pmap doesn't, so the stubs are never invoked in practice. activate.sh also
 * sets NCCL_WIN_ENABLE=0 to disable the feature inside RCCL altogether.
 *
 * If anything ever does call into the stub it returns ncclInternalError.
 *
 * References:
 *   https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/
 *   https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html
 */

typedef enum { ncclSuccess = 0, ncclUnhandledCudaError = 1, ncclSystemError = 2,
               ncclInternalError = 3, ncclInvalidArgument = 4,
               ncclInvalidUsage = 5, ncclRemoteError = 6, ncclInProgress = 7,
               ncclNumResults = 8 } ncclResult_t;

ncclResult_t ncclCommWindowRegister(void* comm, void* buff, unsigned long size,
                                    void** win, int winflags) {
    (void)comm; (void)buff; (void)size; (void)win; (void)winflags;
    return ncclInternalError;
}

ncclResult_t ncclCommWindowDeregister(void* comm, void* win) {
    (void)comm; (void)win;
    return ncclInternalError;
}
