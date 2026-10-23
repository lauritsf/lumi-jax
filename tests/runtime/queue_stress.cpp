#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP(call) do { auto e = (call); if (e != hipSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(1); } } while (0)

__global__ void queue_tick(int* value) { ++*value; }

int main() {
  constexpr int streams = 100, iterations = 100;
  hipStream_t queues[streams];
  int *device, result[streams];
  HIP(hipMalloc(&device, sizeof(result)));
  HIP(hipMemset(device, 0, sizeof(result)));
  for (auto& queue : queues) HIP(hipStreamCreate(&queue));
  for (int i = 0; i < iterations; ++i)
    for (int s = 0; s < streams; ++s)
      queue_tick<<<1, 1, 0, queues[s]>>>(device + s);
  HIP(hipGetLastError());
  HIP(hipDeviceSynchronize());
  HIP(hipMemcpy(result, device, sizeof(result), hipMemcpyDeviceToHost));
  for (int value : result) if (value != iterations) return 2;
  for (auto queue : queues) HIP(hipStreamDestroy(queue));
  HIP(hipFree(device));
  std::puts("QUEUE STRESS OK: 10000 dispatches, 100 streams");
}
