#include <cstddef>
#include <stdexcept>

namespace impala
{

void* allocatePinnedMemory(std::size_t size)
{
	void* ptr;
	auto ret = cudaMallocHost(&ptr, size);
	if (ret != cudaSuccess) {
		throw std::runtime_error("cudaMallocHost failed");
	}
	return ptr;
}

void freePinnedMemory(void* ptr)
{
	auto ret = cudaFreeHost(ptr);
	if (ret != cudaSuccess) {
		throw std::runtime_error("cudaFreeHost failed");
	}
}

}  // namespace impala
