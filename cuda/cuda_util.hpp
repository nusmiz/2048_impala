#pragma once

#include <cstddef>

#include <boost/container/vector.hpp>

namespace impala
{

#ifdef IMPALA_USE_CUDA
constexpr bool USE_CUDA = true;

void* allocatePinnedMemory(std::size_t size);
void freePinnedMemory(void* ptr);

template <class T>
class PinnedMemoryAllocator
{
public:
	using value_type = T;

	PinnedMemoryAllocator() = default;

	template <class U>
	PinnedMemoryAllocator(const PinnedMemoryAllocator<U>&) noexcept
	{}

	value_type* allocate(std::size_t n)
	{
		return reinterpret_cast<value_type*>(allocatePinnedMemory(sizeof(value_type) * n));
	}

	void deallocate(T* ptr, std::size_t)
	{
		freePinnedMemory(ptr);
	}

	bool operator==(const PinnedMemoryAllocator&) const noexcept
	{
		return true;
	}
	bool operator!=(const PinnedMemoryAllocator&) const noexcept
	{
		return false;
	}
};

template <class T>
using PinnedMemoryVector = boost::container::vector<T, PinnedMemoryAllocator<T>>;

#else

constexpr bool USE_CUDA = false;

template <class T>
using PinnedMemoryVector = boost::container::vector<T>;

#endif

}  // namespace impala
