#pragma once

#include <algorithm>
#include <cassert>
#include <string>
#include <utility>

#include <boost/container/vector.hpp>
#include <boost/filesystem.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <range/v3/span.hpp>

#include "cuda/cuda_util.hpp"
#include "tensor.hpp"

namespace impala
{

class PythonInitializer
{
public:
	PythonInitializer(bool init_signal_handler = true)
	{
		assert(!::Py_IsInitialized());
		::Py_InitializeEx(init_signal_handler ? 1 : 0);
		boost::python::numpy::initialize();
	}
	~PythonInitializer()
	{
		::Py_FinalizeEx();
	}
};

inline boost::python::object makePythonMainNameSpace()
{
	auto main_ns = boost::python::import("__main__").attr("__dict__");
	boost::python::exec("import sys", main_ns);
	boost::python::exec(("sys.path.append(\"" + boost::filesystem::canonical(".").string() + "\")").data(), main_ns);
	return main_ns;
}

namespace detail
{

template <std::size_t Current, std::size_t M, std::size_t... Ms, std::size_t... Strides, class... SizeT>
inline boost::python::tuple stridesOfNdArrayHelper(std::index_sequence<M, Ms...>, std::index_sequence<Strides...>, SizeT... prefix)
{
	if constexpr (sizeof...(Ms) == 0) {
		return boost::python::make_tuple(static_cast<int>(prefix)..., static_cast<int>(Strides)..., static_cast<int>(Current / M));
	} else {
		return stridesOfNdArrayHelper<Current / M>(std::index_sequence<Ms...>{}, std::index_sequence<Strides..., Current / M>{}, prefix...);
	}
}

struct NdArrayTraitsPlaceHolder
{};

template <std::size_t size_of_all, std::size_t... Ms, class... SizeTOrPlaceHolder>
inline boost::python::tuple stridesOfNdArrayHelper2(std::size_t current, std::size_t batch_size, SizeTOrPlaceHolder... data)
{
	return stridesOfNdArrayHelper2<size_of_all, Ms...>(current / batch_size, data..., current / batch_size);
}

template <std::size_t size_of_all, std::size_t... Ms, class... SizeTOrPlaceHolder>
inline boost::python::tuple stridesOfNdArrayHelper2([[maybe_unused]] std::size_t current, NdArrayTraitsPlaceHolder, SizeTOrPlaceHolder... data)
{
	assert(current == size_of_all);
	return stridesOfNdArrayHelper<size_of_all>(std::index_sequence<Ms...>{}, std::index_sequence<>{}, data...);
}

}  // namespace detail

template <class T, std::size_t... Ns>
class NdArrayTraits
{
public:
	using value_type = T;
	using BufferType = PinnedMemoryVector<T>;
	using TensorType = Tensor<T, Ns...>;
	using TensorRefType = TensorRef<T, Ns...>;

	static inline constexpr std::size_t size_of_all = (Ns * ...);

	static boost::python::tuple shapeOfNdArray()
	{
		return boost::python::make_tuple(static_cast<int>(Ns)...);
	}
	template <class... SizeT, std::enable_if_t<std::conjunction_v<std::is_convertible<SizeT, std::size_t>...>, std::nullptr_t> = nullptr>
	static boost::python::tuple shapeOfBatchedNdArray(SizeT... batch_sizes)
	{
		return boost::python::make_tuple(static_cast<int>(batch_sizes)..., static_cast<int>(Ns)...);
	}
	static boost::python::tuple stridesOfNdArray()
	{
		return detail::stridesOfNdArrayHelper<sizeof(T) * size_of_all>(std::index_sequence<Ns...>{}, std::index_sequence<>{});
	}
	template <class... SizeT, std::enable_if_t<std::conjunction_v<std::is_convertible<SizeT, std::size_t>...>, std::nullptr_t> = nullptr>
	static boost::python::tuple stridesOfBatchedNdArray(SizeT... batch_sizes)
	{
		return detail::stridesOfNdArrayHelper2<sizeof(T) * size_of_all, Ns...>((batch_sizes * ... * (sizeof(T) * size_of_all)), batch_sizes..., detail::NdArrayTraitsPlaceHolder{});
	}

	static boost::python::numpy::ndarray convertToNdArray(Tensor<T, Ns...>& tensor)
	{
		namespace np = boost::python::numpy;
		return np::from_data(tensor.data(), np::dtype::get_builtin<T>(), shapeOfNdArray(), stridesOfNdArray(), boost::python::object());
	}

	template <class ForwardIterator,
	    std::enable_if_t<
	        std::conjunction_v<
	            std::is_base_of<std::forward_iterator_tag, typename std::iterator_traits<ForwardIterator>::iterator_category>,
	            std::is_convertible<typename std::iterator_traits<ForwardIterator>::reference, const Tensor<T, Ns...>&>>,
	        std::nullptr_t> = nullptr>
	static void makeBufferForBatch(ForwardIterator first, ForwardIterator last, BufferType& buffer)
	{
		const auto batch_size = static_cast<std::size_t>(std::distance(first, last));
		buffer.resize(batch_size * size_of_all, boost::container::default_init);
		auto dest = buffer.begin();
		for (; first != last; ++first) {
			if constexpr (std::is_convertible_v<typename std::iterator_traits<ForwardIterator>::reference, const Tensor<T, Ns...>&>) {
				const Tensor<T, Ns...>& src = *first;
				std::copy_n(src.data(), size_of_all, dest);
			} else {
				const std::optional<Tensor<T, Ns...>>& src = *first;
				if (src.has_value()) {
					std::copy_n(src.value().data(), size_of_all, dest);
				}
			}
			dest += size_of_all;
		}
	}
	template <class ForwardIterator, class Callback,
	    std::enable_if_t<
	        std::conjunction_v<
	            std::is_base_of<std::forward_iterator_tag, typename std::iterator_traits<ForwardIterator>::iterator_category>,
	            std::is_invocable<Callback, typename std::iterator_traits<ForwardIterator>::reference, TensorRef<T, Ns...>&>>,
	        std::nullptr_t> = nullptr>
	static void makeBufferForBatch(ForwardIterator first, ForwardIterator last, BufferType& buffer, Callback&& callback)
	{
		const auto batch_size = static_cast<std::size_t>(std::distance(first, last));
		buffer.resize(batch_size * size_of_all, boost::container::default_init);
		auto dest = buffer.data();
		for (; first != last; ++first) {
			TensorRef<T, Ns...> tensor_ref{dest};
			std::invoke(callback, *first, tensor_ref);
			dest += size_of_all;
		}
	}

	// 返り値のndarrayはspanの元となったメモリ領域を直接参照するため、lifetimeに注意
	template <class... SizeT, std::enable_if_t<std::conjunction_v<std::is_convertible<SizeT, std::size_t>...>, std::nullptr_t> = nullptr>
	static boost::python::numpy::ndarray convertToBatchedNdArray(ranges::span<T> buffer, SizeT... batch_sizes)
	{
		assert(static_cast<std::size_t>(buffer.size()) == (batch_sizes * ... * size_of_all));
		namespace np = boost::python::numpy;
		return np::from_data(buffer.data(), np::dtype::get_builtin<T>(), shapeOfBatchedNdArray(batch_sizes...), stridesOfBatchedNdArray(batch_sizes...), boost::python::object());
	}
};


}  // namespace impala
