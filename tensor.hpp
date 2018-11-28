#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include <boost/container/vector.hpp>

namespace impala
{

template <class T, std::size_t N, std::size_t... Ns>
class Tensor;
template <class T, std::size_t N, std::size_t... Ns>
class StaticTensor;
template <class T, std::size_t N, std::size_t... Ns>
class TensorRef;

template <class T, std::size_t... Ns>
class TensorRefIterator
{
public:
	using iterator_category = std::input_iterator_tag;
	using value_type = TensorRef<T, Ns...>;
	using difference_type = std::ptrdiff_t;
	using pointer = value_type*;
	using reference = value_type&;

	TensorRefIterator() = default;
	constexpr explicit TensorRefIterator(T* ptr) noexcept : m_ptr{ptr} {}

	constexpr value_type operator*() const noexcept
	{
		return value_type{m_ptr};
	}

private:
	class TensorRefIteratorProxy
	{
	public:
		constexpr explicit TensorRefIteratorProxy(T* ptr) noexcept : m_value{ptr} {}

		constexpr std::add_const_t<value_type>* operator->() const noexcept
		{
			return &m_value;
		}

	private:
		value_type m_value;
	};

public:
	constexpr TensorRefIteratorProxy operator->() const noexcept
	{
		return TensorRefIteratorProxy{m_ptr};
	}
	constexpr TensorRefIterator& operator++() noexcept
	{
		m_ptr += static_cast<difference_type>((Ns * ...));
		return *this;
	}
	constexpr TensorRefIterator operator++(int) noexcept
	{
		const auto temp = *this;
		++(*this);
		return temp;
	}
	constexpr TensorRefIterator& operator--() noexcept
	{
		m_ptr -= static_cast<difference_type>((Ns * ...));
		return *this;
	}
	constexpr TensorRefIterator operator--(int) noexcept
	{
		const auto temp = *this;
		--(*this);
		return temp;
	}
	constexpr TensorRefIterator& operator+=(difference_type diff) noexcept
	{
		m_ptr += static_cast<difference_type>((Ns * ...)) * diff;
		return *this;
	}
	constexpr TensorRefIterator operator+(difference_type diff) const noexcept
	{
		auto temp = *this;
		temp += diff;
		return temp;
	}
	friend constexpr TensorRefIterator operator+(difference_type diff, const TensorRefIterator& itr) noexcept
	{
		return itr + diff;
	}
	constexpr TensorRefIterator& operator-=(difference_type diff) noexcept
	{
		m_ptr -= static_cast<difference_type>((Ns * ...)) * diff;
		return *this;
	}
	constexpr TensorRefIterator operator-(difference_type diff) noexcept
	{
		auto temp = *this;
		temp -= diff;
		return temp;
	}
	constexpr difference_type operator-(const TensorRefIterator& other) noexcept
	{
		auto diff = m_ptr - other.m_ptr;
		assert(diff % static_cast<difference_type>((Ns * ...)) == 0);
		return diff / static_cast<difference_type>((Ns * ...));
	}
	constexpr value_type operator[](difference_type n) const noexcept
	{
		return value_type{m_ptr + static_cast<difference_type>((Ns * ...)) * n};
	}
	bool operator==(const TensorRefIterator& other) const noexcept
	{
		return m_ptr == other.m_ptr;
	}
	bool operator!=(const TensorRefIterator& other) const noexcept
	{
		return m_ptr != other.m_ptr;
	}
	bool operator<(const TensorRefIterator& other) const noexcept
	{
		return m_ptr < other.m_ptr;
	}
	bool operator<=(const TensorRefIterator& other) const noexcept
	{
		return m_ptr <= other.m_ptr;
	}
	bool operator>(const TensorRefIterator& other) const noexcept
	{
		return m_ptr > other.m_ptr;
	}
	bool operator>=(const TensorRefIterator& other) const noexcept
	{
		return m_ptr >= other.m_ptr;
	}

private:
	T* m_ptr;
};

namespace detail
{

template <class T, std::size_t N, std::size_t... Ns>
struct TensorTraits
{
	using iterator = TensorRefIterator<T, Ns...>;
	using reference = TensorRef<T, Ns...>;

	static constexpr iterator makeIterator(T* ptr) noexcept
	{
		return iterator{ptr};
	}
	static constexpr reference makeReference(T* ptr) noexcept
	{
		return reference{ptr};
	}
};

template <class T, std::size_t N>
struct TensorTraits<T, N>
{
	using iterator = T*;
	using reference = T&;

	static constexpr iterator makeIterator(T* ptr) noexcept
	{
		return ptr;
	}
	static constexpr reference makeReference(T* ptr) noexcept
	{
		return *ptr;
	}
};

}  // namespace detail

template <class T, std::size_t N, std::size_t... Ns>
class TensorRef
{
public:
	using Traits = detail::TensorTraits<T, N, Ns...>;
	using iterator = typename Traits::iterator;
	using reference = typename Traits::reference;

	constexpr explicit TensorRef(T* data) noexcept : m_data{data} {}

	template <class U, std::enable_if_t<std::is_same_v<std::add_const_t<U>, T>, std::nullptr_t> = nullptr>
	constexpr TensorRef(TensorRef<U, N, Ns...> other) noexcept : m_data{other.data()}
	{
	}

	constexpr T* data() const noexcept
	{
		return m_data;
	}
	constexpr std::size_t size() const noexcept
	{
		return N;
	}
	constexpr std::size_t sizeOfAll() const noexcept
	{
		return (N * ... * Ns);
	}
	constexpr reference operator[](std::size_t n) const noexcept
	{
		assert(n < size());
		return Traits::makeReference(data() + n * (1 * ... * Ns));
	}
	constexpr iterator begin() const noexcept
	{
		return Traits::makeIterator(data());
	}
	constexpr iterator end() const noexcept
	{
		return Traits::makeIterator(data() + (N * ... * Ns));
	}
	constexpr iterator cbegin() const noexcept
	{
		return begin();
	}
	constexpr iterator cend() const noexcept
	{
		return end();
	}
	constexpr TensorRef<std::add_const_t<T>, N, Ns...> toConstRef() const noexcept
	{
		return TensorRef<std::add_const_t<T>, N, Ns...>{m_data};
	}

	TensorRef& assign(const Tensor<T, N, Ns...>& src)
	{
		std::copy_n(src.data(), sizeOfAll(), m_data);
		return *this;
	}
	TensorRef& assign(const StaticTensor<T, N, Ns...>& src)
	{
		std::copy_n(src.data(), sizeOfAll(), m_data);
		return *this;
	}
	TensorRef& assign(TensorRef<std::add_const_t<T>, N, Ns...> src)
	{
		std::copy_n(src.data(), sizeOfAll(), m_data);
		return *this;
	}

private:
	T* m_data;
};

namespace detail
{

template <class ContainerType, class T, std::size_t N, std::size_t... Ns>
class TensorImpl
{
public:
	static_assert(!std::is_const_v<T>);
	static_assert(((N > 0) && ... && (Ns > 0)));

	using Traits = detail::TensorTraits<T, N, Ns...>;
	using ConstTraits = detail::TensorTraits<std::add_const_t<T>, N, Ns...>;
	using iterator = typename Traits::iterator;
	using const_iterator = typename ConstTraits::iterator;
	using reference = typename Traits::reference;
	using const_reference = typename ConstTraits::reference;

	T* data() noexcept
	{
		return m_data.data();
	}
	std::add_const_t<T>* data() const noexcept
	{
		return m_data.data();
	}
	constexpr std::size_t size() const noexcept
	{
		return N;
	}
	constexpr std::size_t sizeOfAll() const noexcept
	{
		return (N * ... * Ns);
	}
	reference operator[](std::size_t n) noexcept
	{
		assert(n < size());
		return Traits::makeReference(data() + n * (1 * ... * Ns));
	}
	const_reference operator[](std::size_t n) const noexcept
	{
		assert(n < size());
		return ConstTraits::makeReference(data() + n * (1 * ... * Ns));
	}
	iterator begin() noexcept
	{
		return Traits::makeIterator(data());
	}
	iterator end() noexcept
	{
		return Traits::makeIterator(data() + (N * ... * Ns));
	}
	const_iterator begin() const noexcept
	{
		return ConstTraits::makeIterator(data());
	}
	const_iterator end() const noexcept
	{
		return ConstTraits::makeIterator(data() + (N * ... * Ns));
	}
	const_iterator cbegin() const noexcept
	{
		return begin();
	}
	const_iterator cend() const noexcept
	{
		return end();
	}

	TensorRef<T, N, Ns...> ref() noexcept
	{
		return TensorRef<T, N, Ns...>(data());
	}
	TensorRef<std::add_const_t<T>, N, Ns...> cref() noexcept
	{
		return TensorRef<std::add_const_t<T>, N, Ns...>(data());
	}

	bool operator==(const TensorImpl& other) const
	{
		return m_data == other.m_data;
	}
	bool operator!=(const TensorImpl& other) const
	{
		return m_data != other.m_data;
	}

private:
	friend class Tensor<T, N, Ns...>;
	friend class StaticTensor<T, N, Ns...>;

	ContainerType m_data;
};

}  // namespace detail

template <class T, std::size_t N, std::size_t... Ns>
class Tensor : public detail::TensorImpl<boost::container::vector<T>, T, N, Ns...>
{
public:
	Tensor()
	{
		this->m_data.resize((N * ... * Ns), boost::container::default_init);
	}

private:
	Tensor(const Tensor&) = default;

public:
	Tensor(Tensor&&) = default;
	Tensor& operator=(const Tensor&) = delete;
	Tensor& operator=(Tensor&&) = default;

	Tensor clone()
	{
		return Tensor(*this);
	}
	Tensor& assign(const Tensor& src)
	{
		this->m_data = src.m_data;
		return *this;
	}
	Tensor& assign(Tensor&& src)
	{
		this->m_data = std::move(src.m_data);
		return *this;
	}
	Tensor& assign(const StaticTensor<T, N, Ns...>& src)
	{
		this->m_data.assign(src.data(), src.data() + src.sizeOfAll());
		return *this;
	}
	Tensor& assign(TensorRef<std::add_const_t<T>, N, Ns...> src)
	{
		this->m_data.assign(src.data(), src.data() + src.sizeOfAll());
		return *this;
	}
};

template <class T, std::size_t N, std::size_t... Ns>
class StaticTensor : public detail::TensorImpl<std::array<T, (N * ... * Ns)>, T, N, Ns...>
{
public:
	StaticTensor clone()
	{
		return *this;
	}
	StaticTensor& assign(const StaticTensor& src)
	{
		*this = src;
		return *this;
	}
	StaticTensor& assign(const Tensor<T, N, Ns...>& src)
	{
		std::copy_n(src.data(), this->sizeOfAll(), this->data());
		return *this;
	}
	StaticTensor& assign(TensorRef<std::add_const_t<T>, N, Ns...> src)
	{
		std::copy_n(src.data(), this->sizeOfAll(), this->data());
		return *this;
	}
};

}  // namespace impala
