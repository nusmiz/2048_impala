#pragma once

#include <iostream>
#include <type_traits>

namespace impala
{

namespace detail
{

template <class T,
    std::enable_if_t<
        std::conjunction_v<
            std::is_default_constructible<T>,
            std::is_copy_constructible<T>,
            std::is_copy_assignable<T>,
            std::is_same<T, decltype(exponentialMovingAverage(std::declval<T>(), std::declval<T>(), std::declval<double>()))>,
            std::is_convertible<decltype(std::declval<std::ostream&>() << std::declval<T>()), std::ostream&>>,
        std::nullptr_t> = nullptr>
inline constexpr std::true_type isLossTypeHelper(const volatile T*);

inline constexpr std::false_type isLossTypeHelper(const volatile void*);

}  // namespace detail

template <class T>
struct IsLossType
    : public std::conditional_t<
          std::is_reference_v<T>,
          std::false_type,
          decltype(detail::isLossTypeHelper(std::declval<T*>()))>
{};

template <class T>
inline constexpr bool IsLossTypeV = IsLossType<T>::value;

struct A3CLoss
{
	double v_loss;
	double pi_loss;
	double entropy_loss;
};

inline A3CLoss exponentialMovingAverage(A3CLoss current_average, A3CLoss new_loss, double decay)
{
	auto average_v_loss = decay * current_average.v_loss + (1.0 - decay) * new_loss.v_loss;
	auto average_pi_loss = decay * current_average.pi_loss + (1.0 - decay) * new_loss.pi_loss;
	auto average_entropy_loss = decay * current_average.entropy_loss + (1.0 - decay) * new_loss.entropy_loss;
	return A3CLoss{average_v_loss, average_pi_loss, average_entropy_loss};
}

inline std::ostream& operator<<(std::ostream& os, const A3CLoss& loss)
{
	os << loss.v_loss << ' ' << loss.pi_loss << ' ' << loss.entropy_loss;
	return os;
}

}  // namespace impala
