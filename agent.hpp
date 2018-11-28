#pragma once

#include <cstdint>
#include <type_traits>

#include <range/v3/span.hpp>

#include "loss.hpp"

namespace impala
{

namespace detail
{

void dummyPredictCallback();

template <class Loss>
void dummyTrainCallback(Loss);

template <class T,
    std::enable_if_t<
        std::conjunction_v<
            IsLossType<typename T::Loss>,
            std::is_same<void, decltype(std::declval<T&>().predict(std::declval<std::add_lvalue_reference_t<typename T::ObsBatch>>(), std::declval<ranges::span<std::int64_t>>(), std::declval<ranges::span<float>>(), dummyPredictCallback))>,
            std::is_same<void, decltype(std::declval<T&>().train(std::declval<std::add_lvalue_reference_t<typename T::ObsBatch>>(), std::declval<ranges::span<std::int64_t>>(), std::declval<ranges::span<typename T::Reward>>(), std::declval<ranges::span<float>>(), std::declval<ranges::span<float>>(), std::declval<ranges::span<float>>(), std::declval<ranges::span<std::int64_t>>(), dummyTrainCallback<typename T::Loss>))>,
            std::is_same<void, decltype(std::declval<T&>().sync())>,
            std::is_same<void, decltype(std::declval<T&>().save(std::declval<std::int64_t>()))>,
            std::is_same<void, decltype(std::declval<T&>().load(std::declval<std::int64_t>()))>>,
        std::nullptr_t> = nullptr>
inline constexpr std::true_type isAgentHelper(const volatile T*);

inline constexpr std::false_type isAgentHelper(const volatile void*);

}  // namespace detail

template <class T>
struct IsAgent
    : public std::conditional_t<
          std::is_reference_v<T> || std::is_const_v<T> || std::is_volatile_v<T>,
          std::false_type,
          decltype(detail::isAgentHelper(std::declval<T*>()))>
{};

template <class T>
inline constexpr bool IsAgentV = IsAgent<T>::value;

}  // namespace impala
