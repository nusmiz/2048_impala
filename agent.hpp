#pragma once

#include <cstdint>
#include <type_traits>

#include <range/v3/span.hpp>

#include "environment.hpp"
#include "loss.hpp"

namespace impala
{

namespace detail
{

void dummyPredictCallback();

template <class Loss>
void dummyTrainCallback(Loss);

template <class T, class Environment,
    std::enable_if_t<
        std::conjunction_v<
            IsEnvironment<Environment>,
            IsLossType<typename T::Loss>,
            std::is_same<void, decltype(std::declval<T&>().template predict<DiscreteActionTraits<typename Environment::Action>::num_actions>(std::declval<std::add_lvalue_reference_t<typename Environment::ObsBatch>>(), std::declval<ranges::span<float>>(), dummyPredictCallback))>,
            std::is_same<void, decltype(std::declval<T&>().train(std::declval<std::add_lvalue_reference_t<typename Environment::ObsBatch>>(), std::declval<ranges::span<std::int64_t>>(), std::declval<ranges::span<typename Environment::Reward>>(), std::declval<ranges::span<float>>(), std::declval<ranges::span<float>>(), std::declval<ranges::span<float>>(), std::declval<ranges::span<std::int64_t>>(), dummyTrainCallback<typename T::Loss>))>,
            std::is_same<void, decltype(std::declval<T&>().sync())>,
            std::is_same<void, decltype(std::declval<T&>().save(std::declval<std::int64_t>()))>,
            std::is_same<void, decltype(std::declval<T&>().load(std::declval<std::int64_t>()))>>,
        std::nullptr_t> = nullptr>
inline constexpr std::true_type isAgentForGivenEnvironmentHelper(const volatile T*, const volatile Environment*);

inline constexpr std::false_type isAgentForGivenEnvironmentHelper(const volatile void*);

}  // namespace detail

template <class T, class Environment>
struct IsAgentForGivenEnvironment
    : public std::conditional_t<
          std::is_reference_v<T> || std::is_const_v<T> || std::is_volatile_v<T>,
          std::false_type,
          decltype(detail::isAgentForGivenEnvironmentHelper(std::declval<T*>(), std::declval<Environment*>()))>
{};

template <class T, class Environment>
inline constexpr bool IsAgentForGivenEnvironmentV = IsAgentForGivenEnvironment<T, Environment>::value;

}  // namespace impala
