#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <utility>

namespace impala
{

template <class T>
struct DiscreteActionTraits;

namespace detail
{

template <class T,
    std::enable_if_t<
        std::conjunction_v<
            std::is_default_constructible<T>,
            std::is_copy_constructible<T>,
            std::is_copy_assignable<T>,
            std::is_same<decltype(DiscreteActionTraits<T>::num_actions), const std::int64_t>,
            std::is_same<std::int64_t, decltype(DiscreteActionTraits<T>::convertToID(std::declval<T>()))>,
            std::is_same<T, decltype(DiscreteActionTraits<T>::convertFromID(std::declval<std::int64_t>()))>>,
        std::nullptr_t> = nullptr>
inline constexpr std::true_type isDiscreteActionHelper(const volatile T*);

inline constexpr std::false_type isDiscreteActionHelper(const volatile void*);

}  // namespace detail

template <class T>
struct IsDiscreteAction
    : public std::conditional_t<
          std::is_reference_v<T>,
          std::false_type,
          decltype(detail::isDiscreteActionHelper(std::declval<T*>()))>
{};

template <class T>
inline constexpr bool IsDiscreteActionV = IsDiscreteAction<T>::value;

template <class T, std::int64_t NumActions>
struct EnumActionTraits
{
	static inline constexpr std::int64_t num_actions = NumActions;
	static_assert(0 < num_actions);
	static std::int64_t convertToID(T action)
	{
		auto id = static_cast<std::underlying_type_t<T>>(action);
		assert(0 <= id && id < num_actions);
		return static_cast<std::int64_t>(id);
	}
	static T convertFromID(std::int64_t id)
	{
		assert(0 <= id && id < num_actions);
		return static_cast<T>(static_cast<std::underlying_type_t<T>>(id));
	}
};

template <class T>
struct TupleActionTraits;

template <class... Ts>
struct TupleActionTraits<std::tuple<Ts...>>
{
public:
	static_assert(sizeof...(Ts) > 0);
	static_assert((IsDiscreteActionV<Ts> && ...));

	using ActionType = std::tuple<Ts...>;

	static inline constexpr std::int64_t num_actions = (DiscreteActionTraits<Ts>::num_actions * ...);
	static std::int64_t convertToID(const ActionType& action)
	{
		return convertToIDHelper(action, std::make_index_sequence<sizeof...(Ts)>{});
	}
	static ActionType convertFromID(std::int64_t id)
	{
		assert(0 <= id && id < num_actions);
		return convertFromIDHelper(id, std::make_index_sequence<sizeof...(Ts)>{});
	}

private:
	template <std::size_t I>
	static constexpr std::int64_t strides()
	{
		return stridesHelper(std::make_index_sequence<I>{});
	}
	template <std::size_t... Is>
	static constexpr std::int64_t stridesHelper(std::index_sequence<Is...>)
	{
		return (1 * ... * DiscreteActionTraits<std::tuple_element_t<Is, ActionType>>::num_actions);
	}
	template <std::size_t... Is>
	static std::int64_t convertToIDHelper(const ActionType& action, std::index_sequence<Is...>)
	{
		return ((strides<Is>() * DiscreteActionTraits<std::tuple_element_t<Is, ActionType>>::convertToID(std::get<Is>(action))) + ...);
	}
	template <std::size_t... Is>
	static ActionType convertFromIDHelper(std::int64_t id, std::index_sequence<Is...>)
	{
		return std::make_tuple(DiscreteActionTraits<std::tuple_element_t<Is, ActionType>>::convertFromID((id / strides<Is>()) % DiscreteActionTraits<std::tuple_element_t<Is, ActionType>>::num_actions)...);
	}
};

enum class FourDirections : std::uint8_t
{
	UP,
	DOWN,
	LEFT,
	RIGHT,
};

template <>
struct DiscreteActionTraits<FourDirections> : public EnumActionTraits<FourDirections, 4>
{};

enum class FiveDirections : std::uint8_t
{
	NEUTRAL,
	UP,
	DOWN,
	LEFT,
	RIGHT,
};

template <>
struct DiscreteActionTraits<FiveDirections> : public EnumActionTraits<FiveDirections, 5>
{};

enum class EightDirections : std::uint8_t
{
	UP,
	DOWN,
	LEFT,
	RIGHT,
	UP_LEFT,
	UP_RIGHT,
	DOWN_LEFT,
	DOWN_RIGHT,
};

template <>
struct DiscreteActionTraits<EightDirections> : public EnumActionTraits<EightDirections, 8>
{};

enum class NineDirections : std::uint8_t
{
	NEUTRAL,
	UP,
	DOWN,
	LEFT,
	RIGHT,
	UP_LEFT,
	UP_RIGHT,
	DOWN_LEFT,
	DOWN_RIGHT,
};

template <>
struct DiscreteActionTraits<NineDirections> : public EnumActionTraits<NineDirections, 9>
{};

enum class AtariButton : std::uint8_t
{
	NONE,
	FIRE
};

template <>
struct DiscreteActionTraits<AtariButton> : public EnumActionTraits<AtariButton, 2>
{};

using AtariAction = std::tuple<NineDirections, AtariButton>;

template <>
class DiscreteActionTraits<AtariAction> : public TupleActionTraits<AtariAction>
{};

}  // namespace impala
