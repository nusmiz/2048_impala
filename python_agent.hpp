#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <type_traits>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <range/v3/span.hpp>
#include <range/v3/view/zip.hpp>

#include "agent.hpp"
#include "loss.hpp"
#include "python_util.hpp"

namespace impala
{

namespace detail
{

template <class T,
    std::enable_if_t<
        std::conjunction_v<
            IsLossType<typename T::Loss>,
            std::is_same<boost::python::object, decltype(T::create(std::declval<boost::python::object&>()))>,
            std::is_same<boost::python::object, decltype(T::convertObsBatch(std::declval<std::add_lvalue_reference_t<typename T::ObsBatch>>(), std::declval<std::size_t>()))>,
            std::is_same<boost::python::object, decltype(T::convertObsBatch(std::declval<std::add_lvalue_reference_t<typename T::ObsBatch>>(), std::declval<std::size_t>(), std::declval<std::size_t>()))>,
            std::is_same<boost::python::object, decltype(T::convertRewardBatch(std::declval<ranges::span<typename T::Reward>>(), std::declval<std::size_t>(), std::declval<std::size_t>()))>,
            std::is_same<typename T::Loss, decltype(T::convertToLoss(std::declval<boost::python::object&&>()))>>,
        std::nullptr_t> = nullptr>
inline constexpr std::true_type isPythonAgentTraitsHelper(const volatile T*);

inline constexpr std::false_type isPythonAgentTraitsHelper(const volatile void*);

}  // namespace detail

template <class T>
struct IsPythonAgentTraits
    : public std::conditional_t<
          std::is_reference_v<T> || std::is_const_v<T> || std::is_volatile_v<T>,
          std::false_type,
          decltype(detail::isPythonAgentTraitsHelper(std::declval<T*>()))>
{};

template <class T>
inline constexpr bool IsPythonAgentTraitsV = IsPythonAgentTraits<T>::value;


template <class PythonAgentTraits>
class PythonAgent
{
public:
	static_assert(IsPythonAgentTraitsV<PythonAgentTraits>);

	using Loss = typename PythonAgentTraits::Loss;
	using Reward = typename PythonAgentTraits::Reward;
	using ObsBatch = typename PythonAgentTraits::ObsBatch;

	PythonAgent()
	{
		try {
			m_python_main_ns = makePythonMainNameSpace();
			m_agent_object = PythonAgentTraits::create(m_python_main_ns);
			m_predict_func = m_agent_object.attr("predict");
			m_train_func = m_agent_object.attr("train");
			m_sync_func = m_agent_object.attr("sync");
			m_save_func = m_agent_object.attr("save_model");
			m_load_func = m_agent_object.attr("load_model");
		} catch (boost::python::error_already_set) {
			::PyErr_Print();
			std::terminate();
		}
	}

	template <class Callback, std::enable_if_t<std::is_invocable_v<Callback>, std::nullptr_t> = nullptr>
	void predict(ObsBatch& states, ranges::span<std::int64_t> action_id_buffer, ranges::span<float> policy_buffer, Callback&& callback)
	{
		auto prev_callback = std::move(m_callback);
		m_callback = [callback = std::move(callback)](boost::python::object&&) {
			callback();
		};
		try {
			const auto batch_size = static_cast<std::size_t>(action_id_buffer.size());
			auto states_pyobj = PythonAgentTraits::convertObsBatch(states, batch_size);
			auto action_id_buffer_ndarray = NdArrayTraits<std::int64_t, 1>::convertToBatchedNdArray(action_id_buffer, batch_size);
			auto policy_buffer_ndarray = NdArrayTraits<float, 1>::convertToBatchedNdArray(policy_buffer, batch_size);
			auto result = m_predict_func(states_pyobj, action_id_buffer_ndarray, policy_buffer_ndarray);
			if (prev_callback) {
				prev_callback(std::move(result));
			}
		} catch (boost::python::error_already_set) {
			::PyErr_Print();
			std::terminate();
		}
	}
	template <class Callback, std::enable_if_t<std::is_invocable_v<Callback, Loss>, std::nullptr_t> = nullptr>
	void train(ObsBatch& states, ranges::span<std::int64_t> action_ids, ranges::span<Reward> rewards, ranges::span<float> behaviour_policies, ranges::span<float> discounts, ranges::span<float> loss_coefs, ranges::span<std::int64_t> data_sizes, Callback&& callback)
	{
		auto prev_callback = std::move(m_callback);
		m_callback = [callback = std::move(callback)](boost::python::object&& result) {
			callback(PythonAgentTraits::convertToLoss(std::move(result)));
		};
		try {
			namespace np = boost::python::numpy;
			const auto t_max = static_cast<std::size_t>(data_sizes.size());
			const auto batch_size = static_cast<std::size_t>(action_ids.size()) / t_max;
			auto states_pyobj = PythonAgentTraits::convertObsBatch(states, t_max + 1, batch_size);
			auto action_ids_ndarray = NdArrayTraits<std::int64_t, 1>::convertToBatchedNdArray(action_ids, t_max, batch_size);
			auto rewards_pyobj = PythonAgentTraits::convertRewardBatch(rewards, t_max, batch_size);
			auto bp_ndarray = NdArrayTraits<float, 1>::convertToBatchedNdArray(behaviour_policies, t_max, batch_size);
			auto discounts_ndarray = NdArrayTraits<float, 1>::convertToBatchedNdArray(discounts, t_max, batch_size);
			auto loss_coefs_ndarray = NdArrayTraits<float, 1>::convertToBatchedNdArray(loss_coefs, t_max, batch_size);
			boost::python::list data_sizes_list;
			for (auto&& s : data_sizes) {
				data_sizes_list.append(s);
			}
			auto result = m_train_func(states_pyobj, action_ids_ndarray, rewards_pyobj, bp_ndarray, discounts_ndarray, loss_coefs_ndarray, data_sizes_list);
			if (prev_callback) {
				prev_callback(std::move(result));
			}
		} catch (boost::python::error_already_set) {
			::PyErr_Print();
			std::terminate();
		}
	}

	void sync()
	{
		auto prev_callback = std::move(m_callback);
		m_callback = nullptr;
		try {
			auto result = m_sync_func();
			if (prev_callback) {
				prev_callback(std::move(result));
			}
		} catch (boost::python::error_already_set) {
			::PyErr_Print();
			std::terminate();
		}
	}

	void save(std::int64_t index)
	{
		try {
			m_save_func(index);
		} catch (boost::python::error_already_set) {
			::PyErr_Print();
			std::terminate();
		}
	}

	void load(std::int64_t index)
	{
		try {
			m_load_func(index);
		} catch (boost::python::error_already_set) {
			::PyErr_Print();
			std::terminate();
		}
	}

private:
	boost::python::object m_python_main_ns;
	boost::python::object m_agent_object;
	boost::python::object m_predict_func;
	boost::python::object m_train_func;
	boost::python::object m_sync_func;
	boost::python::object m_save_func;
	boost::python::object m_load_func;
	std::function<void(boost::python::object&&)> m_callback;
};

struct A3CLossTraits
{
	using Loss = A3CLoss;

	static Loss convertToLoss(boost::python::object&& loss_obj)
	{
		Loss loss;
		loss.v_loss = boost::python::extract<double>(loss_obj[0]);
		loss.pi_loss = boost::python::extract<double>(loss_obj[1]);
		loss.entropy_loss = boost::python::extract<double>(loss_obj[2]);
		return loss;
	}
};

struct FloatRewardTraits
{
	using Reward = float;

	template <class... SizeT, std::enable_if_t<std::conjunction_v<std::is_convertible<SizeT, std::size_t>...>, std::nullptr_t> = nullptr>
	static boost::python::object convertRewardBatch(ranges::span<Reward> rewards, SizeT... batch_sizes)
	{
		return impala::NdArrayTraits<Reward, 1>::convertToBatchedNdArray(rewards, batch_sizes...);
	}
};

}  // namespace impala
