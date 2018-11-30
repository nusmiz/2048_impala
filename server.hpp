#pragma once

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <variant>
#include <vector>

#include <boost/container/static_vector.hpp>
#include <range/v3/algorithm/copy.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/span.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/zip.hpp>

#include "agent.hpp"
#include "cuda/cuda_util.hpp"
#include "environment.hpp"

namespace impala
{

struct DefaultTrainParams
{
	static inline constexpr std::size_t NUM_ACTORS = 2048;
	static inline constexpr std::size_t NUM_PREDICTORS = 2;
	static inline constexpr std::size_t NUM_TRAINERS = 2;

	static inline constexpr std::size_t MIN_PREDICTION_BATCH_SIZE = 512;
	static inline constexpr std::size_t MAX_PREDICTION_BATCH_SIZE = 1024;
	static inline constexpr std::size_t MIN_TRAINING_BATCH_SIZE = 512;
	static inline constexpr std::size_t MAX_TRAINING_BATCH_SIZE = 1024;

	static inline constexpr std::size_t T_MAX = 5;
	static inline constexpr std::optional<std::size_t> MAX_EPISODE_LENGTH = std::nullopt;
	static inline constexpr float DISCOUNT = 0.99f;

	static inline constexpr double AVERAGE_LOSS_DECAY = 0.99;
	static inline constexpr std::optional<std::size_t> LOG_INTERVAL_STEPS = 10000;
	static inline constexpr std::optional<std::size_t> SAVE_INTERVAL_STEPS = 1000000;
};

template <class Environment, class Agent, class Parameters = DefaultTrainParams>
class Server
{
public:
	static_assert(IsEnvironmentV<Environment>);
	static_assert(IsAgentForGivenEnvironmentV<Agent, Environment>);

	using Reward = typename Environment::Reward;
	using Observation = typename Environment::Observation;
	using ObsBatch = typename Environment::ObsBatch;
	using Action = typename Environment::Action;
	using Loss = typename Agent::Loss;

	static inline constexpr std::size_t NUM_ACTORS = Parameters::NUM_ACTORS;
	static inline constexpr std::size_t NUM_PREDICTORS = Parameters::NUM_PREDICTORS;
	static inline constexpr std::size_t NUM_TRAINERS = Parameters::NUM_TRAINERS;

	static inline constexpr std::size_t MIN_PREDICTION_BATCH_SIZE = Parameters::MIN_PREDICTION_BATCH_SIZE;
	static inline constexpr std::size_t MAX_PREDICTION_BATCH_SIZE = Parameters::MAX_PREDICTION_BATCH_SIZE;
	static inline constexpr std::size_t MIN_TRAINING_BATCH_SIZE = Parameters::MIN_TRAINING_BATCH_SIZE;
	static inline constexpr std::size_t MAX_TRAINING_BATCH_SIZE = Parameters::MAX_TRAINING_BATCH_SIZE;

	static inline constexpr std::size_t T_MAX = Parameters::T_MAX;
	static inline constexpr std::optional<std::size_t> MAX_EPISODE_LENGTH = Parameters::MAX_EPISODE_LENGTH;
	static inline constexpr float DISCOUNT = Parameters::DISCOUNT;

	static inline constexpr double AVERAGE_LOSS_DECAY = Parameters::AVERAGE_LOSS_DECAY;
	static inline constexpr std::optional<std::size_t> LOG_INTERVAL_STEPS = Parameters::LOG_INTERVAL_STEPS;
	static inline constexpr std::optional<std::size_t> SAVE_INTERVAL_STEPS = Parameters::SAVE_INTERVAL_STEPS;

	Server(std::unique_ptr<Agent> agent) : m_agent(std::move(agent))
	{
		for ([[maybe_unused]] auto&& i : ranges::view::indices(NUM_PREDICTORS)) {
			m_predictors.emplace_back(*this);
		}
		for ([[maybe_unused]] auto&& i : ranges::view::indices(NUM_TRAINERS)) {
			m_trainers.emplace_back(*this);
		}
		for ([[maybe_unused]] auto&& i : ranges::view::indices(NUM_ACTORS)) {
			m_actors.emplace_back(*this);
		}
	}
	~Server()
	{
		for (auto&& predictor : m_predictors) {
			predictor.exit();
		}
		m_predictor_event.notify_all();
		m_predictors.clear();
		for (auto&& trainer : m_trainers) {
			trainer.exit();
		}
		m_trainer_event.notify_all();
		m_trainers.clear();
		for (auto&& actor : m_actors) {
			actor.exit();
		}
		m_actors.clear();
	}

	void train(const std::size_t training_steps)
	{
		std::size_t trained_steps = 0;

		Loss average_loss{};

		std::vector<std::reference_wrapper<Trainer>> training_batches;
		std::vector<std::reference_wrapper<Predictor>> prediction_batches;

		while (true) {
			training_batches.clear();
			prediction_batches.clear();
			{
				std::unique_lock lock{m_batches_lock};
				m_server_event.wait(lock, [this] { return !m_training_batches.empty() || !m_prediction_batches.empty(); });
				std::swap(m_training_batches, training_batches);
				std::swap(m_prediction_batches, prediction_batches);
			}
			for (auto&& trainer : training_batches) {
				auto& batch = trainer.get().getBatchData();
				auto num_datas = ranges::accumulate(batch.data_sizes, static_cast<std::int64_t>(0));
				m_agent->train(batch.states, batch.actions, batch.rewards, batch.policies, batch.discounts, batch.loss_coefs, batch.data_sizes, [this, &average_loss, &trained_steps, trainer, num_datas](const Loss& loss) {
					trainer.get().processFinished();
					average_loss = exponentialMovingAverage(average_loss, loss, AVERAGE_LOSS_DECAY);
					auto prev_trained_steps = trained_steps;
					trained_steps += static_cast<std::size_t>(num_datas);
					if constexpr (LOG_INTERVAL_STEPS.has_value()) {
						if (trained_steps / LOG_INTERVAL_STEPS.value() != prev_trained_steps / LOG_INTERVAL_STEPS.value()) {
							std::cout << "steps " << trained_steps << " , loss " << average_loss << std::endl;
						}
					}
					if constexpr (SAVE_INTERVAL_STEPS.has_value()) {
						if (trained_steps / SAVE_INTERVAL_STEPS.value() != prev_trained_steps / SAVE_INTERVAL_STEPS.value()) {
							m_agent->save(static_cast<int>(trained_steps));
						}
					}
				});
			}
			for (auto&& predictor : prediction_batches) {
				m_agent->template predict<DiscreteActionTraits<Action>::num_actions>(predictor.get().getStates(), predictor.get().getBufferForPolicies(), [predictor]() {
					predictor.get().processFinished();
				});
			}
			if (trained_steps >= training_steps) {
				std::cout << "training finished" << std::endl;
				break;
			}
		}
	}

private:
	class Predictor;
	class Trainer;
	class Actor;

	struct PredictionData
	{
		std::reference_wrapper<std::add_const_t<Observation>> observation;
		std::reference_wrapper<Actor> actor;
	};
	struct StepData
	{
		Observation observation;
		Action action;
		Reward reward;
		float policy;
		bool next_goal;
		bool aborted_terminal;
	};
	struct TrainingData
	{
		std::vector<StepData> steps;
		Observation terminal;
	};
	struct TrainingBatch
	{
		std::array<std::int64_t, T_MAX> data_sizes;
		ObsBatch states;
		PinnedMemoryVector<std::int64_t> actions;
		PinnedMemoryVector<Reward> rewards;
		PinnedMemoryVector<float> policies;
		PinnedMemoryVector<float> discounts;
		PinnedMemoryVector<float> loss_coefs;
	};

	class Predictor
	{
	public:
		explicit Predictor(Server& server) noexcept : m_server(server)
		{
			m_policy_lists.reserve(MAX_PREDICTION_BATCH_SIZE * DiscreteActionTraits<Action>::num_actions);
			m_thread = std::thread{[this] {
				run();
			}};
		}
		~Predictor()
		{
			m_thread.join();
		}

		void run()
		{
			std::vector<std::reference_wrapper<std::add_const_t<Observation>>> observations;
			std::vector<std::reference_wrapper<Actor>> actors;
			observations.reserve(MAX_PREDICTION_BATCH_SIZE);
			actors.reserve(MAX_PREDICTION_BATCH_SIZE);
			while (true) {
				observations.clear();
				actors.clear();
				bool data_remain = false;
				{
					std::unique_lock lock{m_server.get().m_prediction_queue_lock};
					m_server.get().m_predictor_event.wait(lock, [this] { return m_server.get().m_prediction_queue.size() >= MIN_PREDICTION_BATCH_SIZE || m_exit_flag; });
					if (m_exit_flag) {
						break;
					}
					auto& queue = m_server.get().m_prediction_queue;
					while (!queue.empty()) {
						if (observations.size() >= MAX_PREDICTION_BATCH_SIZE) {
							break;
						}
						auto& data = queue.front();
						observations.emplace_back(data.observation);
						actors.emplace_back(data.actor);
						queue.pop_front();
					}
					data_remain = (queue.size() >= MIN_PREDICTION_BATCH_SIZE);
				}
				if (data_remain) {
					m_server.get().m_predictor_event.notify_one();
				}
				m_policy_lists.resize(actors.size() * DiscreteActionTraits<Action>::num_actions, boost::container::default_init);
				Environment::makeBatch(observations.begin(), observations.end(), m_states);
				{
					std::lock_guard lock{m_server.get().m_batches_lock};
					m_server.get().m_prediction_batches.emplace_back(*this);
					m_processing_flag = true;
				}
				m_server.get().m_server_event.notify_one();
				{
					std::unique_lock lock{m_mutex};
					m_event.wait(lock, [this] { return !m_processing_flag || m_exit_flag; });
					if (m_exit_flag) {
						break;
					}
				}
				for (auto&& [i, actor] : ranges::view::zip(ranges::view::indices, actors)) {
					actor.get().setNextPolicyList({m_policy_lists.data() + i * DiscreteActionTraits<Action>::num_actions, DiscreteActionTraits<Action>::num_actions});
				}
			}
		}

		ObsBatch& getStates()
		{
			return m_states;
		}

		PinnedMemoryVector<float>& getBufferForPolicies()
		{
			return m_policy_lists;
		}

		void exit()
		{
			{
				std::lock_guard lock{m_mutex};
				m_exit_flag = true;
			}
			m_event.notify_one();
		}

		void processFinished()
		{
			{
				std::lock_guard lock{m_mutex};
				m_processing_flag = false;
			}
			m_event.notify_one();
		}

	private:
		std::reference_wrapper<Server> m_server;
		std::thread m_thread;
		std::mutex m_mutex;
		std::condition_variable m_event;
		bool m_processing_flag = false;
		bool m_exit_flag = false;
		ObsBatch m_states;
		PinnedMemoryVector<float> m_policy_lists;
	};

	class Trainer
	{
	public:
		explicit Trainer(Server& server) noexcept : m_server(server)
		{
			m_batch.actions.reserve(MAX_TRAINING_BATCH_SIZE * T_MAX);
			m_batch.rewards.reserve(MAX_TRAINING_BATCH_SIZE * T_MAX);
			m_batch.policies.reserve(MAX_TRAINING_BATCH_SIZE * T_MAX);
			m_batch.discounts.reserve(MAX_TRAINING_BATCH_SIZE * T_MAX);
			m_batch.loss_coefs.reserve(MAX_TRAINING_BATCH_SIZE * T_MAX);
			m_thread = std::thread{[this] {
				run();
			}};
		}
		~Trainer()
		{
			m_thread.join();
		}

		void run()
		{
			std::vector<TrainingData> datas;
			datas.reserve(MAX_TRAINING_BATCH_SIZE);
			std::vector<Observation> observations;
			observations.reserve(MAX_TRAINING_BATCH_SIZE * (T_MAX + 1));
			while (true) {
				datas.clear();
				observations.clear();
				m_batch.actions.clear();
				m_batch.rewards.clear();
				m_batch.policies.clear();
				m_batch.discounts.clear();
				m_batch.loss_coefs.clear();
				bool data_remain = false;
				{
					std::unique_lock lock{m_server.get().m_training_queue_lock};
					m_server.get().m_trainer_event.wait(lock, [this] { return m_server.get().m_training_queue.size() >= MIN_TRAINING_BATCH_SIZE || m_exit_flag; });
					if (m_exit_flag) {
						break;
					}
					auto& queue = m_server.get().m_training_queue;
					while (!queue.empty()) {
						if (datas.size() >= MAX_TRAINING_BATCH_SIZE) {
							break;
						}
						auto& data = queue.front();
						datas.emplace_back(std::move(data));
						queue.pop_front();
					}
					data_remain = (queue.size() >= MIN_TRAINING_BATCH_SIZE);
				}
				if (data_remain) {
					m_server.get().m_trainer_event.notify_one();
				}
				for (auto i : ranges::view::indices(T_MAX)) {
					m_batch.data_sizes.at(i) = 0;
					for (auto& data : datas) {
						auto& step = data.steps.at(i);
						observations.emplace_back(std::move(step.observation));
						m_batch.actions.emplace_back(DiscreteActionTraits<Action>::convertToID(step.action));
						m_batch.rewards.emplace_back(std::move(step.reward));
						m_batch.policies.emplace_back(std::move(step.policy));
						m_batch.discounts.emplace_back(step.next_goal ? 0.0f : DISCOUNT);
						m_batch.loss_coefs.emplace_back(step.aborted_terminal ? 0.0f : 1.0f);
						m_batch.data_sizes.at(i) += step.aborted_terminal ? 0 : 1;
					}
				}
				for (auto& data : datas) {
					observations.emplace_back(std::move(data.terminal));
				}
				Environment::makeBatch(observations.cbegin(), observations.cend(), m_batch.states);
				{
					std::lock_guard lock{m_server.get().m_batches_lock};
					m_server.get().m_training_batches.emplace_back(*this);
					m_processing_flag = true;
				}
				m_server.get().m_server_event.notify_one();
				{
					std::unique_lock lock{m_mutex};
					m_event.wait(lock, [this] { return !m_processing_flag || m_exit_flag; });
					if (m_exit_flag) {
						break;
					}
				}
			}
		}

		TrainingBatch& getBatchData()
		{
			return m_batch;
		}

		void exit()
		{
			{
				std::lock_guard lock{m_mutex};
				m_exit_flag = true;
			}
			m_event.notify_one();
		}

		void processFinished()
		{
			{
				std::lock_guard lock{m_mutex};
				m_processing_flag = false;
			}
			m_event.notify_one();
		}

	private:
		std::reference_wrapper<Server> m_server;
		std::thread m_thread;
		std::mutex m_mutex;
		std::condition_variable m_event;
		bool m_processing_flag = false;
		bool m_exit_flag = false;
		TrainingBatch m_batch;
	};

	class Actor
	{
	public:
		explicit Actor(Server& server) noexcept : m_server(server)
		{
			m_thread = std::thread{[this] {
				run();
			}};
			m_action_sample_random_engine.seed(std::random_device{}());
		}
		~Actor()
		{
			m_thread.join();
		}

		void run()
		{
			std::vector<StepData> step_datas;
			step_datas.reserve(T_MAX);
			while (true) {
				Reward sum_of_reward = Reward{};
				std::size_t t = 0;
				Observation observation = m_env.reset();
				while (true) {
					{
						bool enough_predictor_data = false;
						{
							std::lock_guard lock{m_server.get().m_prediction_queue_lock};
							m_server.get().m_prediction_queue.emplace_back(PredictionData{std::cref(observation), *this});
							m_predicting_flag = true;
							enough_predictor_data = m_server.get().m_prediction_queue.size() >= MIN_PREDICTION_BATCH_SIZE;
						}
						if (enough_predictor_data) {
							m_server.get().m_predictor_event.notify_one();
						}
					}
					{
						std::unique_lock lock{m_mutex};
						m_event.wait(lock, [this] { return !m_predicting_flag || m_exit_flag; });
						if (m_exit_flag) {
							return;
						}
					}
					Action next_action;
					float policy;
					while (true) {
						auto action_id = std::discrete_distribution<std::int64_t>(m_policy_list.begin(), m_policy_list.end())(m_action_sample_random_engine);
						next_action = DiscreteActionTraits<Action>::convertFromID(action_id);
						if (m_env.isValidAction(next_action)) {
							policy = m_policy_list[action_id];
							break;
						}
					}
					if (isMainActor()) {
						m_env.render();
					}
					auto&& [next_obs, current_reward, status] = m_env.step(next_action);
					++t;
					sum_of_reward += current_reward;
					step_datas.push_back({std::move(observation), next_action, current_reward, policy, status == EnvState::FINISHED, false});
					auto addTrainingData = [&] {
						if constexpr (NUM_TRAINERS > 0) {
							TrainingData data{std::move(step_datas), next_obs.clone()};
							bool enough_trainer_data = false;
							{
								std::lock_guard lock{m_server.get().m_training_queue_lock};
								auto& queue = m_server.get().m_training_queue;
								queue.emplace_back(std::move(data));
								enough_trainer_data = (queue.size() >= MIN_TRAINING_BATCH_SIZE);
							}
							if (enough_trainer_data) {
								m_server.get().m_trainer_event.notify_one();
							}
						}
						step_datas.clear();
						step_datas.reserve(T_MAX);
					};
					if (step_datas.size() == T_MAX) {
						addTrainingData();
					}
					if (status == EnvState::FINISHED) {
						break;
					}
					if constexpr (MAX_EPISODE_LENGTH.has_value()) {
						if (t >= MAX_EPISODE_LENGTH.value()) {
							if (!step_datas.empty()) {
								step_datas.push_back({next_obs.clone(), Action{}, Reward{}, 1.0f, true, true});
								if (step_datas.size() == T_MAX) {
									addTrainingData();
								}
							}
							break;
						}
					}
					observation = std::move(next_obs);
				}
				if (isMainActor()) {
					std::cout << "finish episode : " << t << " " << std::setprecision(5) << sum_of_reward << std::endl;
				}
			}
		}

		void exit()
		{
			{
				std::lock_guard lock{m_mutex};
				m_exit_flag = true;
			}
			m_event.notify_one();
		}

		void setNextPolicyList(ranges::span<float> policy_list)
		{
			{
				std::lock_guard lock{m_mutex};
				ranges::copy(policy_list, m_policy_list.begin());
				m_predicting_flag = false;
			}
			m_event.notify_one();
		}

		bool isMainActor() const
		{
			return this == &m_server.get().m_actors.front();
		}

	private:
		std::reference_wrapper<Server> m_server;
		std::thread m_thread;
		std::mutex m_mutex;
		std::condition_variable m_event;
		std::array<float, DiscreteActionTraits<Action>::num_actions> m_policy_list;
		bool m_predicting_flag = false;
		bool m_exit_flag = false;
		Environment m_env;
		std::mt19937 m_action_sample_random_engine;
	};

	std::unique_ptr<Agent> m_agent;
	boost::container::static_vector<Predictor, NUM_PREDICTORS> m_predictors;
	boost::container::static_vector<Trainer, NUM_TRAINERS> m_trainers;
	boost::container::static_vector<Actor, NUM_ACTORS> m_actors;
	std::deque<PredictionData> m_prediction_queue;
	std::mutex m_prediction_queue_lock;
	std::condition_variable m_predictor_event;
	std::deque<TrainingData> m_training_queue;
	std::mutex m_training_queue_lock;
	std::condition_variable m_trainer_event;
	std::vector<std::reference_wrapper<Predictor>> m_prediction_batches;
	std::vector<std::reference_wrapper<Trainer>> m_training_batches;
	std::mutex m_batches_lock;
	std::condition_variable m_server_event;
};

}  // namespace impala
