#include <iostream>
#include <memory>

#include "action.hpp"
#include "environment.hpp"
#include "python_agent.hpp"
#include "python_util.hpp"
#include "server.hpp"
#include "tensor.hpp"

#ifdef IMPALA_USE_GUI_VIEWER
#include "viewer/gl_util.hpp"
#endif

#include "envs/g2048/g2048_env.hpp"


struct G2048TrainParams
{
	static inline constexpr std::size_t NUM_ACTORS = 4096;
	static inline constexpr std::size_t NUM_PREDICTORS = 4;
	static inline constexpr std::size_t NUM_TRAINERS = 16;

	static inline constexpr std::size_t MIN_PREDICTION_BATCH_SIZE = 1024;
	static inline constexpr std::size_t MAX_PREDICTION_BATCH_SIZE = 1024;
	static inline constexpr std::size_t MIN_TRAINING_BATCH_SIZE = 256;
	static inline constexpr std::size_t MAX_TRAINING_BATCH_SIZE = 256;

	static inline constexpr std::size_t T_MAX = 12;
	static inline constexpr std::optional<std::size_t> MAX_EPISODE_LENGTH = std::nullopt;
	static inline constexpr float DISCOUNT = 0.99f;

	static inline constexpr double AVERAGE_LOSS_DECAY = 0.99;
	static inline constexpr std::optional<std::size_t> LOG_INTERVAL_STEPS = 100000;
	static inline constexpr std::optional<std::size_t> SAVE_INTERVAL_STEPS = 10000000;
};

struct G2048AgentTraits : impala::FloatRewardTraits, impala::A3CLossTraits
{
	using Environment = impala::G2048Env;

	static boost::python::object create(boost::python::object& main_ns)
	{
		boost::python::exec("from models.g2048_a3c_model import G2048A3CModel", main_ns);
		boost::python::exec("from agents import Impala", main_ns);
		boost::python::exec("import torch.optim as optim", main_ns);
		boost::python::exec("def make_optimizer(parameters):\n"
		                    "    return optim.RMSprop(parameters, lr=0.01, alpha=0.95, eps=0.1)\n",
		    main_ns);
		auto model = main_ns["G2048A3CModel"]();
		auto optimizer_maker = boost::python::eval("make_optimizer", main_ns);
		return main_ns["Impala"](model, optimizer_maker, impala::USE_CUDA);
	}

	static boost::python::object convertObsBatch(Environment::ObsBatch& batch, std::size_t batch_size)
	{
		return boost::python::make_tuple(
		    Environment::RawObsTraits::convertToBatchedNdArray(std::get<0>(batch), batch_size),
		    Environment::ConvObsTraits::convertToBatchedNdArray(std::get<1>(batch), batch_size),
		    Environment::InvalidMaskTraits::convertToBatchedNdArray(std::get<2>(batch), batch_size));
	}
};

int main()
{
#ifdef IMPALA_USE_GUI_VIEWER
	viewer::GlfwInitializer glfw_initializer;
#endif
	using namespace impala;
	PythonInitializer py_initializer{false};

	using Agent = PythonAgent<G2048AgentTraits>;
	auto agent = std::make_unique<Agent>();
	auto server = std::make_unique<Server<G2048Env, Agent, G2048TrainParams>>(std::move(agent));
	server->train(4000000000);
	return 0;
}
