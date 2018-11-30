#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <tuple>

#include "action.hpp"
#include "environment.hpp"
#include "python_util.hpp"
#include "tensor.hpp"

namespace impala
{

class G2048Env
{
public:
	static constexpr int BOARD_SIZE = 4;
	static constexpr int MAX_NUMBER = BOARD_SIZE * BOARD_SIZE + 1;
	static constexpr int CONV_KERNEL_SIZE = 3;

	using RawObsTraits = NdArrayTraits<float, 8, MAX_NUMBER + 1, BOARD_SIZE * BOARD_SIZE>;
	using ConvObsTraits = NdArrayTraits<float, 8, MAX_NUMBER - CONV_KERNEL_SIZE + 1, CONV_KERNEL_SIZE + 3, BOARD_SIZE * BOARD_SIZE>;
	using InvalidMaskTraits = NdArrayTraits<std::uint8_t, 4>;

	using Observation = StaticTensor<std::uint8_t, BOARD_SIZE, BOARD_SIZE>;
	using ObsBatch = std::tuple<RawObsTraits::BufferType, ConvObsTraits::BufferType, InvalidMaskTraits::BufferType>;
	using Reward = float;
	using Action = FourDirections;

	G2048Env() : m_random_engine{std::random_device{}()} {}
	~G2048Env();

	Observation reset();
	std::tuple<Observation, Reward, EnvState> step(const Action& action);
	void render() const;

	template <class ForwardIterator,
	    std::enable_if_t<
	        std::conjunction_v<
	            std::is_base_of<std::forward_iterator_tag, typename std::iterator_traits<ForwardIterator>::iterator_category>,
	            std::is_convertible<typename std::iterator_traits<ForwardIterator>::reference, const Observation&>>,
	        std::nullptr_t> = nullptr>
	static void makeBatch(ForwardIterator first, ForwardIterator last, ObsBatch& output)
	{
		RawObsTraits::makeBufferForBatch(first, last, std::get<0>(output), writeRawData);
		ConvObsTraits::makeBufferForBatch(first, last, std::get<1>(output), writeConvData);
		InvalidMaskTraits::makeBufferForBatch(first, last, std::get<2>(output), writeInvalidMaskData);
	}

	bool isValidAction(Action action) const;

private:
	static void writeRawData(const Observation& obs, RawObsTraits::TensorRefType& dest);
	static void writeConvData(const Observation& obs, ConvObsTraits::TensorRefType& dest);
	static void writeInvalidMaskData(const Observation& obs, InvalidMaskTraits::TensorRefType& dest);

	int countEmpty() const;
	std::uint8_t maxNumber() const;
	bool isGameOver() const;

	void randomGen();

	Observation m_state;
	std::mt19937 m_random_engine;
#ifdef IMPALA_USE_GUI_VIEWER
	class RenderData;
	mutable RenderData* m_render_data = nullptr;
#endif
};

static_assert(IsEnvironmentV<G2048Env>);

}  // namespace impala
