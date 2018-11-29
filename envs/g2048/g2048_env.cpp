#include "g2048_env.hpp"

#include <algorithm>
#include <cmath>

#include <range/v3/view/indices.hpp>

#ifdef IMPALA_USE_GUI_VIEWER
#include <range/v3/view/indices.hpp>
#include <range/v3/view/zip.hpp>

#include "viewer/gl_util.hpp"
#include "viewer/load_png.hpp"
#include "viewer/texture.hpp"
#include "viewer/window.hpp"
#endif

namespace impala
{

G2048Env::Observation G2048Env::reset()
{
	for (auto&& row : m_state) {
		for (auto&& data : row) {
			data = 0;
		}
	}
	randomGen();
	randomGen();
	return m_state.clone();
}

namespace
{

template <int DIR>
std::uint8_t& get(G2048Env::Observation& obs, int x, int y)
{
	static_assert(0 <= DIR && DIR < 8);
	if constexpr (DIR == 0) {
		return obs[y][x];
	} else if constexpr (DIR == 1) {
		return obs[G2048Env::BOARD_SIZE - 1 - x][y];
	} else if constexpr (DIR == 2) {
		return obs[G2048Env::BOARD_SIZE - 1 - y][G2048Env::BOARD_SIZE - 1 - x];
	} else if constexpr (DIR == 3) {
		return obs[x][G2048Env::BOARD_SIZE - 1 - y];
	} else if constexpr (DIR == 4) {
		return obs[x][y];
	} else if constexpr (DIR == 5) {
		return obs[y][G2048Env::BOARD_SIZE - 1 - x];
	} else if constexpr (DIR == 6) {
		return obs[G2048Env::BOARD_SIZE - 1 - x][G2048Env::BOARD_SIZE - 1 - y];
	} else if constexpr (DIR == 7) {
		return obs[G2048Env::BOARD_SIZE - 1 - y][x];
	}
}
template <int DIR>
std::uint8_t get(const G2048Env::Observation& obs, int x, int y)
{
	static_assert(0 <= DIR && DIR < 8);
	if constexpr (DIR == 0) {
		return obs[y][x];
	} else if constexpr (DIR == 1) {
		return obs[G2048Env::BOARD_SIZE - 1 - x][y];
	} else if constexpr (DIR == 2) {
		return obs[G2048Env::BOARD_SIZE - 1 - y][G2048Env::BOARD_SIZE - 1 - x];
	} else if constexpr (DIR == 3) {
		return obs[x][G2048Env::BOARD_SIZE - 1 - y];
	} else if constexpr (DIR == 4) {
		return obs[x][y];
	} else if constexpr (DIR == 5) {
		return obs[y][G2048Env::BOARD_SIZE - 1 - x];
	} else if constexpr (DIR == 6) {
		return obs[G2048Env::BOARD_SIZE - 1 - x][G2048Env::BOARD_SIZE - 1 - y];
	} else if constexpr (DIR == 7) {
		return obs[G2048Env::BOARD_SIZE - 1 - y][x];
	}
}

template <int DIR>
void moveLeft(G2048Env::Observation& state)
{
	for (int y : ranges::view::indices(G2048Env::BOARD_SIZE)) {
		for (int new_x : ranges::view::indices(G2048Env::BOARD_SIZE)) {
			std::uint8_t val1 = 0;
			std::uint8_t val2 = 0;
			for (int x : ranges::view::indices(new_x, G2048Env::BOARD_SIZE)) {
				if (get<DIR>(state, x, y) != 0) {
					if (val1 == 0) {
						val1 = get<DIR>(state, x, y);
						get<DIR>(state, x, y) = 0;
					} else {
						val2 = get<DIR>(state, x, y);
						get<DIR>(state, x, y) = 0;
						break;
					}
				}
			}
			if (val1 == 0) {
				break;
			}
			if (val1 == val2) {
				get<DIR>(state, new_x, y) = static_cast<std::uint8_t>(val1 + 1);
			} else {
				get<DIR>(state, new_x, y) = val1;
				if (val2 != 0) {
					get<DIR>(state, new_x + 1, y) = val2;
				}
			}
		}
	}
}

template <int DIR>
void writeRawDataHelper(const G2048Env::Observation& obs, G2048Env::RawObsTraits::TensorRefType& dest)
{
	if constexpr (DIR < 8) {
		for (auto y : ranges::view::indices(G2048Env::BOARD_SIZE)) {
			for (auto x : ranges::view::indices(G2048Env::BOARD_SIZE)) {
				auto number = get<DIR>(obs, x, y);
				for (auto n : ranges::view::indices(G2048Env::MAX_NUMBER + 1)) {
					dest[DIR][n][y * G2048Env::BOARD_SIZE + x] = (n == number ? 1.0f : 0.0f);
				}
			}
		}
		writeRawDataHelper<DIR + 1>(obs, dest);
	}
}

template <int DIR>
void writeConvDataHelper(const G2048Env::Observation& obs, G2048Env::ConvObsTraits::TensorRefType& dest)
{
	if constexpr (DIR < 8) {
		for (auto n : ranges::view::indices(G2048Env::MAX_NUMBER - G2048Env::CONV_KERNEL_SIZE + 1)) {
			for (auto y : ranges::view::indices(G2048Env::BOARD_SIZE)) {
				for (auto x : ranges::view::indices(G2048Env::BOARD_SIZE)) {
					auto number = get<DIR>(obs, x, y);
					for (auto n2 : ranges::view::indices(G2048Env::CONV_KERNEL_SIZE)) {
						dest[DIR][n][n2][y * G2048Env::BOARD_SIZE + x] = (n + 1 + n2 == number ? 1.0f : 0.0f);
					}
					dest[DIR][n][G2048Env::CONV_KERNEL_SIZE + 0][y * G2048Env::BOARD_SIZE + x] = (number == 0 ? 1.0f : 0.0f);
					dest[DIR][n][G2048Env::CONV_KERNEL_SIZE + 1][y * G2048Env::BOARD_SIZE + x] = ((number < n + 1 && number != 0) ? 1.0f : 0.0f);
					dest[DIR][n][G2048Env::CONV_KERNEL_SIZE + 2][y * G2048Env::BOARD_SIZE + x] = (number >= n + 1 + G2048Env::CONV_KERNEL_SIZE ? 1.0f : 0.0f);
				}
			}
		}
		writeConvDataHelper<DIR + 1>(obs, dest);
	}
}

}  // namespace

std::tuple<G2048Env::Observation, G2048Env::Reward, EnvState> G2048Env::step(const Action& action)
{
	auto prev_state = m_state;
	if (action == FourDirections::LEFT) {
		moveLeft<0>(m_state);
	} else if (action == FourDirections::RIGHT) {
		moveLeft<2>(m_state);
	} else if (action == FourDirections::UP) {
		moveLeft<3>(m_state);
	} else if (action == FourDirections::DOWN) {
		moveLeft<1>(m_state);
	}
	if (m_state == prev_state) {
		return std::make_tuple(m_state, -11.0f, EnvState::RUNNING);
	}
	randomGen();
	if (isGameOver()) {
		return std::make_tuple(m_state, -10.0f, EnvState::FINISHED);
	}
	return std::make_tuple(m_state, 1.0f, EnvState::RUNNING);
}

void G2048Env::writeRawData(const Observation& obs, RawObsTraits::TensorRefType& dest)
{
	writeRawDataHelper<0>(obs, dest);
}
void G2048Env::writeConvData(const Observation& obs, ConvObsTraits::TensorRefType& dest)
{
	writeConvDataHelper<0>(obs, dest);
}

int G2048Env::countEmpty() const
{
	int count = 0;
	for (auto&& row : m_state) {
		for (auto&& data : row) {
			if (data == 0) {
				++count;
			}
		}
	}
	return count;
}

std::uint8_t G2048Env::maxNumber() const
{
	std::uint8_t max = 0;
	for (auto&& row : m_state) {
		for (auto&& data : row) {
			max = std::max(max, data);
		}
	}
	return max;
}

bool G2048Env::isGameOver() const
{
	auto temp = m_state;
	moveLeft<0>(temp);
	if (temp != m_state) {
		return false;
	}
	moveLeft<1>(temp);
	if (temp != m_state) {
		return false;
	}
	moveLeft<2>(temp);
	if (temp != m_state) {
		return false;
	}
	moveLeft<3>(temp);
	if (temp != m_state) {
		return false;
	}
	return true;
}

void G2048Env::randomGen()
{
	assert(countEmpty() > 0);
	int position = std::uniform_int_distribution<>{0, countEmpty() - 1}(m_random_engine);
	for (auto&& row : m_state) {
		for (auto&& data : row) {
			if (data == 0) {
				if (position == 0) {
					if (std::uniform_int_distribution<>{0, 10 - 1}(m_random_engine) == 0) {
						data = 2;
					} else {
						data = 1;
					}
					return;
				} else {
					--position;
				}
			}
		}
	}
}

#ifdef IMPALA_USE_GUI_VIEWER
class G2048Env::RenderData
{
public:
	RenderData() : m_window(600, 600, "2048")
	{
		m_window.setToCurrentContext();
		m_board_texture = viewer::loadPng("./envs/g2048/image/board.png");
		for (auto i : ranges::view::indices(MAX_NUMBER)) {
			m_number_textures[i] = viewer::loadPng("./envs/g2048/image/num_" + std::to_string(i + 1) + ".png");
		}
	}
	void render(const Observation& state)
	{
		using namespace viewer;
		m_window.setToCurrentContext();
		disableDepthTest();
		clearScreen();
		setOrthoProj(0, 0, 600, 600);
		::glMatrixMode(GL_MODELVIEW);
		::glLoadIdentity();
		m_board_texture.draw(0, 0);
		for (auto y : ranges::view::indices(BOARD_SIZE)) {
			for (auto x : ranges::view::indices(BOARD_SIZE)) {
				auto n = state[y][x];
				if (n != 0) {
					m_number_textures.at(n - 1).draw(66 + static_cast<float>(x) * 121, 66 + static_cast<float>(y) * 121, 107, 107);
				}
			}
		}
		m_window.swapBuffers();
	}

private:
	viewer::Window m_window;
	viewer::Texture m_board_texture;
	std::array<viewer::Texture, MAX_NUMBER> m_number_textures;
};

void G2048Env::render() const
{
	if (m_render_data == nullptr) {
		m_render_data = new RenderData();
	}
	m_render_data->render(m_state);
}
G2048Env::~G2048Env()
{
	delete m_render_data;
}
#else
G2048Env::~G2048Env() = default;
void G2048Env::render() const
{}
#endif

}  // namespace impala
