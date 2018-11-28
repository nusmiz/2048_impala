#pragma once

#include <cstdint>

namespace viewer
{

struct Color
{
	float r;
	float g;
	float b;
	float a;

	Color() = default;
	Color(float r, float g, float b) : r(r), g(g), b(b), a(1.0f) {}
	Color(float r, float g, float b, float a) : r(r), g(g), b(b), a(a) {}
};

struct Color8Bit
{
	std::uint8_t r;
	std::uint8_t g;
	std::uint8_t b;
	std::uint8_t a;

	Color8Bit() = default;
	Color8Bit(std::uint8_t r, std::uint8_t g, std::uint8_t b) : r(r), g(g), b(b), a(255) {}
	Color8Bit(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) : r(r), g(g), b(b), a(a) {}

	explicit operator Color() const noexcept
	{
		return Color(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
	}
};

}  // namespace viewer
