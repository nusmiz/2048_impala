#pragma once

#include <array>
#include <cassert>

#include <GL/gl.h>
#include <range/v3/span.hpp>

#include "viewer/color.hpp"

namespace viewer
{

enum class Filter
{
	NEAREST = GL_NEAREST,
	LINEAR = GL_LINEAR
};

enum class WrapMode
{
	REPEAT = GL_REPEAT,
	MIRROR = GL_MIRRORED_REPEAT,
	EDGE = GL_CLAMP_TO_EDGE,
	BORDER = GL_CLAMP_TO_BORDER,
};

class Texture
{
public:
	Texture() noexcept : m_id{0}, m_width{0}, m_height{0} {}
	Texture(ranges::span<Color8Bit> colors, int width, int height) : m_id{0}
	{
		setData(colors, width, height);
	}

	Texture(const Texture&) = delete;
	Texture(Texture&& src) noexcept : m_id{src.m_id}, m_width{src.m_width}, m_height{src.m_height}, m_filter{src.m_filter}, m_wrap_mode{src.m_wrap_mode}, m_border_color{src.m_border_color}
	{
		src.m_id = 0;
	}
	Texture& operator=(const Texture&) = delete;
	Texture& operator=(Texture&& src)
	{
		if (valid()) {
			::glDeleteTextures(1, &m_id);
		}
		m_id = src.m_id;
		m_width = src.m_width;
		m_height = src.m_height;
		m_filter = src.m_filter;
		m_wrap_mode = src.m_wrap_mode;
		m_border_color = src.m_border_color;
		src.m_id = 0;
		return *this;
	}

	~Texture()
	{
		if (valid()) {
			::glDeleteTextures(1, &m_id);
		}
	}

	void setData(ranges::span<Color8Bit> colors, int width, int height);

	::GLuint id() const noexcept
	{
		return m_id;
	}

	void bind() const
	{
		::glBindTexture(GL_TEXTURE_2D, m_id);
	}

	bool valid() const noexcept
	{
		return id() != 0;
	}
	explicit operator bool() const noexcept
	{
		return valid();
	}

	int width() const noexcept
	{
		return m_width;
	}
	int height() const noexcept
	{
		return m_height;
	}

	Filter filter() const noexcept
	{
		return m_filter;
	}
	void setFilter(Filter filter)
	{
		m_filter = filter;
		if (valid()) {
			bind();
			::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(m_filter));
			::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(m_filter));
		}
	}

	WrapMode wrapMode() const noexcept
	{
		return m_wrap_mode;
	}
	void setWrapMode(WrapMode mode)
	{
		m_wrap_mode = mode;
		if (valid()) {
			bind();
			::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, static_cast<GLint>(m_wrap_mode));
			::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, static_cast<GLint>(m_wrap_mode));
		}
	}

	Color borderColor() const noexcept
	{
		return m_border_color;
	}
	void setBorderColor(Color color)
	{
		m_border_color = color;
		if (valid()) {
			bind();
			std::array<GLfloat, 4> border = {m_border_color.r, m_border_color.g, m_border_color.b, m_border_color.a};
			::glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border.data());
		}
	}

	void draw(float x, float y, float width, float height, Color color = {1.0f, 1.0f, 1.0f});

	void draw(float x, float y, Color color = {1.0f, 1.0f, 1.0f})
	{
		draw(x, y, static_cast<float>(m_width), static_cast<float>(m_height), color);
	}

private:
	::GLuint m_id;
	int m_width;
	int m_height;
	Filter m_filter = Filter::LINEAR;
	WrapMode m_wrap_mode = WrapMode::BORDER;
	Color m_border_color = {0.0f, 0.0f, 0.0f, 0.0f};
};

}  // namespace viewer
