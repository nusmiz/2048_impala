#include "texture.hpp"

namespace viewer
{

void Texture::setData(ranges::span<Color8Bit> colors, int width, int height)
{
	assert(colors.size() == width * height);
	if (valid()) {
		::glDeleteTextures(1, &m_id);
		m_id = 0;
	}
	::glGenTextures(1, &m_id);
	bind();
	m_width = width;
	m_height = height;
	::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	::glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, static_cast<GLsizei>(width), static_cast<GLsizei>(height), 0, GL_RGBA, GL_UNSIGNED_BYTE, colors.data());
	::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(m_filter));
	::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(m_filter));
	::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, static_cast<GLint>(m_wrap_mode));
	::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, static_cast<GLint>(m_wrap_mode));
	std::array<GLfloat, 4> border = {m_border_color.r, m_border_color.g, m_border_color.b, m_border_color.a};
	::glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border.data());
}

void Texture::draw(float x, float y, float width, float height, Color color)
{
	::glEnable(GL_TEXTURE_2D);
	bind();
	::glBegin(GL_POLYGON);
	::glColor4f(color.r, color.g, color.b, color.a);
	::glTexCoord2f(1.0f, 1.0f);
	::glVertex3f(x + width, y + height, 0.0f);
	::glTexCoord2f(0.0f, 1.0f);
	::glVertex3f(x, y + height, 0.0f);
	::glTexCoord2f(0.0f, 0.0f);
	::glVertex3f(x, y, 0.0f);
	::glTexCoord2f(1.0f, 0.0f);
	::glVertex3f(x + width, y, 0.0f);
	::glEnd();
	::glDisable(GL_TEXTURE_2D);
}

}  // namespace viewer
