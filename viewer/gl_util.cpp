#include <cmath>
#include <cstdlib>
#include <iostream>

#include "gl_util.hpp"

namespace viewer
{

GlfwInitializer::GlfwInitializer()
{
	::glfwSetErrorCallback([](int error, const char* description) {
		std::cerr << "GLFW error " << error << ": " << description << std::endl;
	});
	if (!::glfwInit()) {
		throw GlfwException("glfwInit failed");
	}
	::glfwWindowHint(GLFW_SAMPLES, 4);
}

GlfwInitializer::~GlfwInitializer()
{
	::glfwTerminate();
}

void setOrthoProj(float x, float y, float width, float height, float near, float far)
{
	::glMatrixMode(GL_PROJECTION);
	::glLoadIdentity();
	::glOrtho(x, x + width, y + height, y, near, far);
}

void fillRect(float x, float y, float width, float height, Color c)
{
	::glBegin(GL_POLYGON);
	::glColor4f(c.r, c.g, c.b, c.a);
	::glVertex3f(x + width, y + height, 0.0f);
	::glVertex3f(x, y + height, 0.0f);
	::glVertex3f(x, y, 0.0f);
	::glVertex3f(x + width, y, 0.0f);
	::glEnd();
}

void fillCircle(float x, float y, float r, Color c)
{
	static constexpr float PI = 3.14159265358979f;
	::glBegin(GL_POLYGON);
	::glColor4f(c.r, c.g, c.b, c.a);
	static constexpr int N = 32;
	for (int i = 0; i < N; ++i) {
		::glVertex3f(x + r * std::cos(2 * PI * static_cast<float>(i) / N), y + r * std::sin(2 * PI * static_cast<float>(i) / N), 0.0f);
	}
	::glEnd();
}

}  // namespace viewer
