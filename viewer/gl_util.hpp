#pragma once

#include <stdexcept>

#include <GLFW/glfw3.h>

#include "color.hpp"

namespace viewer
{

class GlfwException : public std::runtime_error
{
public:
	using runtime_error::runtime_error;
};

class GlfwInitializer
{
public:
	GlfwInitializer();
	~GlfwInitializer();
};

inline void enableAlphaBlend()
{
	::glEnable(GL_BLEND);
	::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

inline void enableDepthTest()
{
	::glEnable(GL_DEPTH_TEST);
}
inline void disableDepthTest()
{
	::glDisable(GL_DEPTH_TEST);
}

inline void clearScreen()
{
	::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void setOrthoProj(float x, float y, float width, float height, float near = -1, float far = 1);

void fillRect(float x, float y, float width, float height, Color c);

void fillCircle(float x, float y, float r, Color c);

}  // namespace viewer
