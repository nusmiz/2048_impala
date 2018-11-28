#pragma once

#include <cassert>
#include <string>

#include <GLFW/glfw3.h>

#include "viewer/gl_util.hpp"

namespace viewer
{

class Window
{
public:
	Window(int width, int height, const std::string& title)
	{
		assert(width > 0 && height > 0);
		m_window = ::glfwCreateWindow(width, height, title.data(), nullptr, nullptr);
		if (!m_window) {
			throw GlfwException("glfwCreateWindow failed");
		}
	}

	Window(const Window&) = delete;
	Window(Window&& src)
	{
		m_window = src.m_window;
		src.m_window = nullptr;
	}
	Window& operator=(const Window&) = delete;
	Window& operator=(Window&& src)
	{
		if (valid()) {
			::glfwDestroyWindow(m_window);
		}
		m_window = src.m_window;
		src.m_window = nullptr;
		return *this;
	}

	~Window()
	{
		if (valid()) {
			::glfwDestroyWindow(m_window);
		}
	}

	bool valid() const noexcept
	{
		return m_window != nullptr;
	}
	explicit operator bool() const noexcept
	{
		return valid();
	}

	void setToCurrentContext() const
	{
		::glfwMakeContextCurrent(m_window);
	}

	bool shouldClose() const
	{
		return ::glfwWindowShouldClose(m_window) != 0;
	}

	void swapBuffers() const
	{
		::glfwSwapBuffers(m_window);
		::glfwPollEvents();
	}

private:
	::GLFWwindow* m_window;
};

}  // namespace viewer
