#pragma once

#include <stdexcept>
#include <string>

#include "texture.hpp"

namespace viewer
{

class PngLoadException : public std::runtime_error
{
public:
	using runtime_error::runtime_error;
};

Texture loadPng(const std::string& path);

}  // namespace viewer
