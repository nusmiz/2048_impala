#include "load_png.hpp"

#include <cstdio>

#include <boost/container/vector.hpp>
#include <png.h>

namespace viewer
{

Texture loadPng(const std::string& path)
{
	::png_structp png_ptr = ::png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	if (png_ptr == nullptr) {
		throw PngLoadException{"png_create_read_struct failed"};
	}
	::png_infop info = ::png_create_info_struct(png_ptr);
	if (info == nullptr) {
		::png_destroy_read_struct(&png_ptr, nullptr, nullptr);
		throw PngLoadException{"png_create_info_struct failed"};
	}
	std::FILE* fp = std::fopen(path.data(), "rb");
	if (fp == nullptr) {
		::png_destroy_read_struct(&png_ptr, nullptr, nullptr);
		throw PngLoadException{"cannot open " + path};
	}
	::png_init_io(png_ptr, fp);
	::png_set_sig_bytes(png_ptr, 0);
	::png_read_info(png_ptr, info);
	::png_uint_32 width, height;
	int bit_depth, color_type;
	if (::png_get_IHDR(png_ptr, info, &width, &height, &bit_depth, &color_type, nullptr, nullptr, nullptr) == 0) {
		::png_destroy_read_struct(&png_ptr, &info, nullptr);
		throw PngLoadException{"png_get_IHDR failed"};
	}
	::png_set_expand(png_ptr);
	::png_set_gray_to_rgb(png_ptr);
	::png_set_strip_16(png_ptr);
	::png_set_add_alpha(png_ptr, 0xFF, PNG_FILLER_AFTER);
	::png_read_update_info(png_ptr, info);

	boost::container::vector<Color8Bit> color_buffer(width * height, boost::container::default_init);
	for (std::uint32_t i = 0; i < height; ++i) {
		::png_read_row(png_ptr, reinterpret_cast<png_bytep>(color_buffer.data() + (i * width)), nullptr);
	}
	::png_read_end(png_ptr, info);
	::png_destroy_read_struct(&png_ptr, &info, nullptr);
	std::fclose(fp);

	return Texture(color_buffer, static_cast<int>(width), static_cast<int>(height));
}

}  // namespace viewer
