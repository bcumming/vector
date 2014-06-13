#pragma once

#include <cstddef>
#include <sstream>

namespace memory {

namespace types {
    typedef std::ptrdiff_t  difference_type;
    typedef std::size_t     size_type;
} // namespace types

namespace util {

    template <typename T>
    struct pretty_printer{
        static std::string print(const T& val) {
            return std::string("T()");
        }
    };

    template <>
    struct pretty_printer<float>{
        static std::string print(const float& val) {
            std::stringstream str;
            str << "float(" << val << ")";
            return str.str();
        }
    };

    template <>
    struct pretty_printer<double>{
        static std::string print(const double& val) {
            std::stringstream str;
            str << "double(" << val << ")";
            return str.str();
        }
    };

    template <>
    struct pretty_printer<size_t>{
        static std::string print(const size_t& val) {
            std::stringstream str;
            str << "size_t(" << val << ")";
            return str.str();
        }
    };

    template <typename T>
    struct type_printer{
        static std::string print() {
            return std::string("T");
        }
    };

    template <>
    struct type_printer<float>{
        static std::string print() {
            return std::string("float");
        }
    };

    template <>
    struct type_printer<double>{
        static std::string print() {
            return std::string("double");
        }
    };

    template <>
    struct type_printer<size_t>{
        static std::string print() {
            return std::string("size_t");
        }
    };

    template <>
    struct type_printer<int>{
        static std::string print() {
            return std::string("int");
        }
    };
} // namespace util

} // namespace memory

