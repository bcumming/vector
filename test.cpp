#include <iostream>

template <typename T, std::size_t S>
struct helper {};

template <typename T>
struct helper<T, 1> {
    using type = uint8_t;
};
template <typename T>
struct helper<T, 2> {
    using type = uint16_t;
};
template <typename T>
struct helper<T, 4> {
    using type = uint32_t;
};
template <typename T>
struct helper<T, 8> {
    using type = uint64_t;
};

template <typename T>
using int_type = typename helper<T, sizeof(T)>::type;

template <typename T>
int_type<T> caster(T value) {
    int_type<T> v;
    *reinterpret_cast<T*>(&v) = value;
    return v;
}

int main(void) {
    {
    auto val = caster(0.);
    static_assert(std::is_same<uint64_t, decltype(val)>::value, "no good");
    std::cout << sizeof(val)*8 << " " << val << std::endl;
    }

    {
    auto val = caster(0.f);
    static_assert(std::is_same<uint32_t, decltype(val)>::value, "no good");
    std::cout << sizeof(val)*8 << " " << val << std::endl;
    }

    {
    auto val = caster(-0.);
    static_assert(std::is_same<uint64_t, decltype(val)>::value, "no good");
    std::cout << sizeof(val)*8 << " " << val << std::endl;
    }

    {
    auto val = caster(-0.f);
    static_assert(std::is_same<uint32_t, decltype(val)>::value, "no good");
    std::cout << sizeof(val)*8 << " " << val << std::endl;
    }

    return 0;
}
