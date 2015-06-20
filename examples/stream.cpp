#include <chrono>
#include <iostream>

#include <Vector.hpp>

using value_type = double;
using size_type  = std::size_t;

using clock_type    = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double>;

// let's use 256-bit byte alignment
// aka. the alignment of an AVX register
constexpr std::size_t alignment() { return 256/8; }

using namespace memory;
template <typename T>
using vector
    = Array<T, HostCoordinator<T, Allocator<T, impl::AlignedPolicy<alignment()>>>>;

template <typename T>
void triad(vector<T>      & a,
           vector<T> const& b,
           vector<T> const& c,
           T scalar)
{
    auto const N = a.size();
    #pragma omp parallel for
    for(auto i=size_type{0}; i<N; ++i) {
        a[i] = b[i] + scalar * c[i];
    }
}

template <typename T>
void scale(vector<T>      & a,
           vector<T> const& b,
           T scalar)
{
    auto const N = a.size();
    #pragma omp parallel for
    for(auto i=size_type{0}; i<N; ++i) {
        a[i] = scalar * b[i];
    }
}

template <typename T>
void copy(vector<T>      & a,
          vector<T> const& b)
{
    auto const N = a.size();
    #pragma omp parallel for
    for(auto i=size_type{0}; i<N; ++i) {
        a[i] = b[i];
    }
}

template <typename T>
void   add(vector<T>      & a,
           vector<T> const& b,
           vector<T> const& c)
{
    auto const N = a.size();
    #pragma omp parallel for
    for(auto i=size_type{0}; i<N; ++i) {
        a[i] = b[i] + c[i];
    }
}

template <typename T>
void init(vector<T> & a,
          vector<T> & b,
          vector<T> & c)
{
    auto const N = a.size();
    #pragma omp parallel for
    for(auto i=size_type{0}; i<N; ++i) {
        a[i] = T{1};
        b[i] = T{2};
        c[i] = T{3};
    }
}

int main(int argc, char **argv) {
    size_type pow = 24;
    if(argc>1) {
        pow = std::stod(argv[1]);
    }
    auto const N = size_type{1} << pow;
    auto num_trials = 4;

    std::cout << "arrays of length " << N << std::endl;

    // create arrays
    vector<value_type> a(N);
    vector<value_type> b(N);
    vector<value_type> c(N);

    auto scalar = value_type{2};

    // do a dry run
    init(a, b, c);
    triad(a, b, c, scalar);
    copy(a, b);

    // do timed runs
    auto triad_time = 0.;
    auto copy_time  = 0.;
    auto add_time  = 0.;
    auto scale_time  = 0.;
    for(auto i=0; i<num_trials; ++i) {
        {
            auto start = clock_type::now();
            triad(a, b, c, scalar);
            triad_time += duration_type(clock_type::now()-start).count();
        }
        {
            auto start = clock_type::now();
            copy(a, b);
            copy_time += duration_type(clock_type::now()-start).count();
        }
        {
            auto start = clock_type::now();
            add(a, b, c);
            add_time += duration_type(clock_type::now()-start).count();
        }
        {
            auto start = clock_type::now();
            scale(a, b, scalar);
            scale_time += duration_type(clock_type::now()-start).count();
        }
    }

    auto bytes_per_array = sizeof(value_type)*N*num_trials;
    auto copy_bytes  = 2*bytes_per_array;
    auto scale_bytes = 2*bytes_per_array;
    auto triad_bytes = 3*bytes_per_array;
    auto add_bytes   = 3*bytes_per_array;

    auto triad_BW  = triad_bytes / triad_time;
    auto copy_BW   = copy_bytes  / copy_time;
    auto add_BW    = add_bytes   / add_time;
    auto scale_BW  = scale_bytes / scale_time;

    std::cout << "triad " << triad_BW/1.e9 << " GB/s" << std::endl;
    std::cout << "copy  " << copy_BW/1.e9  << " GB/s" << std::endl;
    std::cout << "add   " << add_BW/1.e9   << " GB/s" << std::endl;
    std::cout << "scale " << scale_BW/1.e9 << " GB/s" << std::endl;

    return 0;
}

