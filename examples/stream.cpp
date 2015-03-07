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
           vector<T> const& c)
{
    auto const N = a.size();
    #pragma omp parallel for
    for(auto i=size_type{0}; i<N; ++i) {
        a[i] += b[i] * c[i];
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

int main(void) {
    auto const N = size_type{1} << 25;
    auto num_trials = 4;

    std::cout << "arrays of length " << N << std::endl;

    // create arrays
    vector<value_type> a(N);
    vector<value_type> b(N);
    vector<value_type> c(N);

    // initialize values in arrays
    init(a, b, c);

    // do a dry run
    triad(a, b, c);

    // do timed runs
    auto start = clock_type::now();
    for(auto i=0; i<num_trials; ++i) {
        triad(a, b, c);
    }
    auto total_time  = duration_type(clock_type::now() - start).count();
    auto total_bytes = sizeof(value_type)*4*N*num_trials;
    auto BW          = total_bytes / total_time;

    std::cout << "that took " << total_time << " seconds" << std::endl;
    std::cout << "bandwidth " << BW/1.e9 << " GB/s" << std::endl;

    return 0;
}

