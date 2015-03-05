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

int main(void) {
    const size_type N = size_type{1} << 28;
    auto num_trials = 4;

    std::cout << "arrays of length " << N << std::endl;

    // create arrays
    vector<value_type> a(N);
    vector<value_type> b(N);
    vector<value_type> c(N);

    // initialize values in arrays
    a(all) = 1.;
    b(all) = 2.;
    c(all) = 3.;

    triad(a, b, c);

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

