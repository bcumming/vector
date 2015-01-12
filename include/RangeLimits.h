#pragma once

namespace memory {
// tag for final element in a range
struct end_type {};

// tag for complete range
struct all_type { };

namespace{
    end_type end;
    all_type all;
}
} // namespace memory
