#pragma once

namespace memory {
// tag for final element in a range
struct end_type {};

// tag for complete range
struct all_type { };

namespace{
    // attach the unused attribute so that -Wall won't generate warnings when
    // translation units that include this file don't use these variables
    end_type __attribute__((unused)) end;
    all_type __attribute__((unused)) all;
}
} // namespace memory
