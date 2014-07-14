#pragma once

namespace dispatch {

enum EventStatus = {kEventBusy, kEventReady};

// empty event that can be used for synchronous events, or for events that are
// guarenteed to have completed when the event will be queried
class EventBase {
public:
    EventBase() = default;

    void wait() {};
    EventStatus query() {
        return kEventReady;
    }
};

} // namespace events