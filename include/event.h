#pragma once

#include "definitions.h"

namespace memory {

enum EventStatus {kEventBusy, kEventReady};

// empty event that can be used for synchronous events, or for events that are
// guaranteed to have completed when the event will be queried
class SynchEvent {
public:
    SynchEvent() = default;

    // pause execution in calling thread until event is finished
    // this returns instantly for a synchronous event
    void wait() {};

    EventStatus query() {
        return kEventReady;
    }
};

// abstract base class for an asynchronous event
class AsynchEvent {
public:
    // pause execution in calling thread until event is finished
    // this returns instantly for a synchronous event
    virtual void wait() = 0;

    virtual EventStatus query() = 0;
};

// abstract base class for an asynchronous event
class CUDAEvent
: public AsynchEvent {
public:
    CUDAEvent() = default;

    virtual void
    wait() override {
    }

    virtual EventStatus
    query() override {
        return kEventReady;
    }
};

namespace util {
    template <>
    struct pretty_printer<SynchEvent>{
        static std::string print(const SynchEvent&) {
            return std::string("SynchEvent()");
        }
    };

    template <>
    struct pretty_printer<CUDAEvent>{
        static std::string print(const CUDAEvent& event) {
            return std::string("CUDAEvent()");
        }
    };

    template <>
    struct type_printer<SynchEvent>{
        static std::string print() {
            return std::string("SynchEvent");
        }
    };

    template <>
    struct type_printer<CUDAEvent>{
        static std::string print() {
            return std::string("CUDAEvent");
        }
    };
} // namespace util

} // namespace events