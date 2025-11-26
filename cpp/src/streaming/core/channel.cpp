/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <iomanip>
#include <sstream>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

namespace rapidsmpf::streaming {

coro::task<bool> Channel::send(Message msg) {
    auto start_time = std::chrono::steady_clock::now();
    auto seq_num = msg.sequence_number();

    auto result = co_await rb_.produce(sm_->insert(std::move(msg)));

    auto end_time = std::chrono::steady_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
            .count();

    bool success = result == coro::ring_buffer_result::produce::produced;

    if (logger_) {
        std::ostringstream channel_addr;
        channel_addr << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(this);
        logger_->log(
            rapidsmpf::Communicator::Logger::LOG_LEVEL::TRACE,
            "Channel ",
            channel_addr.str(),
            " send msg seq=",
            seq_num,
            " duration=",
            duration_us,
            "us result=",
            (success ? "success" : "failed")
        );
    }

    co_return success;
}

coro::task<Message> Channel::receive() {
    auto start_time = std::chrono::steady_clock::now();

    auto msg = co_await rb_.consume();

    auto end_time = std::chrono::steady_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
            .count();

    if (msg.has_value()) {
        auto extracted_msg = sm_->extract(*msg);

        if (logger_) {
            std::ostringstream channel_addr;
            channel_addr << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(this);
            logger_->log(
                rapidsmpf::Communicator::Logger::LOG_LEVEL::TRACE,
                "Channel ",
                channel_addr.str(),
                " recv msg seq=",
                extracted_msg.sequence_number(),
                " duration=",
                duration_us,
                "us"
            );
        }

        co_return extracted_msg;
    } else {
        if (logger_) {
            std::ostringstream channel_addr;
            channel_addr << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(this);
            logger_->log(
                rapidsmpf::Communicator::Logger::LOG_LEVEL::TRACE,
                "Channel ",
                channel_addr.str(),
                " recv empty (shutdown) duration=",
                duration_us,
                "us"
            );
        }

        co_return Message{};
    }
}

Node Channel::drain(std::unique_ptr<coro::thread_pool>& executor) {
    return rb_.shutdown_drain(executor);
}

Node Channel::shutdown() {
    return rb_.shutdown();
}

bool Channel::empty() const noexcept {
    return rb_.empty();
}

}  // namespace rapidsmpf::streaming
