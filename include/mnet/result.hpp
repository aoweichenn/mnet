// mnet 公共契约头：Result<T> 与 Status。
//
// 不使用异常（项目约定），失败载荷为 Failure（见 error.hpp）。
// 刻意不提供 value_or()：静默吞掉错误是本库最不希望看到的用法。

#ifndef MNET_RESULT_HPP
#define MNET_RESULT_HPP

#include <cstddef>

#include <new>
#include <type_traits>
#include <utility>

#include <mnet/error.hpp>

namespace mnet {

template<typename T>
class Result {
    static_assert(!std::is_reference_v<T>, "Result<T> does not support reference types");
    static_assert(!std::is_void_v<T>, "use Status for void results");

    // 存储区按 T 的真实大小推导，**不是**固定字节数。
    //
    // 早先写成 aligned_storage_t<8 * sizeof(void*)>（64 字节）：本库当前的
    // T（span / vector / Archive）都 <= 24 字节，看起来够用，但这是巧合。
    // 换成任何大于 64 字节的 T，placement new 会直接写穿栈；
    // mpickle 的 Value 有 384 字节，就踩到了（ASan: stack-buffer-overflow）。
    using Storage = std::aligned_storage_t<sizeof(T), alignof(T)>;

public:
    Result(T value) noexcept(std::is_nothrow_move_constructible_v<T>) : ok_(true) {
        ::new (static_cast<void*>(&this->storage_)) T(std::move(value));
    }

    Result(Failure failure) noexcept : ok_(false), failure_(failure) {}

    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;

    // 可移动：没有它的话 `return local_result;` 无法编译（用户声明了析构函数，
    // 编译器不会再隐式生成移动构造），调用方只能写 Result<T>{x.value()}。
    Result(Result&& other) noexcept(std::is_nothrow_move_constructible_v<T>) : ok_(other.ok_) {
        if (this->ok_) {
            ::new (static_cast<void*>(&this->storage_)) T(std::move(other.value()));
        } else {
            this->failure_ = other.failure_;
        }
    }

    Result& operator=(Result&& other) noexcept(std::is_nothrow_move_constructible_v<T> &&
                                               std::is_nothrow_move_assignable_v<T>) {
        if (this != &other) {
            if (this->ok_) {
                this->get()->~T();
            }
            this->ok_ = other.ok_;
            if (this->ok_) {
                ::new (static_cast<void*>(&this->storage_)) T(std::move(other.value()));
            } else {
                this->failure_ = other.failure_;
            }
        }
        return *this;
    }

    ~Result() {
        if (this->ok_) {
            this->get()->~T();
        }
    }

    [[nodiscard]] bool ok() const noexcept { return this->ok_; }

    [[nodiscard]] explicit operator bool() const noexcept { return this->ok_; }

    // Precondition: ok(). Calling on a failure is undefined behaviour.
    [[nodiscard]] T& value() & noexcept { return *this->get(); }

    [[nodiscard]] const T& value() const& noexcept { return *this->get(); }

    // Returns nullptr on failure instead of dereferencing.
    [[nodiscard]] const T* operator->() const noexcept { return this->ok_ ? this->get() : nullptr; }

    [[nodiscard]] const Failure& failure() const noexcept { return this->failure_; }

private:
    [[nodiscard]] T* get() const noexcept {
        return std::launder(reinterpret_cast<T*>(const_cast<Storage*>(&this->storage_)));
    }

    bool ok_;
    Storage storage_{};
    Failure failure_{};
};

// Void specialisation: carries only success or failure.
class Status {
public:
    Status() noexcept = default;
    Status(Failure failure) noexcept : failure_(failure), ok_(false) {}

    [[nodiscard]] bool ok() const noexcept { return this->ok_; }

    [[nodiscard]] explicit operator bool() const noexcept { return this->ok_; }

    [[nodiscard]] const Failure& failure() const noexcept { return this->failure_; }

private:
    Failure failure_{};
    bool ok_ = true;
};

}  // namespace mnet

#endif  // MNET_RESULT_HPP
