// nlcpy_shim
// 20261112
// GLIBCXX_3.4.29 で追加された関数を、古いlibstdc++の環境で補う(LD_PRELOADで使う)
#include <new>
namespace std {
    [[noreturn]] void __throw_bad_array_new_length() { throw std::bad_array_new_length(); }
}
