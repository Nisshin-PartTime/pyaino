# nlcpy_glibcxx_compat
# 20261112
# PyPIのNLCPy(GCC 11以降でビルド)を、libstdc++が古い(GCC 8, GLIBCXX_3.4.25)RHEL8系で
# 読み込めるようにする。NLCPyが新しいlibstdc++に求めるのは
# std::__throw_bad_array_new_length() (GLIBCXX_3.4.29) だけなので、
#   1) その記号の版指定を外す(.gnu.version を VER_NDX_GLOBAL に)
#   2) GLIBCXX_3.4.29 の版要求を weak にする(glibcは見つからなくても読み込みを続ける)
# を拡張モジュールに施し、記号そのものは nlcpy_shim.cpp で作る共有ライブラリを
# LD_PRELOAD して補う
#   python nlcpy_glibcxx_compat.py <site-packages/nlcpy>
import os, struct, sys

VERSION = b'GLIBCXX_3.4.29'
SYMBOL = b'_ZSt28__throw_bad_array_new_lengthv'
SHT_DYNSYM, SHT_GNU_VERSYM, SHT_GNU_VERNEED = 11, 0x6fffffff, 0x6ffffffe

def cstr(b, off):
    return bytes(b[off: b.index(0, off)])

def patch(path):
    b = bytearray(open(path, 'rb').read())
    if b[:4] != b'\x7fELF' or b[4] != 2:   # ELF64のみ
        return 0
    shoff, = struct.unpack_from('<Q', b, 0x28)
    shentsize, shnum = struct.unpack_from('<HH', b, 0x3A)
    secs = [struct.unpack_from('<IIQQQQIIQQ', b, shoff + i * shentsize) for i in range(shnum)]
    n = 0
    dynsym = next((s for s in secs if s[1] == SHT_DYNSYM), None)
    versym = next((s for s in secs if s[1] == SHT_GNU_VERSYM), None)
    if dynsym and versym:                  # 1) 記号の版指定を外す
        strtab = secs[dynsym[6]][4]
        for i in range(dynsym[5] // 24):
            name, = struct.unpack_from('<I', b, dynsym[4] + i * 24)
            if cstr(b, strtab + name) == SYMBOL:
                struct.pack_into('<H', b, versym[4] + 2 * i, 1); n += 1
    for s in secs:                         # 2) 版要求をweakに
        if s[1] != SHT_GNU_VERNEED:
            continue
        strtab, off = secs[s[6]][4], s[4]
        while True:
            _, cnt, _, aux, nxt = struct.unpack_from('<HHIII', b, off)
            a = off + aux
            for _ in range(cnt):
                _, flags, _, name, anxt = struct.unpack_from('<IHHII', b, a)
                if cstr(b, strtab + name) == VERSION:
                    struct.pack_into('<H', b, a + 4, flags | 2); n += 1
                a += anxt
            if nxt == 0:
                break
            off += nxt
    if n:
        open(path, 'wb').write(b)
    return n

if __name__ == '__main__':
    root = sys.argv[1]
    total = 0
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith('.so'):
                p = os.path.join(d, f)
                if VERSION in open(p, 'rb').read():
                    total += 1 if patch(p) else 0
    print(f'patched {total} extension modules under {root}')
