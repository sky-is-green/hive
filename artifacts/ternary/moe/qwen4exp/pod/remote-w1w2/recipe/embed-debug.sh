#!/bin/bash
# embed-debug.sh - print the exact tensor/offset/size behind the embed read-back abort, then reproduce once.
set -uo pipefail
cd /workspace/llama-qwen4exp
python3 - <<'EOF'
p = 'ggml/src/ggml-backend.cpp'
s = open(p).read()
old = ('    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");\n'
       '\n'
       '    if (backend->iface.get_tensor_async == NULL) {\n'
       '        ggml_backend_synchronize(backend);\n'
       '        ggml_backend_tensor_get(tensor, data, offset, size);')
new = ('    if (offset + size > ggml_nbytes(tensor)) {\n'
       '        fprintf(stderr, "DEBUG read oob: name=%s ne=[%lld,%lld,%lld,%lld] offset=%zu size=%zu nbytes=%zu type=%d\\n",\n'
       '                tensor->name, (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3],\n'
       '                (size_t) offset, (size_t) size, (size_t) ggml_nbytes(tensor), (int) tensor->type);\n'
       '    }\n'
       '    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");\n'
       '\n'
       '    if (backend->iface.get_tensor_async == NULL) {\n'
       '        ggml_backend_synchronize(backend);\n'
       '        ggml_backend_tensor_get(tensor, data, offset, size);')
if old in s:
    open(p, 'w').write(s.replace(old, new, 1))
    print("PATCHED")
elif "DEBUG read oob" in s:
    print("ALREADY")
else:
    print("PATTERN NOT FOUND"); raise SystemExit(1)
EOF
cmake --build build -j"$(nproc)" --target llama-server > /tmp/rebuild.log 2>&1 && echo REBUILD_OK || { echo REBUILD_FAIL; tail -20 /tmp/rebuild.log; exit 1; }
pkill -f "[b]uild/bin/llama-server"; sleep 2
setsid nohup ./build/bin/llama-server -m /root/models/Scion-FlashNext-176B-A6B-Q6K.gguf -ngl 99 -c 4096 -np 1 -t 8 --no-warmup \
    --host 127.0.0.1 --port 8897 --embeddings --pooling last --embd-normalize -1 -fa auto --jinja \
    > server5.log 2>&1 < /dev/null &
for i in $(seq 1 150); do curl -sf http://127.0.0.1:8897/health >/dev/null 2>&1 && break; sleep 2; done
echo "server up; requesting embedding $(date -u)"
curl -s --max-time 180 http://127.0.0.1:8897/embedding -H 'Content-Type: application/json' \
    -d '{"content":"hello world","embd_normalize":-1}' >/dev/null 2>&1
echo "--- DEBUG line:"; grep -a "DEBUG read oob" server5.log | head -3
echo "--- model dims:"; grep -aE "n_embd_out|n_embd +=" server5.log | head -6
