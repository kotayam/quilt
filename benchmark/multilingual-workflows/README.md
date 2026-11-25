# Multilingual workflow fixtures

One C caller fans out to three interchangeable leaf routes. The caller reads
a mode line containing either `sync` or `async`, followed by a newline and
the payload. Sync calls the routes in order and async calls them concurrently.
Both modes return a JSON array in route order.

Register `/multilingual-leaf-a`, `/multilingual-leaf-b`, and
`/multilingual-leaf-c` at the gateway and map each to a leaf of your choice.
Each leaf accepts up to 4,096 printable ASCII bytes, including an empty
payload, on stdin through EOF. It returns the 32-bit FNV-1a checksum as eight
lowercase hexadecimal digits followed by a newline.

Run `make` with a C compiler, libcurl development files, `pkg-config`,
`rustc`, and the standard Go toolchain. The Python leaf runs directly with
`/usr/bin/python3`. Build outputs are in `build/`.

```sh
printf 'hello' | build/leaf-python
# 4f9f2cab

printf 'async\nhello' | BENCHMARK_GATEWAY=http://127.0.0.1:8080 build/caller
# ["4f9f2cab","4f9f2cab","4f9f2cab"]
```

Set `BENCHMARK_GATEWAY` to a base URL without a trailing slash. The default
is `http://router.fission.svc.cluster.local:80`.
