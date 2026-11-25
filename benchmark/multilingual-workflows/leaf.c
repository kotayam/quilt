#include "contract.h"

#include <inttypes.h>
#include <stdlib.h>

int main(void) {
    char payload[max_payload + 1];
    if (!read_payload(payload))
        return EXIT_FAILURE;

    uint32_t hash = UINT32_C(2166136261);
    for (const char *byte = payload; *byte != '\0'; ++byte)
        hash = (hash ^ (unsigned char)*byte) * UINT32_C(16777619);

    if (printf("%08" PRIx32 "\n", hash) < 0 || fflush(stdout) == EOF)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
