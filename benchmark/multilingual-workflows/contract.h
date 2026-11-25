#ifndef MULTILINGUAL_CONTRACT_H
#define MULTILINGUAL_CONTRACT_H

#include <stdbool.h>
#include <stdio.h>

enum { max_payload = 4096, checksum_width = 8 };

static inline bool read_payload(char payload[static max_payload + 1]) {
    size_t length = fread(payload, 1, max_payload + 1, stdin);
    if (ferror(stdin) || length > max_payload)
        return false;
    for (size_t i = 0; i < length; ++i)
        if (payload[i] < ' ' || payload[i] > '~')
            return false;
    payload[length] = '\0';
    return true;
}

#endif
