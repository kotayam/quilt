#include "contract.h"

#include <curl/curl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct response {
    char text[checksum_width + 1];
    size_t length;
};

struct leaf_call {
    const char *gateway;
    const char *route;
    const char *payload;
    char checksum[checksum_width + 1];
    bool ok;
};

static size_t collect_response(char *data, size_t size, size_t count,
                               void *context) {
    struct response *response = context;
    size_t remaining = sizeof(response->text) - response->length;
    if (size != 0 && count > remaining / size)
        return 0;
    size_t bytes = size * count;
    memcpy(response->text + response->length, data, bytes);
    response->length += bytes;
    return bytes;
}

static bool valid_checksum(const struct response *response) {
    if (response->length != sizeof(response->text) ||
        response->text[checksum_width] != '\n')
        return false;
    for (size_t i = 0; i < checksum_width; ++i)
        if (memchr("0123456789abcdef", response->text[i], 16) == NULL)
            return false;
    return true;
}

static bool make_rpc(const char *gateway, const char *route, const char *payload,
                     char checksum[static checksum_width + 1]) {
    char url[1024];
    int length = snprintf(url, sizeof(url), "%s/%s", gateway, route);
    if (length < 0 || (size_t)length >= sizeof(url))
        return false;

    CURL *handle = curl_easy_init();
    if (handle == NULL)
        return false;
    struct response response = {0};
    curl_easy_setopt(handle, CURLOPT_URL, url);
    curl_easy_setopt(handle, CURLOPT_NOPROXY, "*");
    curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, 5000L);
    curl_easy_setopt(handle, CURLOPT_POSTFIELDS, payload);
    curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE, (long)strlen(payload));
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, collect_response);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &response);
    CURLcode code = curl_easy_perform(handle);
    long status = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(handle);
    if (code != CURLE_OK || status != 200 || !valid_checksum(&response))
        return false;

    memcpy(checksum, response.text, checksum_width);
    checksum[checksum_width] = '\0';
    return true;
}

static void *call_leaf(void *argument) {
    struct leaf_call *call = argument;
    call->ok = make_rpc(call->gateway, call->route, call->payload, call->checksum);
    return NULL;
}

enum { leaf_count = 3 };

static bool run_parallel(struct leaf_call calls[static leaf_count]) {
    pthread_t threads[leaf_count];
    size_t started = 0;
    while (started < leaf_count &&
           pthread_create(&threads[started], NULL, call_leaf, &calls[started]) == 0)
        ++started;
    for (size_t i = 0; i < started; ++i)
        if (pthread_join(threads[i], NULL) != 0)
            exit(EXIT_FAILURE);
    return started == leaf_count;
}

int main(void) {
    char mode[8];
    char payload[max_payload + 1];
    if (fgets(mode, sizeof(mode), stdin) == NULL ||
        (strcmp(mode, "sync\n") != 0 && strcmp(mode, "async\n") != 0) ||
        !read_payload(payload)) {
        fputs("expected a sync or async line followed by a printable ASCII payload\n", stderr);
        return EXIT_FAILURE;
    }
    const char *gateway = getenv("BENCHMARK_GATEWAY");
    if (gateway == NULL)
        gateway = "http://router.fission.svc.cluster.local:80";
    const char *routes[leaf_count] = {
        "multilingual-leaf-a", "multilingual-leaf-b", "multilingual-leaf-c",
    };
    struct leaf_call calls[leaf_count];
    for (size_t i = 0; i < leaf_count; ++i)
        calls[i] = (struct leaf_call){.gateway = gateway, .route = routes[i],
                                      .payload = payload};

    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK)
        return EXIT_FAILURE;
    bool complete = true;
    if (strcmp(mode, "async\n") == 0)
        complete = run_parallel(calls);
    else
        for (size_t i = 0; i < leaf_count; ++i)
            call_leaf(&calls[i]);
    curl_global_cleanup();
    if (!complete) {
        fputs("could not launch all leaf calls\n", stderr);
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < leaf_count; ++i) {
        if (!calls[i].ok) {
            fprintf(stderr, "invalid or failed response from %s\n", calls[i].route);
            return EXIT_FAILURE;
        }
    }
    if (printf("[\"%s\",\"%s\",\"%s\"]\n", calls[0].checksum,
               calls[1].checksum, calls[2].checksum) < 0 || fflush(stdout) == EOF)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
