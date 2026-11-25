CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra
RUSTC ?= rustc
GO ?= go
CURL_CFLAGS ?= $(shell pkg-config --cflags libcurl)
CURL_LIBS ?= $(shell pkg-config --libs libcurl)
BUILD ?= build

.PHONY: all
all: $(BUILD)/caller $(BUILD)/leaf-c $(BUILD)/leaf-rust $(BUILD)/leaf-python $(BUILD)/leaf-go

$(BUILD):
	mkdir -p $@

$(BUILD)/caller: caller.c contract.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(CURL_CFLAGS) -fPIE -pie -pthread $< $(LDFLAGS) $(CURL_LIBS) -o $@

$(BUILD)/leaf-c: leaf.c contract.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIE -pie $< $(LDFLAGS) -o $@

$(BUILD)/leaf-rust: leaf.rs | $(BUILD)
	$(RUSTC) -O -C relocation-model=pic $< -o $@

$(BUILD)/leaf-python: leaf.py | $(BUILD)
	install -m 755 $< $@

$(BUILD)/leaf-go: leaf.go | $(BUILD)
	CGO_ENABLED=0 $(GO) build -buildmode=pie -o $@ $<
