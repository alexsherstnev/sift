# SIFT
PROJECT := sift
BUILD_TYPE ?= DEBUG

# Directories
BUILD_DIR := build
INCLUDE_DIR := include
SRC_DIR := src
CLI_DIR := cli

BIN_DIR := $(BUILD_DIR)/bin
LIB_DIR := $(BUILD_DIR)/lib
OBJ_DIR := $(BUILD_DIR)/obj

# Toolchain
CC := gcc
AR := ar # TODO: Need think about cross-platform?
RM := rm
MD := mkdir -p
CFLAGS := -Wall -Wextra -MMD -MP -I$(INCLUDE_DIR)
LDFLAGS := -lm -ldl

# Build Type
ifeq ($(BUILD_TYPE), RELEASE)
	CFLAGS += -O3 -DNDEBUG
else
	CFLAGS += -O0 -g
endif

# TODO: SSE?
# USE_SSE ?= 0

# Files
LIB_SOURCES := $(wildcard $(SRC_DIR)/*.c)
LIB_OBJ := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/lib/%.o, $(LIB_SOURCES))
LIB_TARGET := $(LIB_DIR)/lib$(PROJECT).a

CLI_SOURCES := $(wildcard $(CLI_DIR)/*.c)
CLI_OBJ := $(patsubst $(CLI_DIR)/%.c, $(OBJ_DIR)/cli/%.o, $(CLI_SOURCES))
CLI_TARGET := $(BIN_DIR)/$(PROJECT)

# Phony Targets
.PHONY: all lib cli tests clean

all: lib cli tests

lib: $(LIB_TARGET)

cli: $(CLI_TARGET)

tests: $(TESTS_TARGET)

clean:
	$(RM) -rf $(BUILD_DIR)

# Library Rules
$(LIB_TARGET): $(LIB_OBJ)
	@$(MD) $(dir $@)
	$(AR) rcs $@ $^

$(OBJ_DIR)/lib/%.o: $(SRC_DIR)/%.c
	@$(MD) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# CLI Rules
$(CLI_TARGET): $(CLI_OBJ) $(LIB_TARGET)
	@$(MD) $(dir $@)
	$(CC) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/cli/%.o: $(CLI_DIR)/%.c
	@$(MD) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Dependencies
-include $(LIB_OBJ:.o=.d)
-include $(CLI_OBJ:.o=.d)

