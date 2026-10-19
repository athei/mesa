/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_COMMAND_QUEUE_H
#define MTL_COMMAND_QUEUE_H 1

#include "mtl_types.h"

#include <stdbool.h>
#include <stdint.h>

/* MTL4CommitOptions */
mtl_commit_options *mtl_new_commit_options(void);
void
mtl_commit_options_add_feedback_handler(mtl_commit_options *options,
                                        mtl_feedback_handler_callback callback,
                                        void *data);

/* MTL4CommandQueue */
mtl_command_queue *mtl_new_command_queue(mtl_device *device);

void mtl_command_queue_add_residency_set(mtl_command_queue *cmd_queue,
                                         mtl_residency_set *set);

void mtl_command_queue_remove_residency_set(mtl_command_queue *cmd_queue,
                                            mtl_residency_set *set);

void mtl_signal_event(mtl_command_queue *queue, mtl_event *event,
                      uint64_t value);
void mtl_wait_for_event(mtl_command_queue *queue, mtl_event *event,
                        uint64_t value);

struct mtl_sparse_buffer_mapping {
   uint64_t first_page;
   uint64_t page_count;
   uint64_t heap_page;
   bool map;
};

void mtl_command_queue_update_buffer_mappings(
   mtl_command_queue *queue, mtl_buffer *buffer, mtl_heap *heap,
   const struct mtl_sparse_buffer_mapping *mappings, uint32_t count);

struct mtl_sparse_texture_mapping {
   struct mtl_size origin;
   struct mtl_size size;
   uint32_t level;
   uint32_t slice;
   uint64_t heap_page;
   bool map;
};

void mtl_command_queue_update_texture_mappings(
   mtl_command_queue *queue, mtl_texture *texture, mtl_heap *heap,
   const struct mtl_sparse_texture_mapping *mappings, uint32_t count);

struct mtl_buffer_fill {
   uint64_t offset;
   uint64_t size;
   uint8_t value;
};

void mtl_command_queue_commit_mapping_barrier(
   mtl_command_queue *queue, mtl_device *device, mtl_buffer *fill_buffer,
   const struct mtl_buffer_fill *fills, uint32_t fill_count);

void mtl_command_queue_commit(mtl_command_queue *queue,
                              mtl_command_buffer **command_buffers,
                              uint32_t count, mtl_commit_options *options);

void mtl_command_queue_wait_for_drawable(mtl_command_queue *queue,
                                         void *drawable);

void mtl_command_queue_signal_drawable(mtl_command_queue *queue,
                                       void *drawable);

#endif /* MTL_COMMAND_QUEUE_H */