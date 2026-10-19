/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_command_queue.h"

#include <Metal/MTLDevice.h>
#include <Metal/MTLCommandQueue.h>

/* MTL4CommitOptions */
mtl_commit_options *
mtl_new_commit_options(void)
{
   @autoreleasepool {
      return [MTL4CommitOptions new];
   }
}

void
mtl_commit_options_add_feedback_handler(mtl_commit_options *options,
                                        mtl_feedback_handler_callback callback,
                                        void *data)
{
   @autoreleasepool {
      MTL4CommitOptions* opt = (MTL4CommitOptions *)options;
      [opt addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
         @autoreleasepool {
            NSError *error = feedback.error;
            struct mtl_feedback_data feedback_data = (struct mtl_feedback_data){
               .user_data = data,
               .error_message =
                  error ? error.localizedDescription.UTF8String : NULL,
               .gpu_start = feedback.GPUStartTime,
               .gpu_end = feedback.GPUEndTime,
               .error = error
                           ? (enum mtl_command_queue_error)error.code
                           : MTL_COMMAND_QUEUE_ERROR_NONE,
            };
            callback(&feedback_data);
         }
      }];
   }
}

/* MTL4CommandQueue */
mtl_command_queue *
mtl_new_command_queue(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev newMTL4CommandQueue];
   }
}

void
mtl_command_queue_add_residency_set(mtl_command_queue *cmd_queue,
                                    mtl_residency_set *set)
{
   @autoreleasepool {
      id<MTL4CommandQueue> queue = (id<MTL4CommandQueue>)cmd_queue;
      id<MTLResidencySet> s = (id<MTLResidencySet>)set;
      return [queue addResidencySet:s];
   }
}

void
mtl_command_queue_remove_residency_set(mtl_command_queue *cmd_queue,
                                       mtl_residency_set *set)
{
   @autoreleasepool {
      id<MTL4CommandQueue> queue = (id<MTL4CommandQueue>)cmd_queue;
      id<MTLResidencySet> s = (id<MTLResidencySet>)set;
      return [queue removeResidencySet:s];
   }
}

void
mtl_signal_event(mtl_command_queue *queue, mtl_event *event, uint64_t value)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLEvent> e = (id<MTLEvent>)event;
      [q signalEvent:e value:value];
   }
}

void
mtl_wait_for_event(mtl_command_queue *queue, mtl_event *event, uint64_t value)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLEvent> e = (id<MTLEvent>)event;
      [q waitForEvent:e value:value];
   }
}

void
mtl_command_queue_update_buffer_mappings(
   mtl_command_queue *queue, mtl_buffer *buffer, mtl_heap *heap,
   const struct mtl_sparse_buffer_mapping *mappings, uint32_t count)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      MTL4UpdateSparseBufferMappingOperation ops[count];
      for (uint32_t i = 0; i < count; i++) {
         ops[i] = (MTL4UpdateSparseBufferMappingOperation){
            .mode = mappings[i].map ? MTLSparseTextureMappingModeMap
                                    : MTLSparseTextureMappingModeUnmap,
            .bufferRange =
               NSMakeRange(mappings[i].first_page, mappings[i].page_count),
            .heapOffset = mappings[i].heap_page,
         };
      }
      [q updateBufferMappings:(id<MTLBuffer>)buffer
                         heap:(id<MTLHeap>)heap
                   operations:ops
                        count:count];
   }
}

void
mtl_command_queue_update_texture_mappings(
   mtl_command_queue *queue, mtl_texture *texture, mtl_heap *heap,
   const struct mtl_sparse_texture_mapping *mappings, uint32_t count)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      MTL4UpdateSparseTextureMappingOperation ops[count];
      for (uint32_t i = 0; i < count; i++) {
         const struct mtl_sparse_texture_mapping *m = &mappings[i];
         ops[i] = (MTL4UpdateSparseTextureMappingOperation){
            .mode = m->map ? MTLSparseTextureMappingModeMap
                           : MTLSparseTextureMappingModeUnmap,
            .textureRegion = MTLRegionMake3D(m->origin.x, m->origin.y,
                                             m->origin.z, m->size.x,
                                             m->size.y, m->size.z),
            .textureLevel = m->level,
            .textureSlice = m->slice,
            .heapOffset = m->heap_page,
         };
      }
      [q updateTextureMappings:(id<MTLTexture>)texture
                          heap:(id<MTLHeap>)heap
                    operations:ops
                         count:count];
   }
}

void
mtl_command_queue_commit_mapping_barrier(mtl_command_queue *queue,
                                         mtl_device *device,
                                         mtl_buffer *fill_buffer,
                                         const struct mtl_buffer_fill *fills,
                                         uint32_t fill_count)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLDevice> dev = (id<MTLDevice>)device;
      id<MTL4CommandAllocator> allocator = [dev newCommandAllocator];
      id<MTL4CommandBuffer> cmd_buf = [dev newCommandBuffer];
      [cmd_buf beginCommandBufferWithAllocator:allocator];
      id<MTL4ComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
      [enc barrierAfterQueueStages:MTLStageResourceState
                      beforeStages:MTLStageDispatch | MTLStageBlit
                 visibilityOptions:MTL4VisibilityOptionDevice];
      for (uint32_t i = 0; i < fill_count; i++) {
         [enc fillBuffer:(id<MTLBuffer>)fill_buffer
                   range:NSMakeRange(fills[i].offset, fills[i].size)
                   value:fills[i].value];
      }
      [enc barrierAfterStages:MTLStageDispatch | MTLStageBlit
            beforeQueueStages:MTLStageVertex | MTLStageFragment |
                              MTLStageDispatch | MTLStageBlit
            visibilityOptions:MTL4VisibilityOptionDevice];
      [enc endEncoding];
      [cmd_buf endCommandBuffer];

      MTL4CommitOptions *options = [[MTL4CommitOptions new] autorelease];
      [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
         [cmd_buf release];
         [allocator release];
      }];
      [q commit:&cmd_buf count:1 options:options];
   }
}

void
mtl_command_queue_commit(mtl_command_queue *queue,
                         mtl_command_buffer **command_buffers, uint32_t count,
                         mtl_commit_options *options)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTL4CommandBuffer> *cmds = (id<MTL4CommandBuffer> *)command_buffers;
      MTL4CommitOptions *opt = (MTL4CommitOptions *)options;
      [q commit:cmds count:count options:opt];
   }
}

void
mtl_command_queue_wait_for_drawable(mtl_command_queue *queue, void *drawable)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLDrawable> d = (id<MTLDrawable>)drawable;
      [q waitForDrawable:d];
   }
}

void
mtl_command_queue_signal_drawable(mtl_command_queue *queue, void *drawable)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLDrawable> d = (id<MTLDrawable>)drawable;
      [q signalDrawable:d];
   }
}
