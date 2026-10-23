/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_queue.h"
#include "kk_buffer.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_device_memory.h"
#include "kk_image.h"
#include "kk_bo.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"
#include "kk_sync.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"

#include "vk_cmd_queue.h"

struct kk_commit_data {
   struct kk_queue *queue;
   struct kk_cmd_buffer *cmd;
};

void
kk_queue_wait_for_idle(struct kk_queue *queue)
{
   mtx_lock(&queue->mutex);
   while (queue->work_in_flight > 0u)
      cnd_wait(&queue->cond, &queue->mutex);
   mtx_unlock(&queue->mutex);
}

static void
kk_queue_work_done_locked(struct kk_queue *queue)
{
   assert(queue->work_in_flight > 0u);
   if (--queue->work_in_flight == 0u)
      cnd_broadcast(&queue->cond);
}

static void
kk_queue_release_cmd_buffer_locked(struct kk_queue *queue,
                                   struct kk_cmd_buffer *cmd)
{
   kk_cmd_buffer_ops.reset(&cmd->vk, 0u);
   util_dynarray_append(&queue->free_cmd_buffers, &cmd->vk);
}

static void
check_device_lost(struct kk_device *dev, struct mtl_feedback_data *data)
{
   if (data->error != MTL_COMMAND_QUEUE_ERROR_NONE) {
      vk_device_set_lost(
         &dev->vk, "Command queue error: %s, with message \"%s\"",
         mtl_command_queue_error_to_string(data->error), data->error_message);
      memset((uint8_t *)dev->atomic64.bo->cpu + KK_ATOMIC64_LOCK_OFFSET, 0,
             KK_ATOMIC64_ARENA_SIZE - KK_ATOMIC64_LOCK_OFFSET);
   }

   const uint32_t *atomic64 = dev->atomic64.bo->cpu;
   if (p_atomic_read(&atomic64[KK_ATOMIC64_TIMEOUT]))
      vk_device_set_lost(&dev->vk,
                         "A 64-bit atomic timed out waiting for its lock");
}

static void
commit_callback(struct mtl_feedback_data *data)
{
   struct kk_device *dev = (struct kk_device *)data->user_data;
   check_device_lost(dev, data);

   mtx_lock(&dev->queue.mutex);
   kk_queue_work_done_locked(&dev->queue);
   mtx_unlock(&dev->queue.mutex);
}

static void
rerecord_commit_callback(struct mtl_feedback_data *data)
{
   struct kk_commit_data *commit = (struct kk_commit_data *)data->user_data;
   struct kk_queue *queue = commit->queue;
   struct kk_device *dev = kk_queue_device(queue);

   check_device_lost(dev, data);

   /* Completion callbacks are called from multiple threads, so we need to
    * ensure the access to queue resources is safe. */
   mtx_lock(&queue->mutex);

   kk_queue_release_cmd_buffer_locked(queue, commit->cmd);
   vk_free(&dev->vk.alloc, commit);

   assert(queue->commits_in_flight > 0u);
   if (--queue->commits_in_flight == 0u) {
      kk_ResetCommandPool(kk_device_to_handle(dev),
                          kk_cmd_pool_to_handle(queue->cmd_pool), 0u);
      cnd_broadcast(&queue->cond);
   }
   kk_queue_work_done_locked(queue);

   mtx_unlock(&queue->mutex);
}

static bool
kk_cmd_buffer_has_work(struct kk_cmd_buffer *cmd)
{
   return cmd->metal.cmd_buf;
}

static void
kk_queue_commit(struct kk_queue *queue, struct kk_cmd_buffer *cmd,
                mtl_feedback_handler_callback callback, void *user_data)
{
   assert(kk_cmd_buffer_has_work(cmd));

   mtl_commit_options *options = mtl_new_commit_options();
   mtl_commit_options_add_feedback_handler(options, callback, user_data);

   mtl_command_queue_commit(queue->mtl_handle, &cmd->metal.cmd_buf, 1u,
                            options);
   mtl_release(options);
}

static VkResult
rerecord_and_commit_cmd_buffer(struct kk_queue *queue,
                               struct kk_cmd_buffer *cmd)
{
   struct kk_device *dev = kk_queue_device(queue);
   struct vk_command_buffer *vk_cmd = NULL;
   struct kk_cmd_buffer *rerecord;
   VkResult result = VK_SUCCESS;

   /* Completion callbacks are called from multiple threads, so we need to
    * ensure the access to queue resources is safe. */
   mtx_lock(&queue->mutex);

   if (util_dynarray_num_elements(&queue->free_cmd_buffers,
                                  struct vk_command_buffer *) > 0u) {
      vk_cmd = util_dynarray_pop(&queue->free_cmd_buffers,
                                 struct vk_command_buffer *);
   } else {
      result = kk_cmd_buffer_ops.create(
         &queue->cmd_pool->vk, VK_COMMAND_BUFFER_LEVEL_PRIMARY, &vk_cmd);
      if (result != VK_SUCCESS)
         goto unlock;
   }

   rerecord = container_of(vk_cmd, struct kk_cmd_buffer, vk);
   VkCommandBuffer rerecord_handle = kk_cmd_buffer_to_handle(rerecord);
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   result = kk_BeginCommandBuffer(rerecord_handle, &begin_info);
   if (result != VK_SUCCESS)
      goto release;

   /* The drawable is not part of the recorded commands. */
   if (cmd->drawable)
      mtl_command_buffer_use_drawable_residency(rerecord->metal.cmd_buf,
                                                cmd->drawable);
   vk_cmd_queue_execute(&cmd->vk.cmd_queue, rerecord_handle,
                        &dev->vk.dispatch_table);

   result = kk_EndCommandBuffer(rerecord_handle);
   if (result != VK_SUCCESS)
      goto release;

   if (!kk_cmd_buffer_has_work(rerecord))
      goto release;

   struct kk_commit_data *commit = vk_alloc(&dev->vk.alloc, sizeof(*commit), 8,
                                            VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (commit == NULL) {
      result = vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto release;
   }
   commit->queue = queue;
   commit->cmd = rerecord;

   /* Need to ensure the new buffers allocated at record are resident. */
   kk_device_make_resources_resident(dev);

   queue->commits_in_flight++;
   queue->work_in_flight++;
   kk_queue_commit(queue, rerecord, rerecord_commit_callback, commit);

   mtx_unlock(&queue->mutex);
   return VK_SUCCESS;

release:
   kk_queue_release_cmd_buffer_locked(queue, rerecord);
unlock:
   mtx_unlock(&queue->mutex);
   return result;
}

static void
kk_queue_bind_buffer(struct kk_queue *queue, struct kk_buffer *buffer,
                     const VkSparseMemoryBind *bind,
                     struct util_dynarray *fills)
{
   VK_FROM_HANDLE(kk_device_memory, mem, bind->memory);

   struct mtl_sparse_buffer_mapping mapping = {
      .first_page = bind->resourceOffset / KK_SPARSE_PAGE_SIZE,
      .page_count = DIV_ROUND_UP(bind->size, KK_SPARSE_PAGE_SIZE),
      .heap_page = mem ? bind->memoryOffset / KK_SPARSE_PAGE_SIZE : 0u,
      .map = mem != NULL,
   };
   mtl_command_queue_update_buffer_mappings(
      queue->mtl_handle, buffer->metal.handle,
      mem ? mem->bo->mtl_handle : NULL, &mapping, 1u);

   if (buffer->residency.pages) {
      struct mtl_buffer_fill fill = {
         .offset = buffer->residency.offset + mapping.first_page,
         .size = mapping.page_count,
         .value = mapping.map,
      };
      util_dynarray_append(fills, fill);
   }
}

struct kk_texture_mapping {
   mtl_texture *texture;
   mtl_heap *heap;
   struct mtl_sparse_texture_mapping mapping;
};

static void
kk_queue_map_texture(struct kk_queue *queue, struct util_dynarray *deferred,
                     mtl_texture *texture, struct kk_device_memory *mem,
                     struct mtl_sparse_texture_mapping mapping)
{
   mapping.map = mem != NULL;
   if (!mapping.map)
      mapping.heap_page = 0u;

   struct kk_texture_mapping op = {
      .texture = texture,
      .heap = mem ? mem->bo->mtl_handle : NULL,
      .mapping = mapping,
   };

   if (deferred) {
      util_dynarray_append(deferred, op);
      return;
   }

   mtl_command_queue_update_texture_mappings(queue->mtl_handle, op.texture,
                                             op.heap, &op.mapping, 1u);
}

static struct util_dynarray *
kk_plane_deferred(struct kk_image_plane *plane, struct util_dynarray *deferred)
{
   return plane->mtl_handle_residency ? NULL : deferred;
}

static void
kk_queue_map_tail_page(struct kk_queue *queue, struct util_dynarray *deferred,
                       struct kk_image_plane *plane, mtl_texture *texture,
                       uint32_t texture_tail_level,
                       uint32_t layer, uint64_t page,
                       struct kk_device_memory *mem, uint64_t heap_page)
{
   struct mtl_sparse_texture_mapping mapping = {
      .size = {1, 1, 1},
      .slice = layer,
      .heap_page = heap_page,
   };

   for (uint32_t level = plane->sparse.tail_level; level < texture_tail_level;
        level++) {
      struct mtl_size tiles = kk_sparse_level_tiles(plane, level);
      uint64_t count = tiles.x * tiles.y * tiles.z;
      if (page < count) {
         mapping.level = level;
         mapping.origin = (struct mtl_size){
            page % tiles.x, (page / tiles.x) % tiles.y,
            page / (tiles.x * tiles.y)};
         kk_queue_map_texture(queue, deferred, texture, mem, mapping);
         return;
      }
      page -= count;
   }

   if (texture_tail_level >= plane->layout.levels)
      return;

   mapping.level = texture_tail_level;
   mapping.origin = (struct mtl_size){page, 0, 0};
   kk_queue_map_texture(queue, deferred, texture, mem, mapping);
}

static void
kk_queue_bind_image_tail(struct kk_queue *queue, struct util_dynarray *deferred,
                         struct kk_image_plane *plane, uint32_t layer, uint64_t page, uint64_t count,
                         struct kk_device_memory *mem, uint64_t heap_page)
{
   const struct kk_sparse_layout *sparse = &plane->sparse;
   const uint64_t tail_pages = sparse->tail_size_B / KK_SPARSE_PAGE_SIZE;
   const uint64_t residency_pages =
      plane->mtl_handle_residency
         ? kk_sparse_tail_pages(plane, plane->mtl_handle_residency,
                                sparse->residency_tail_level)
         : 0u;

   for (uint64_t i = 0; i < count; i++, page++, heap_page++) {
      kk_queue_map_tail_page(queue, kk_plane_deferred(plane, deferred), plane,
                             plane->mtl_handle, sparse->texture_tail_level,
                             layer, page, mem, heap_page);

      uint64_t last = page + 1 == tail_pages ? residency_pages : page + 1;
      for (uint64_t r = page; r < MIN2(last, residency_pages); r++)
         kk_queue_map_tail_page(queue, deferred, plane,
                                plane->mtl_handle_residency,
                                sparse->residency_tail_level, layer, r, mem,
                                heap_page);
   }
}

static void
kk_queue_bind_image_opaque(struct kk_queue *queue,
                           struct util_dynarray *deferred,
                           struct kk_image *image,
                           const VkSparseMemoryBind *bind)
{
   VK_FROM_HANDLE(kk_device_memory, mem, bind->memory);
   struct kk_image_plane *plane = &image->planes[0];
   const struct kk_sparse_layout *sparse = &plane->sparse;

   uint64_t page = bind->resourceOffset / KK_SPARSE_PAGE_SIZE;
   uint64_t page_count = DIV_ROUND_UP(bind->size, KK_SPARSE_PAGE_SIZE);
   uint64_t heap_page = bind->memoryOffset / KK_SPARSE_PAGE_SIZE;
   const uint64_t layer_pages = sparse->layer_stride_B / KK_SPARSE_PAGE_SIZE;

   while (page_count > 0 && layer_pages > 0) {
      uint32_t layer = page / layer_pages;
      uint64_t layer_page = page % layer_pages;
      if (layer >= plane->layout.layers)
         break;

      uint32_t level = 0;
      while (level < sparse->tail_level &&
             layer_page >=
                sparse->level_offset_B[level + 1] / KK_SPARSE_PAGE_SIZE)
         level++;

      uint64_t level_page =
         layer_page - sparse->level_offset_B[level] / KK_SPARSE_PAGE_SIZE;
      uint64_t count;
      struct mtl_sparse_texture_mapping mapping = {
         .level = level,
         .slice = layer,
         .heap_page = heap_page,
      };

      if (level == sparse->tail_level) {
         uint64_t tail_pages = sparse->tail_size_B / KK_SPARSE_PAGE_SIZE;
         count = MIN2(page_count, tail_pages - level_page);
         kk_queue_bind_image_tail(queue, deferred, plane, layer, level_page,
                                  count, mem, heap_page);
      } else {
         struct mtl_size tiles = kk_sparse_level_tiles(plane, level);
         uint64_t x = level_page % tiles.x;
         uint64_t y = (level_page / tiles.x) % tiles.y;
         uint64_t z = level_page / (tiles.x * tiles.y);
         count = MIN2(page_count, tiles.x - x);
         mapping.origin = (struct mtl_size){x, y, z};
         mapping.size = (struct mtl_size){count, 1, 1};
         kk_queue_map_texture(queue, kk_plane_deferred(plane, deferred),
                              plane->mtl_handle, mem, mapping);
         if (plane->mtl_handle_residency)
            kk_queue_map_texture(queue, deferred, plane->mtl_handle_residency,
                                 mem, mapping);
      }

      page += count;
      page_count -= count;
      heap_page += count;
   }
}

static void
kk_queue_bind_image(struct kk_queue *queue, struct util_dynarray *deferred,
                    struct kk_image *image,
                    const VkSparseImageMemoryBind *bind)
{
   VK_FROM_HANDLE(kk_device_memory, mem, bind->memory);
   struct kk_image_plane *plane = &image->planes[0];
   const struct mtl_size tile = plane->sparse.tile_px;

   struct mtl_sparse_texture_mapping mapping = {
      .origin = {bind->offset.x / tile.x, bind->offset.y / tile.y,
                 bind->offset.z / tile.z},
      .size = {DIV_ROUND_UP(bind->extent.width, tile.x),
               DIV_ROUND_UP(bind->extent.height, tile.y),
               DIV_ROUND_UP(bind->extent.depth, tile.z)},
      .level = bind->subresource.mipLevel,
      .slice = bind->subresource.arrayLayer,
      .heap_page = bind->memoryOffset / KK_SPARSE_PAGE_SIZE,
   };
   kk_queue_map_texture(queue, kk_plane_deferred(plane, deferred),
                        plane->mtl_handle, mem, mapping);
   if (plane->mtl_handle_residency)
      kk_queue_map_texture(queue, deferred, plane->mtl_handle_residency, mem,
                           mapping);
}

static void
kk_queue_bind_sparse(struct kk_queue *queue, struct vk_queue_submit *submit,
                     struct util_dynarray *fills)
{
   struct util_dynarray deferred = UTIL_DYNARRAY_INIT;

   for (uint32_t i = 0; i < submit->image_opaque_bind_count; i++) {
      const VkSparseImageOpaqueMemoryBindInfo *info =
         &submit->image_opaque_binds[i];
      VK_FROM_HANDLE(kk_image, image, info->image);

      for (uint32_t j = 0; j < info->bindCount; j++)
         kk_queue_bind_image_opaque(queue, &deferred, image, &info->pBinds[j]);
   }

   for (uint32_t i = 0; i < submit->image_bind_count; i++) {
      const VkSparseImageMemoryBindInfo *info = &submit->image_binds[i];
      VK_FROM_HANDLE(kk_image, image, info->image);

      for (uint32_t j = 0; j < info->bindCount; j++)
         kk_queue_bind_image(queue, &deferred, image, &info->pBinds[j]);
   }

   util_dynarray_foreach(&deferred, struct kk_texture_mapping, op) {
      mtl_command_queue_update_texture_mappings(queue->mtl_handle, op->texture,
                                                op->heap, &op->mapping, 1u);
   }
   util_dynarray_fini(&deferred);

   for (uint32_t i = 0; i < submit->buffer_bind_count; i++) {
      const VkSparseBufferMemoryBindInfo *info = &submit->buffer_binds[i];
      VK_FROM_HANDLE(kk_buffer, buffer, info->buffer);

      for (uint32_t j = 0; j < info->bindCount; j++)
         kk_queue_bind_buffer(queue, buffer, &info->pBinds[j], fills);
   }
}

static VkResult kk_queue_submit_locked(struct kk_queue *queue,
                                       struct vk_queue_submit *submit);

static VkResult
kk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct kk_queue *queue = container_of(vk_queue, struct kk_queue, vk);
   struct kk_device *dev = kk_queue_device(queue);

   if (vk_queue_is_lost(&queue->vk))
      return VK_ERROR_DEVICE_LOST;

   simple_mtx_lock(&dev->atomic64.lock);
   VkResult result = kk_queue_submit_locked(queue, submit);
   simple_mtx_unlock(&dev->atomic64.lock);
   return result;
}

static VkResult
kk_queue_submit_locked(struct kk_queue *queue, struct vk_queue_submit *submit)
{
   struct kk_device *dev = kk_queue_device(queue);

   for (struct vk_sync_wait *wait = submit->waits,
                            *end = submit->waits + submit->wait_count;
        wait != end; ++wait) {
      struct kk_sync_timeline *sync =
         container_of(wait->sync, struct kk_sync_timeline, base);
      mtl_wait_for_event(queue->mtl_handle, sync->mtl_handle, wait->wait_value);
   }

   /* Ensure any changes to residency are propagated before we submit any
    * work. All resources should have been allocated before submission.
    * Otherwise, users are playing with fire. */
   kk_device_make_resources_resident(dev);

   if (vk_queue_submit_has_bind(submit)) {
      struct util_dynarray fills = UTIL_DYNARRAY_INIT;
      kk_queue_bind_sparse(queue, submit, &fills);
      mtl_command_queue_commit_mapping_barrier(
         queue->mtl_handle, dev->mtl_handle,
         dev->sparse_residency.bo ? dev->sparse_residency.bo->map : NULL,
         util_dynarray_begin(&fills),
         util_dynarray_num_elements(&fills, struct mtl_buffer_fill));
      util_dynarray_fini(&fills);
   }

   for (uint32_t i = 0; i < submit->command_buffer_count; ++i) {
      struct kk_cmd_buffer *cmd_buffer =
         container_of(submit->command_buffers[i], struct kk_cmd_buffer, vk);

      if (cmd_buffer->drawable) {
         mtl_command_queue_wait_for_drawable(queue->mtl_handle,
                                             cmd_buffer->drawable);
      }

      /* Metal's command buffers are one time use, re-record multiple
       * submissions. */
      if (cmd_buffer->submitted) {
         VkResult result = rerecord_and_commit_cmd_buffer(queue, cmd_buffer);
         if (result != VK_SUCCESS)
            return result;
      } else if (kk_cmd_buffer_has_work(cmd_buffer)) {
         mtx_lock(&queue->mutex);
         queue->work_in_flight++;
         mtx_unlock(&queue->mutex);
         kk_queue_commit(queue, cmd_buffer, commit_callback, dev);
      }

      cmd_buffer->submitted = true;

      if (cmd_buffer->drawable) {
         mtl_command_queue_signal_drawable(queue->mtl_handle,
                                           cmd_buffer->drawable);
         mtl_release(cmd_buffer->drawable);
         cmd_buffer->drawable = NULL;
      }
   }

   for (uint32_t i = 0u; i < submit->signal_count; ++i) {
      struct vk_sync_signal *signal = &submit->signals[i];
      struct kk_sync_timeline *sync =
         container_of(signal->sync, struct kk_sync_timeline, base);
      mtl_signal_event(queue->mtl_handle, sync->mtl_handle,
                       signal->signal_value);
   }

   return VK_SUCCESS;
}

VkResult
kk_queue_init(struct kk_device *dev, struct kk_queue *queue,
              const VkDeviceQueueCreateInfo *pCreateInfo,
              uint32_t index_in_family)
{
   const VkDeviceQueueGlobalPriorityCreateInfo *priority_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO);
   const VkQueueGlobalPriority global_priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM;

   /* From the Vulkan 1.3.295 spec:
    *
    *    "If the globalPriorityQuery feature is enabled and the requested
    *    global priority is not reported via
    *    VkQueueFamilyGlobalPriorityPropertiesKHR, the driver implementation
    *    must fail the queue creation. In this scenario,
    *    VK_ERROR_INITIALIZATION_FAILED is returned."
    */
   if (dev->vk.enabled_features.globalPriorityQuery &&
       global_priority != VK_QUEUE_GLOBAL_PRIORITY_MEDIUM)
      return VK_ERROR_INITIALIZATION_FAILED;

   if (global_priority > VK_QUEUE_GLOBAL_PRIORITY_MEDIUM)
      return VK_ERROR_NOT_PERMITTED;

   VkResult result;

   result = vk_queue_init(&queue->vk, &dev->vk, pCreateInfo, index_in_family);
   if (result != VK_SUCCESS)
      return result;

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = pCreateInfo->queueFamilyIndex,
   };
   VkCommandPool pool_handle;
   result = kk_CreateCommandPool(kk_device_to_handle(dev), &pool_info, NULL,
                                 &pool_handle);
   if (result != VK_SUCCESS) {
      vk_queue_finish(&queue->vk);
      return result;
   }
   queue->cmd_pool = kk_cmd_pool_from_handle(pool_handle);

   queue->free_cmd_buffers = UTIL_DYNARRAY_INIT;
   queue->commits_in_flight = 0u;
   queue->work_in_flight = 0u;
   mtx_init(&queue->mutex, mtx_plain);
   cnd_init(&queue->cond);

   queue->mtl_handle = mtl_new_command_queue(dev->mtl_handle);
   mtl_command_queue_add_residency_set(queue->mtl_handle,
                                       dev->residency_set.handle);

   queue->vk.driver_submit = kk_queue_submit;

   return VK_SUCCESS;
}

void
kk_queue_finish(struct kk_device *dev, struct kk_queue *queue)
{
   /* There may be in-flight command buffers, wait for them to finish. */
   kk_queue_wait_for_idle(queue);

   mtl_command_queue_remove_residency_set(queue->mtl_handle,
                                          dev->residency_set.handle);
   mtl_release(queue->mtl_handle);

   kk_DestroyCommandPool(kk_device_to_handle(dev),
                         kk_cmd_pool_to_handle(queue->cmd_pool), NULL);
   util_dynarray_fini(&queue->free_cmd_buffers);
   cnd_destroy(&queue->cond);
   mtx_destroy(&queue->mutex);

   vk_queue_finish(&queue->vk);
}
