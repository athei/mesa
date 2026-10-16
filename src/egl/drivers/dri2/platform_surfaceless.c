/*
 * Mesa 3-D graphics library
 *
 * Copyright (c) 2014 The Chromium OS Authors.
 * Copyright © 2011 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "pipe/p_screen.h"
#include "util/libdrm.h"
#include <sys/stat.h>
#include <sys/types.h>

#include "egl_dri2.h"
#include "eglglobals.h"
#include "kopper_interface.h"

#ifdef VK_USE_PLATFORM_METAL_EXT
#include <CoreGraphics/CGGeometry.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <vulkan/vulkan_metal.h>
#endif
#include "loader.h"
#include "loader_dri_helper.h"
#include "dri_util.h"
#include "dri_screen.h"

static struct dri_image *
surfaceless_alloc_image(struct dri2_egl_display *dri2_dpy,
                        struct dri2_egl_surface *dri2_surf)
{
   return dri_create_image(
      dri2_dpy->dri_screen_render_gpu, dri2_surf->base.Width,
      dri2_surf->base.Height, dri2_surf->visual, NULL, 0, 0, NULL);
}

static void
surfaceless_free_images(struct dri2_egl_surface *dri2_surf)
{
   if (dri2_surf->front) {
      dri2_destroy_image(dri2_surf->front);
      dri2_surf->front = NULL;
   }

   free(dri2_surf->swrast_device_buffer);
   dri2_surf->swrast_device_buffer = NULL;
}

static int
surfaceless_image_get_buffers(struct dri_drawable *driDrawable, unsigned int format,
                              uint32_t *stamp, void *loaderPrivate,
                              uint32_t buffer_mask,
                              struct __DRIimageList *buffers)
{
   struct dri2_egl_surface *dri2_surf = loaderPrivate;
   struct dri2_egl_display *dri2_dpy =
      dri2_egl_display(dri2_surf->base.Resource.Display);

   buffers->image_mask = 0;
   buffers->front = NULL;
   buffers->back = NULL;

   /* The EGL 1.5 spec states that pbuffers are single-buffered. Specifically,
    * the spec states that they have a back buffer but no front buffer, in
    * contrast to pixmaps, which have a front buffer but no back buffer.
    *
    * Single-buffered surfaces with no front buffer confuse Mesa; so we deviate
    * from the spec, following the precedent of Mesa's EGL X11 platform. The
    * X11 platform correctly assigns pbuffers to single-buffered configs, but
    * assigns the pbuffer a front buffer instead of a back buffer.
    *
    * Pbuffers in the X11 platform mostly work today, so let's just copy its
    * behavior instead of trying to fix (and hence potentially breaking) the
    * world.
    */

   if (buffer_mask & __DRI_IMAGE_BUFFER_FRONT) {

      if (!dri2_surf->front) {
         dri2_surf->front = surfaceless_alloc_image(dri2_dpy, dri2_surf);
         if (!dri2_surf->front)
            return 0;
      }

      buffers->image_mask |= __DRI_IMAGE_BUFFER_FRONT;
      buffers->front = dri2_surf->front;
   }

   return 1;
}

#ifdef VK_USE_PLATFORM_METAL_EXT
static void
surfaceless_metal_layer_size(void *layer, EGLint *width, EGLint *height)
{
   CGRect bounds =
      ((CGRect(*)(id, SEL))objc_msgSend)((id)layer, sel_registerName("bounds"));
   CGFloat scale = ((CGFloat(*)(id, SEL))objc_msgSend)(
      (id)layer, sel_registerName("contentsScale"));

   *width = MAX2(bounds.size.width * scale, 1);
   *height = MAX2(bounds.size.height * scale, 1);
}

/* A window surface holds a reference to its layer, which the application may
 * release once eglDestroySurface returns, while the surface itself lives on
 * until no context has it current. */
static void
surfaceless_metal_layer_retain(void *layer)
{
   ((id(*)(id, SEL))objc_msgSend)((id)layer, sel_registerName("retain"));
}

static void
surfaceless_metal_layer_release(void *layer)
{
   ((void(*)(id, SEL))objc_msgSend)((id)layer, sel_registerName("release"));
}
#endif

static _EGLSurface *
dri2_surfaceless_create_surface(_EGLDisplay *disp, EGLint type,
                                _EGLConfig *conf, void *native_window,
                                const EGLint *attrib_list)
{
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   struct dri2_egl_config *dri2_conf = dri2_egl_config(conf);
   struct dri2_egl_surface *dri2_surf;
   const struct dri_config *config;

   /* Make sure to calloc so all pointers
    * are originally NULL.
    */
   dri2_surf = calloc(1, sizeof *dri2_surf);

   if (!dri2_surf) {
      _eglError(EGL_BAD_ALLOC, "eglCreatePbufferSurface");
      return NULL;
   }

   if (!dri2_init_surface(&dri2_surf->base, disp, type, conf, attrib_list,
                          false, native_window))
      goto cleanup_surface;

#ifdef VK_USE_PLATFORM_METAL_EXT
   if (type == EGL_WINDOW_BIT) {
      surfaceless_metal_layer_retain(native_window);
      surfaceless_metal_layer_size(native_window, &dri2_surf->base.Width,
                                   &dri2_surf->base.Height);
   }
#endif

   config = dri2_get_dri_config(dri2_conf, type, dri2_surf->base.GLColorspace);

   if (!config) {
      _eglError(EGL_BAD_MATCH,
                "Unsupported surfacetype/colorspace configuration");
      goto cleanup_layer;
   }

   dri2_surf->visual = dri2_image_format_for_pbuffer_config(dri2_dpy, config);
   if (dri2_surf->visual == PIPE_FORMAT_NONE)
      goto cleanup_layer;

   if (!dri2_create_drawable(dri2_dpy, config, dri2_surf, dri2_surf))
      goto cleanup_layer;

   return &dri2_surf->base;

cleanup_layer:
#ifdef VK_USE_PLATFORM_METAL_EXT
   if (type == EGL_WINDOW_BIT)
      surfaceless_metal_layer_release(native_window);
#endif
cleanup_surface:
   free(dri2_surf);
   return NULL;
}

static EGLBoolean
surfaceless_destroy_surface(_EGLDisplay *disp, _EGLSurface *surf)
{
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(surf);

   surfaceless_free_images(dri2_surf);

   driDestroyDrawable(dri2_surf->dri_drawable);

#ifdef VK_USE_PLATFORM_METAL_EXT
   if (surf->Type == EGL_WINDOW_BIT)
      surfaceless_metal_layer_release(surf->NativeSurface);
#endif

   dri2_fini_surface(surf);
   free(dri2_surf);
   return EGL_TRUE;
}

static _EGLSurface *
dri2_surfaceless_create_pbuffer_surface(_EGLDisplay *disp, _EGLConfig *conf,
                                        const EGLint *attrib_list)
{
   return dri2_surfaceless_create_surface(disp, EGL_PBUFFER_BIT, conf, NULL,
                                          attrib_list);
}

static const struct dri2_egl_display_vtbl dri2_surfaceless_display_vtbl = {
   .create_pbuffer_surface = dri2_surfaceless_create_pbuffer_surface,
   .destroy_surface = surfaceless_destroy_surface,
   .create_image = dri2_create_image_khr,
};

#ifdef VK_USE_PLATFORM_METAL_EXT
static EGLBoolean
surfaceless_kopper_swap_interval(_EGLDisplay *disp, _EGLSurface *surf,
                                 EGLint interval)
{
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(surf);

   if (surf->Type == EGL_WINDOW_BIT)
      kopperSetSwapInterval(dri2_surf->dri_drawable, interval);
   return EGL_TRUE;
}

static _EGLSurface *
dri2_surfaceless_create_window_surface(_EGLDisplay *disp, _EGLConfig *conf,
                                       void *native_window,
                                       const EGLint *attrib_list)
{
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   _EGLSurface *surf;

   if (!native_window) {
      _eglError(EGL_BAD_NATIVE_WINDOW, "eglCreateWindowSurface");
      return NULL;
   }

   surf = dri2_surfaceless_create_surface(disp, EGL_WINDOW_BIT, conf,
                                          native_window, attrib_list);
   if (surf != NULL) {
      surf->SwapInterval = dri2_dpy->default_swap_interval;
      surfaceless_kopper_swap_interval(disp, surf,
                                       dri2_dpy->default_swap_interval);
   }
   return surf;
}

static EGLBoolean
surfaceless_kopper_swap_buffers(_EGLDisplay *disp, _EGLSurface *draw,
                                const EGLint *rects, EGLint n_rects)
{
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(draw);

   if (draw->Type != EGL_WINDOW_BIT)
      return EGL_TRUE;

   kopperSwapBuffers(dri2_surf->dri_drawable,
                     __DRI2_FLUSH_CONTEXT | __DRI2_FLUSH_INVALIDATE_ANCILLARY,
                     n_rects, rects);
   kopperQuerySurfaceSize(dri2_surf->dri_drawable, &dri2_surf->base.Width,
                          &dri2_surf->base.Height);
   return EGL_TRUE;
}

static EGLint
surfaceless_kopper_query_buffer_age(_EGLDisplay *disp, _EGLSurface *surf)
{
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(surf);

   return surf->Type == EGL_WINDOW_BIT
             ? kopperQueryBufferAge(dri2_surf->dri_drawable)
             : 0;
}

static const struct dri2_egl_display_vtbl dri2_surfaceless_metal_display_vtbl = {
   .create_window_surface = dri2_surfaceless_create_window_surface,
   .create_pbuffer_surface = dri2_surfaceless_create_pbuffer_surface,
   .destroy_surface = surfaceless_destroy_surface,
   .create_image = dri2_create_image_khr,
   .swap_interval = surfaceless_kopper_swap_interval,
   .swap_buffers = surfaceless_kopper_swap_buffers,
   .query_buffer_age = surfaceless_kopper_query_buffer_age,
};

static_assert(sizeof(struct kopper_vk_surface_create_storage) >=
                 sizeof(VkMetalSurfaceCreateInfoEXT),
              "");

static void
surfaceless_kopper_set_surface_create_info(void *_draw,
                                           struct kopper_loader_info *ci)
{
   struct dri2_egl_surface *dri2_surf = _draw;
   VkMetalSurfaceCreateInfoEXT *metal = (VkMetalSurfaceCreateInfoEXT *)&ci->bos;

   if (dri2_surf->base.Type != EGL_WINDOW_BIT)
      return;

   metal->sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
   metal->pNext = NULL;
   metal->flags = 0;
   metal->pLayer = dri2_surf->base.NativeSurface;
   ci->has_alpha = dri2_surf->base.Config->AlphaSize > 0;
   ci->present_opaque = dri2_surf->base.PresentOpaque;
}

static void
surfaceless_kopper_get_drawable_info(struct dri_drawable *draw, int *w, int *h,
                                     void *loaderPrivate)
{
   struct dri2_egl_surface *dri2_surf = loaderPrivate;

   if (dri2_surf->base.Type == EGL_WINDOW_BIT)
      surfaceless_metal_layer_size(dri2_surf->base.NativeSurface,
                                   &dri2_surf->base.Width,
                                   &dri2_surf->base.Height);
   *w = dri2_surf->base.Width;
   *h = dri2_surf->base.Height;
}

static const __DRIkopperLoaderExtension kopper_metal_loader_extension = {
   .SetSurfaceCreateInfo = surfaceless_kopper_set_surface_create_info,
   .GetDrawableInfo = surfaceless_kopper_get_drawable_info,
};
#endif

static void
surfaceless_flush_front_buffer(struct dri_drawable *driDrawable, void *loaderPrivate)
{
}

static unsigned
surfaceless_get_capability(void *loaderPrivate, enum dri_loader_cap cap)
{
   /* Note: loaderPrivate is _EGLDisplay* */
   switch (cap) {
   case DRI_LOADER_CAP_FP16:
      return 1;
   case DRI_LOADER_CAP_RGBA_ORDERING:
      return 1;
   default:
      return 0;
   }
}

static const __DRIimageLoaderExtension image_loader_extension = {
   .getBuffers = surfaceless_image_get_buffers,
   .flushFrontBuffer = surfaceless_flush_front_buffer,
   .getCapability = surfaceless_get_capability,
};

static const struct dri_loader_funcs image_loader_funcs = {
   .image_lookup = &image_lookup_extension,
   .image = &image_loader_extension,
};

static const struct dri_loader_funcs swrast_loader_funcs = {
   .image_lookup = &image_lookup_extension,
   .image = &image_loader_extension,
   .swrast = &swrast_pbuffer_loader_extension,
};

static const struct dri_loader_funcs kopper_loader_funcs = {
   .image_lookup = &image_lookup_extension,
#ifdef VK_USE_PLATFORM_METAL_EXT
   .kopper = &kopper_metal_loader_extension,
#else
   .kopper = &kopper_pbuffer_loader_extension,
#endif
};

static bool
surfaceless_probe_device(_EGLDisplay *disp, bool swrast, bool zink)
{
   const unsigned node_type = swrast ? DRM_NODE_PRIMARY : DRM_NODE_RENDER;
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   _EGLDevice *dev_list = _eglGlobal.DeviceList;
   drmDevicePtr device;

   while (dev_list) {
      if (!_eglDeviceSupports(dev_list, _EGL_DEVICE_DRM))
         goto next;

      if (_eglHasAttrib(disp, EGL_DEVICE_EXT) && dev_list != disp->Device) {
         goto next;
      }

      device = _eglDeviceDrm(dev_list);
      assert(device);

      if (!(device->available_nodes & (1 << node_type)))
         goto next;

      dri2_dpy->fd_render_gpu = loader_open_device(device->nodes[node_type]);
      if (dri2_dpy->fd_render_gpu < 0)
         goto next;

#ifdef HAVE_WAYLAND_PLATFORM
      loader_get_user_preferred_fd(&dri2_dpy->fd_render_gpu,
                                   &dri2_dpy->fd_display_gpu);

      if (dri2_dpy->fd_render_gpu != dri2_dpy->fd_display_gpu) {
         free(dri2_dpy->device_name);
         dri2_dpy->device_name =
            loader_get_device_name_for_fd(dri2_dpy->fd_render_gpu);
         if (!dri2_dpy->device_name) {
            _eglError(EGL_BAD_ALLOC, "surfaceless-egl: failed to get device name "
                                     "for requested GPU");
            goto retry;
         }
      }

      /* we have to do the check now, because loader_get_user_preferred_fd
       * will return a render-node when the requested gpu is different
       * to the server, but also if the client asks for the same gpu than
       * the server by requesting its pci-id */
      dri2_dpy->is_render_node =
         drmGetNodeTypeFromFd(dri2_dpy->fd_render_gpu) == DRM_NODE_RENDER;
#endif
      char *driver_name = loader_get_driver_for_fd(dri2_dpy->fd_render_gpu);

      disp->Device = dev_list;
      if (swrast) {
         /* Use kms swrast only with vgem / virtio_gpu.
          * virtio-gpu fallbacks to software rendering when 3D features
          * are unavailable since 6c5ab, and kms_swrast is more
          * feature complete than swrast.
          */
         if (driver_name && (strcmp(driver_name, "vgem") == 0 ||
                             strcmp(driver_name, "virtio_gpu") == 0))
            dri2_dpy->driver_name = strdup("kms_swrast");
         free(driver_name);
      } else {
         /* Use the given hardware driver */
         dri2_dpy->driver_name = driver_name;
      }

      if (dri2_dpy->driver_name) {
         dri2_detect_swrast_kopper(disp);
         if (dri2_dpy->kopper)
            dri2_dpy->loader_funcs = &kopper_loader_funcs;
         else if (swrast)
            dri2_dpy->loader_funcs = &swrast_loader_funcs;
         else
            dri2_dpy->loader_funcs = &image_loader_funcs;

         if (!dri2_create_screen(disp)) {
            _eglLog(_EGL_WARNING, "DRI2: failed to create screen");
            goto retry;
         }

         if (!dri2_dpy->dri_screen_render_gpu->base.screen->caps.graphics) {

            _eglLog(_EGL_DEBUG, "DRI2: Driver %s doesn't support graphics, skipping.", dri2_dpy->driver_name);

            if (dri2_dpy->dri_screen_display_gpu != dri2_dpy->dri_screen_render_gpu) {
               driDestroyScreen(dri2_dpy->dri_screen_display_gpu);
               dri2_dpy->dri_screen_display_gpu = NULL;
            }

            driDestroyScreen(dri2_dpy->dri_screen_render_gpu);
            dri2_dpy->dri_screen_render_gpu = NULL;

            dri2_dpy->own_dri_screen = false;

            goto retry;
         }

         break;
      }

   retry:
      free(dri2_dpy->driver_name);
      dri2_dpy->driver_name = NULL;
      if (dri2_dpy->fd_display_gpu != dri2_dpy->fd_render_gpu)
         close(dri2_dpy->fd_display_gpu);
      dri2_dpy->fd_display_gpu = -1;
      close(dri2_dpy->fd_render_gpu);
      dri2_dpy->fd_render_gpu = -1;

   next:
      dev_list = _eglDeviceNext(dev_list);
   }

   if (!dev_list)
      return false;

   return true;
}

static bool
surfaceless_probe_device_sw(_EGLDisplay *disp)
{
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   struct _egl_device *device = _eglFindDevice(dri2_dpy->fd_render_gpu, true);

   dri2_dpy->fd_render_gpu = -1;

   if (_eglHasAttrib(disp, EGL_DEVICE_EXT) && disp->Device != device) {
      return false;
   }

   disp->Device = device;
   assert(disp->Device);

   dri2_dpy->driver_name = strdup(disp->Options.Zink ? "zink" : "swrast");
   if (!dri2_dpy->driver_name)
      return false;

   dri2_detect_swrast_kopper(disp);

   if (dri2_dpy->kopper)
      dri2_dpy->loader_funcs = &kopper_loader_funcs;
   else
      dri2_dpy->loader_funcs = &swrast_loader_funcs;

   dri2_dpy->fd_display_gpu = dri2_dpy->fd_render_gpu;

   if (!dri2_create_screen(disp)) {
      _eglLog(_EGL_WARNING, "DRI2: failed to create screen");
      free(dri2_dpy->driver_name);
      dri2_dpy->driver_name = NULL;
      return false;
   }

   return true;
}

EGLBoolean
dri2_initialize_surfaceless(_EGLDisplay *disp)
{
   const char *err;
   bool driver_loaded = false;
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);

   /* When ForceSoftware is false, we try the HW driver.  When ForceSoftware
    * is true, we try kms_swrast and swrast in order.
    */
   driver_loaded = surfaceless_probe_device(disp, disp->Options.ForceSoftware,
                                            disp->Options.Zink);
   if (!driver_loaded && (disp->Options.ForceSoftware || disp->Options.Zink)) {
      _eglLog(_EGL_DEBUG, "Falling back to surfaceless swrast without DRM.");
      driver_loaded = surfaceless_probe_device_sw(disp);
   }

   if (!driver_loaded) {
      err = "DRI2: failed to load driver";
      goto cleanup;
   }

   dri2_setup_screen(disp);
#ifdef HAVE_WAYLAND_PLATFORM
   if (dri2_dpy->fd_render_gpu >= 0) {
      dri2_dpy->device_name =
         loader_get_device_name_for_fd(dri2_dpy->fd_render_gpu);
   }
#endif
   dri2_set_WL_bind_wayland_display(disp);

#ifdef VK_USE_PLATFORM_METAL_EXT
   if (dri2_dpy->kopper) {
      dri2_setup_swap_interval(disp, 1);
      dri2_add_configs_for_visuals(disp, EGL_WINDOW_BIT | EGL_PBUFFER_BIT);
      disp->Extensions.EXT_present_opaque = EGL_TRUE;
   } else
#endif
      dri2_add_pbuffer_configs_for_visuals(disp);

   /* Fill vtbl last to prevent accidentally calling virtual function during
    * initialization.
    */
   dri2_dpy->vtbl = &dri2_surfaceless_display_vtbl;
#ifdef VK_USE_PLATFORM_METAL_EXT
   if (dri2_dpy->kopper)
      dri2_dpy->vtbl = &dri2_surfaceless_metal_display_vtbl;
#endif

   return EGL_TRUE;

cleanup:
   return _eglError(EGL_NOT_INITIALIZED, err);
}
