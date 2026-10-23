/*
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "kk_cmd_buffer.h"
#include "kk_shader.h"

#include "nir.h"
#include "nir_builder.h"

/* Line stipple for Bresenham lines (stippledBresenhamLines)
 *
 * Metal has no line stipple, so the fragment shader discards the fragments
 * whose pattern bit is clear. The stage before the rasterizer writes the
 * window position of each vertex to two outputs it adds: a flat one, which
 * the fragment shader reads from the first vertex of the line segment (Metal
 * takes flat outputs from the first vertex of a primitive, and KosmicKrisp's
 * provoking vertex emulation leaves this output out), and a noperspective
 * one, whose interpolated value lies on the segment and so gives its
 * direction. For a Bresenham line the fragments step one pixel along the major
 * axis, so the stipple counter of a fragment is the number of pixel centers
 * between it and the segment's first fragment on that axis: the first
 * fragment's center is within half a pixel of the start.
 *
 * The counter restarts with every segment. That is the rule for line lists,
 * but the Vulkan spec continues it through a line strip, which would need the
 * lengths of all earlier segments of the strip, which no shader stage here
 * has. The window positions also assume the start vertex is in front of the
 * eye (w > 0). Draws whose line vertices the unroll kernels reverse for a
 * last provoking vertex and flat fragment shader inputs, lines with
 * adjacency without a GS and any kk_unroll_geometry draw of a vertex shader
 * without provoking_ids, start the pattern at the end of the segment;
 * geometry and tessellation pipelines are not unrolled.
 *
 * Runs after the stages are linked, on IO lowered to intrinsics, with output
 * slots the linked stages leave free.
 */

/* Loads from the struct kk_line_stipple_data the root table points to */
static nir_def *
load_stipple_data(nir_builder *b, unsigned num_components, unsigned offset_B,
                  nir_def *index_B)
{
   nir_def *root = nir_load_buffer_ptr_kk(b, 1, 64, .binding = 0);
   nir_def *data = nir_build_load_global_constant(
      b, 1, 64,
      nir_iadd_imm(b, root,
                   offsetof(struct kk_root_descriptor_table, draw.line_stipple)),
      .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
   nir_def *offset = nir_imm_int(b, offset_B);
   if (index_B)
      offset = nir_iadd(b, offset, index_B);
   return nir_build_load_global_constant(b, num_components, 32,
                                         nir_iadd(b, data, nir_u2u64(b, offset)),
                                         .align_mul = 4,
                                         .access = ACCESS_CAN_SPECULATE);
}

struct output_state {
   nir_variable *pos, *viewport;
   unsigned start_slot, pos_slot;
};

static void
store_slot(nir_builder *b, nir_def *value, unsigned slot)
{
   nir_store_output(b, value, nir_imm_int(b, 0), .base = 0, .range = 1,
                    .write_mask = 0x3, .component = 0,
                    .src_type = nir_type_float32,
                    .io_semantics.location = slot,
                    .io_semantics.num_slots = 1);
}

/* The vertex's window position, from the position and viewport index last
 * written */
static void
store_window_position(nir_builder *b, const struct output_state *state)
{
   nir_def *pos = nir_load_var(b, state->pos);
   nir_def *viewport =
      nir_umin(b, nir_load_var(b, state->viewport),
               nir_imm_int(b, KK_MAX_VIEWPORTS - 1));
   /* w = 0 would give infinities that turn into NaN in the fragment shader */
   nir_def *w = nir_channel(b, pos, 3);
   w = nir_bcsel(b, nir_feq_imm(b, w, 0.0f), nir_imm_float(b, 1e-20f), w);
   nir_def *ndc = nir_fdiv(b, nir_trim_vector(b, pos, 2), w);
   nir_def *xform =
      load_stipple_data(b, 4,
                        offsetof(struct kk_line_stipple_data, viewport_xform),
                        nir_imul_imm(b, viewport, 4 * sizeof(float)));
   nir_def *window = nir_ffma(b, ndc, nir_channels(b, xform, 0x3),
                              nir_channels(b, xform, 0xc));

   store_slot(b, window, state->start_slot);
   store_slot(b, window, state->pos_slot);
}

static bool
track_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct output_state *state = data;

   if (intr->intrinsic == nir_intrinsic_emit_vertex) {
      if (nir_intrinsic_stream_id(intr) == 0) {
         b->cursor = nir_before_instr(&intr->instr);
         store_window_position(b, state);
      }
      return true;
   }

   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_variable *var;
   if (sem.location == VARYING_SLOT_POS)
      var = state->pos;
   else if (sem.location == VARYING_SLOT_VIEWPORT)
      var = state->viewport;
   else
      return false;

   b->cursor = nir_after_instr(&intr->instr);
   unsigned component = nir_intrinsic_component(intr);
   unsigned mask = nir_intrinsic_write_mask(intr) << component;
   nir_def *value = intr->src[0].ssa;
   if (var == state->viewport) {
      nir_store_var(b, var, nir_channel(b, value, 0), 0x1);
   } else {
      nir_def *channels[4];
      for (unsigned c = 0; c < 4; c++) {
         channels[c] = c >= component && c - component < value->num_components
                          ? nir_channel(b, value, c - component)
                          : nir_undef(b, 1, 32);
      }
      nir_store_var(b, var, nir_vec(b, channels, 4), mask & 0xf);
   }
   return true;
}

/* The stage before the rasterizer: the vertex shader, the tessellation
 * evaluation shader or the geometry shader */
void
kk_nir_lower_line_stipple_output(nir_shader *nir, unsigned start_slot,
                                 unsigned pos_slot)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   struct output_state state = {
      .pos = nir_local_variable_create(impl, glsl_vec4_type(), "stipple_pos"),
      .viewport =
         nir_local_variable_create(impl, glsl_uint_type(), "stipple_viewport"),
      .start_slot = start_slot,
      .pos_slot = pos_slot,
   };
   nir_store_var(&b, state.pos, nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 1.0f), 0xf);
   nir_store_var(&b, state.viewport, nir_imm_int(&b, 0), 0x1);

   nir_shader_intrinsics_pass(nir, track_output, nir_metadata_control_flow,
                              &state);

   if (nir->info.stage != MESA_SHADER_GEOMETRY) {
      b.cursor = nir_after_impl(impl);
      store_window_position(&b, &state);
   }

   nir->info.outputs_written |=
      BITFIELD64_BIT(start_slot) | BITFIELD64_BIT(pos_slot);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_recompute_io_bases, nir_var_shader_out);
}

void
kk_nir_lower_line_stipple_fs(nir_shader *nir, unsigned start_slot,
                             unsigned pos_slot, bool guard_helper_writes)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_def *start =
      nir_load_input(&b, 2, 32, nir_imm_int(&b, 0), .base = 0, .component = 0,
                     .dest_type = nir_type_float32,
                     .io_semantics.location = start_slot,
                     .io_semantics.num_slots = 1);
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_NOPERSPECTIVE);
   nir_def *on_line = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .dest_type = nir_type_float32, .io_semantics.location = pos_slot,
      .io_semantics.num_slots = 1);

   /* The pixel center, also when the shader runs per sample */
   nir_def *center = nir_fadd_imm(
      &b, nir_ffloor(&b, nir_trim_vector(&b, nir_load_frag_coord(&b), 2)),
      0.5f);

   nir_def *dir = nir_fabs(&b, nir_fsub(&b, on_line, start));
   nir_def *x_major =
      nir_fge(&b, nir_channel(&b, dir, 0), nir_channel(&b, dir, 1));
   nir_def *dist = nir_fabs(&b, nir_fsub(&b, center, start));
   dist = nir_bcsel(&b, x_major, nir_channel(&b, dist, 0),
                    nir_channel(&b, dist, 1));

   /* The first fragment's center is within half a pixel of the start */
   nir_def *counter = nir_f2u32(
      &b, nir_fmin(&b,
                   nir_fmax(&b, nir_fceil(&b, nir_fadd_imm(&b, dist, -0.5f)),
                            nir_imm_float(&b, 0.0f)),
                   nir_imm_float(&b, 16777216.0f)));

   nir_def *stipple = load_stipple_data(
      &b, 1, offsetof(struct kk_line_stipple_data, stipple), NULL);
   nir_def *factor = nir_umax(&b, nir_ushr_imm(&b, stipple, 16),
                              nir_imm_int(&b, 1));
   nir_def *bit = nir_iand_imm(&b, nir_udiv(&b, counter, factor), 15);
   nir_def *keep = nir_iand_imm(&b, nir_ushr(&b, stipple, bit), 1);

   /* Demoted rather than terminated, so the rest of the quad keeps its
    * derivatives */
   nir_demote_if(&b, nir_ieq_imm(&b, keep, 0));

   nir->info.inputs_read |=
      BITFIELD64_BIT(start_slot) | BITFIELD64_BIT(pos_slot);
   BITSET_SET(nir->info.system_values_read, SYSTEM_VALUE_FRAG_COORD);
   nir->info.fs.uses_discard = true;
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_recompute_io_bases, nir_var_shader_in);

   /* The stores of a demoted fragment must not happen. The guards
    * kk_lower_fs added to them track the demotes the shader had then, so
    * guard them again, which also sees this one. */
   if (guard_helper_writes && nir->info.writes_memory) {
      NIR_PASS(_, nir, nir_lower_helper_writes, true);
      NIR_PASS(_, nir, nir_lower_is_helper_invocation);
   }
}
