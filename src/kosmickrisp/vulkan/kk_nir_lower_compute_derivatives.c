/*
 * Copyright 2026 David Andrews
 * SPDX-License-Identifier: MIT
 */

#include "kk_shader.h"

#include "nir_builder.h"
#include "nir_builtin_builder.h"

static nir_def *
quad_derivative(nir_builder *b, nir_def *value, bool vertical, bool fine)
{
   nir_def *lanes[4];
   for (unsigned i = 0; i < 4; i++)
      lanes[i] = nir_quad_broadcast(b, value, nir_imm_int(b, i));

   const unsigned step = vertical ? 2 : 1;
   nir_def *first = nir_fsub(b, lanes[step], lanes[0]);
   if (!fine)
      return first;

   nir_def *second = nir_fsub(b, lanes[3], lanes[3 - step]);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *in_second = nir_test_mask(b, lane, vertical ? 1 : 2);
   return nir_bcsel(b, in_second, second, first);
}

static bool
lower_derivative(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   bool vertical, fine;
   switch (intrin->intrinsic) {
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddx_coarse:
      vertical = false;
      fine = false;
      break;
   case nir_intrinsic_ddx_fine:
      vertical = false;
      fine = true;
      break;
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddy_coarse:
      vertical = true;
      fine = false;
      break;
   case nir_intrinsic_ddy_fine:
      vertical = true;
      fine = true;
      break;
   default:
      return false;
   }

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def_replace(&intrin->def,
                   quad_derivative(b, intrin->src[0].ssa, vertical, fine));
   return true;
}

static nir_def *
half_bits_to_float(nir_builder *b, nir_def *bits)
{
   return nir_f2f32(b, nir_u2u16(b, bits));
}

static bool
lower_tex(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *coord = nir_get_tex_src(tex, nir_tex_src_coord);
   if (tex->is_array)
      coord = nir_trim_vector(b, coord, coord->num_components - 1);

   nir_def *ddx = quad_derivative(b, coord, false, false);
   nir_def *ddy = quad_derivative(b, coord, true, false);

   if (tex->op == nir_texop_txb) {
      nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
      nir_def *scale = nir_fexp2(b, nir_f2f32(b, bias));
      ddx = nir_fmul(b, ddx, scale);
      ddy = nir_fmul(b, ddy, scale);
   }

   nir_tex_instr_add_src(tex, nir_tex_src_ddx, ddx);
   nir_tex_instr_add_src(tex, nir_tex_src_ddy, ddy);
   tex->op = nir_texop_txd;
   return true;
}

static bool
lower_lod(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (tex->op != nir_texop_lod)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *coord = nir_get_tex_src(tex, nir_tex_src_coord);
   nir_tex_instr_add_src(tex, nir_tex_src_ddx,
                         quad_derivative(b, coord, false, false));
   nir_tex_instr_add_src(tex, nir_tex_src_ddy,
                         quad_derivative(b, coord, true, false));

   nir_def *lod = nir_tex_gradient_lod(b, tex);
   nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
   if (bias)
      lod = nir_fadd(b, lod, nir_f2f32(b, bias));

   nir_def *limits = nir_build_texture_query(
      b, tex, nir_texop_sampler_minmax_kk, 3, nir_type_uint32, false, false);
   nir_def *levels = nir_build_texture_query(b, tex, nir_texop_query_levels, 1,
                                             nir_type_int32, false, false);

   nir_def *level =
      nir_fclamp(b, lod, half_bits_to_float(b, nir_channel(b, limits, 1)),
                 half_bits_to_float(b, nir_channel(b, limits, 2)));
   level = nir_fclamp(b, level, nir_imm_float(b, 0.0f),
                      nir_i2f32(b, nir_iadd_imm(b, levels, -1)));

   nir_def *mip_linear = nir_test_mask(b, nir_channel(b, limits, 0),
                                       KK_SAMPLER_MINMAX_MIP_LINEAR);
   nir_def *nearest = nir_fadd_imm(
      b, nir_fceil(b, nir_fadd_imm(b, level, 0.5)), -1.0);
   level = nir_bcsel(b, mip_linear, level, nearest);

   nir_def_replace(&tex->def, nir_vec2(b, level, lod));
   return true;
}

static bool
has_compute_derivatives(const nir_shader *nir)
{
   return nir->info.stage == MESA_SHADER_COMPUTE &&
          nir->info.derivative_group != DERIVATIVE_GROUP_NONE;
}

bool
kk_nir_lower_compute_derivatives(nir_shader *nir)
{
   if (!has_compute_derivatives(nir))
      return false;

   bool progress = false;
   NIR_PASS(progress, nir, nir_shader_tex_pass, lower_tex,
            nir_metadata_control_flow, NULL);
   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, lower_derivative,
            nir_metadata_control_flow, NULL);
   return progress;
}

bool
kk_nir_lower_compute_lod_queries(nir_shader *nir)
{
   if (!has_compute_derivatives(nir))
      return false;

   return nir_shader_tex_pass(nir, lower_lod, nir_metadata_control_flow, NULL);
}
