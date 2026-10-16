/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2023 Advanced Micro Devices, Inc.
 * Copyright 2018 Intel Corporation
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "kk_private.h"

#include "kk_shader.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_builtin_builder.h"

#include "stdbool.h"

static nir_def *
query_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_custom_border_color_agx, 4,
                                  tex->dest_type, false, false);
}

static nir_def *
has_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_has_custom_border_color_agx,
                                  1, nir_type_bool1, false, false);
}

static nir_def *
compare_depth(nir_builder *b, nir_def *op, nir_def *ref, nir_def *depth)
{
   nir_def *lt = nir_flt(b, ref, depth);
   nir_def *eq = nir_feq(b, ref, depth);
   nir_def *gt = nir_flt(b, depth, ref);
   nir_def *res = nir_imm_false(b);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_LESS), lt, res);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_EQUAL), eq, res);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_LESS_OR_EQUAL),
                   nir_ior(b, lt, eq), res);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_GREATER), gt, res);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_NOT_EQUAL),
                   nir_inot(b, eq), res);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_GREATER_OR_EQUAL),
                   nir_ior(b, gt, eq), res);
   res = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_ALWAYS),
                   nir_imm_true(b), res);
   return res;
}

static bool
uses_are_alu(nir_def *def)
{
   nir_foreach_use_including_if(use, def) {
      if (nir_src_is_if(use) ||
          nir_src_use_instr(use)->type != nir_instr_type_alu)
         return false;
   }
   return true;
}

static void
rewrite_sparse_uses(nir_builder *b, nir_def *def, nir_def *value,
                    nir_def *code)
{
   nir_foreach_use_safe(use, def) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_src_use_instr(use));
      nir_alu_src *src = container_of(use, nir_alu_src, src);
      unsigned comps = nir_ssa_alu_instr_src_components(alu, src - alu->src);

      b->cursor = nir_before_instr(&alu->instr);
      nir_def *chans[NIR_MAX_VEC_COMPONENTS];
      for (unsigned c = 0; c < comps; c++) {
         unsigned swz = src->swizzle[c];
         chans[c] = swz == value->num_components ? code
                                                 : nir_channel(b, value, swz);
      }

      nir_src_rewrite(&src->src, nir_vec(b, chans, comps));
      for (unsigned c = 0; c < NIR_MAX_VEC_COMPONENTS; c++)
         src->swizzle[c] = c < comps ? c : 0;
   }
}

static bool
lower(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (!nir_tex_instr_need_sampler(tex) || nir_tex_instr_is_query(tex))
      return false;

   if (tex->is_sparse && !uses_are_alu(&tex->def))
      return false;

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *has_custom = has_custom_border(b, tex);

   nir_instr *orig = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, orig);
   nir_def *clamp_to_1 = &nir_instr_as_tex(orig)->def;
   nir_def *code = tex->is_sparse ? nir_channel(b, clamp_to_1, 4) : NULL;
   if (tex->is_sparse)
      clamp_to_1 = nir_trim_vector(b, clamp_to_1, 4);

   nir_push_if(b, has_custom);
   nir_def *replaced = NULL;
   {
      /* Grab the border colour */
      nir_def *border = query_custom_border(b, tex);

      if (tex->is_shadow) {
         nir_def *flags = nir_channel(
            b,
            nir_build_texture_query(b, tex, nir_texop_sampler_minmax_kk, 3,
                                    nir_type_uint32, false, false),
            0);
         nir_def *op = nir_iand_imm(
            b, nir_ushr_imm(b, flags, KK_SAMPLER_COMPARE_OP_SHIFT),
            KK_SAMPLER_COMPARE_OP_MASK);
         nir_def *image_flags = nir_build_texture_query(
            b, tex, nir_texop_image_flags_kk, 1, nir_type_uint32, false, false);
         nir_def *unorm =
            nir_test_mask(b, image_flags, KK_IMAGE_FLAG_UNORM_DEPTH);

         int ref_idx = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
         nir_def *ref_src = tex->src[ref_idx].src.ssa;
         nir_def *ref = nir_f2f32(b, ref_src);
         nir_def *depth = nir_f2f32(b, nir_channel(b, border, 0));
         ref = nir_bcsel(b, unorm, nir_fsat(b, ref), ref);
         depth = nir_bcsel(b, unorm, nir_fsat(b, depth), depth);
         nir_def *at_border = compare_depth(b, op, ref, depth);
         nir_def *at_one = compare_depth(b, op, ref, nir_imm_float(b, 1.0));

         nir_def *exact = nir_ior(b, nir_ieq_imm(b, op, VK_COMPARE_OP_EQUAL),
                                  nir_ieq_imm(b, op, VK_COMPARE_OP_NOT_EQUAL));
         nir_def *split_ref =
            nir_bcsel(b, exact, nir_imm_floatN_t(b, 1.0, ref_src->bit_size),
                      nir_imm_floatN_t(b, 0.5, ref_src->bit_size));

         nir_tex_instr *split_1 =
            nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
         nir_builder_instr_insert(b, &split_1->instr);
         nir_src_rewrite(&split_1->src[ref_idx].src, split_ref);

         nir_tex_instr *split_0 =
            nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
         split_0->backend_flags |= KK_TEXTURE_FLAG_CLAMP_TO_0;
         nir_builder_instr_insert(b, &split_0->instr);
         nir_src_rewrite(&split_0->src[ref_idx].src, split_ref);

         nir_def *weight_1 = &split_1->def;
         nir_def *weight_0 = &split_0->def;
         if (tex->is_sparse) {
            weight_1 = nir_trim_vector(b, weight_1, 4);
            weight_0 = nir_trim_vector(b, weight_0, 4);
         }
         nir_def *weight = nir_fabs(b, nir_fsub(b, weight_1, weight_0));

         unsigned bit_size = clamp_to_1->bit_size;
         nir_def *delta = nir_fsub(b, nir_b2fN(b, at_border, bit_size),
                                   nir_b2fN(b, at_one, bit_size));
         replaced = nir_ffma(b, weight, delta, clamp_to_1);
      } else {
         /* Sample again, this time with clamp-to-0 instead of clamp-to-1 */
         nir_instr *clone_instr = nir_instr_clone(b->shader, &tex->instr);
         nir_builder_instr_insert(b, clone_instr);

         nir_tex_instr *tex_0 = nir_instr_as_tex(clone_instr);
         nir_def *clamp_to_0 = &tex_0->def;
         if (tex->is_sparse)
            clamp_to_0 = nir_trim_vector(b, clamp_to_0, 4);

         tex_0->backend_flags |= KK_TEXTURE_FLAG_CLAMP_TO_0;

         if (tex->op == nir_texop_tg4) {
            border =
               nir_replicate(b, nir_channel(b, border, tex->component), 4);
         }

         /* Combine together with the border */
         if (nir_alu_type_get_base_type(tex->dest_type) == nir_type_float &&
             tex->op != nir_texop_tg4) {

            /* For floats, lerp together:
             *
             * For border texels:  (1 * border) + (0 * border      ) = border
             * For regular texels: (x * border) + (x * (1 - border)) = x.
             *
             * Linear filtering is linear (duh), so lerping is compatible.
             */
            replaced = nir_flrp(b, clamp_to_0, clamp_to_1, border);
            b->shader->info.flrp_lowered = false;
         } else {
            /* For integers, just select componentwise since there is no
             * linear filtering. Gathers also use this path since they are
             * unfiltered in each component.
             */
            replaced = nir_bcsel(b, nir_ieq(b, clamp_to_0, clamp_to_1),
                                 clamp_to_0, border);
         }
      }
   }
   nir_pop_if(b, NULL);

   /* Put it together with a phi */
   nir_def *phi = nir_if_phi(b, replaced, clamp_to_1);
   if (tex->is_sparse) {
      rewrite_sparse_uses(b, &tex->def, phi, code);
      nir_instr_remove(&tex->instr);
   } else {
      nir_def_replace(&tex->def, phi);
   }
   return true;
}

bool
kk_nir_lower_custom_border(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower, nir_metadata_none, NULL);
}
