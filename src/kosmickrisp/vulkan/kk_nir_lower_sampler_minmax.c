/*
 * Copyright 2026 David Andrews
 * SPDX-License-Identifier: MIT
 */

#include "kk_shader.h"

#include "nir_builder.h"

static nir_tex_instr *
kk_new_tex(nir_builder *b, const nir_tex_instr *tex, nir_texop op,
           unsigned num_srcs, nir_alu_type dest_type, unsigned comps)
{
   nir_tex_instr *instr = nir_tex_instr_create(b->shader, num_srcs + 2);
   instr->op = op;
   instr->sampler_dim = tex->sampler_dim;
   instr->is_array = tex->is_array;
   instr->coord_components = tex->coord_components;
   instr->dest_type = dest_type;
   instr->texture_index = tex->texture_index;
   instr->sampler_index = tex->sampler_index;
   instr->backend_flags = tex->backend_flags | KK_TEXTURE_FLAG_MINMAX_DONE;

   unsigned s = num_srcs;
   int tex_deref = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   int smp_deref = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   instr->src[s++] = nir_tex_src_for_ssa(nir_tex_src_texture_deref,
                                         tex->src[tex_deref].src.ssa);
   instr->src[s++] = nir_tex_src_for_ssa(nir_tex_src_sampler_deref,
                                         tex->src[smp_deref].src.ssa);
   nir_def_init(&instr->instr, &instr->def,
                comps ? comps : nir_tex_instr_dest_size(instr),
                nir_alu_type_get_type_size(dest_type) ? : 32);
   return instr;
}

static nir_def *
kk_sample_lod(nir_builder *b, const nir_tex_instr *tex, nir_def *coord,
              nir_def *lod)
{
   nir_tex_instr *s = kk_new_tex(b, tex, nir_texop_txl, 2, tex->dest_type,
                                 tex->def.num_components);
   s->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   s->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
   nir_builder_instr_insert(b, &s->instr);
   return &s->def;
}

static nir_def *
kk_level_size(nir_builder *b, const nir_tex_instr *tex, nir_def *level,
              unsigned dims)
{
   nir_tex_instr *s = kk_new_tex(b, tex, nir_texop_txs, 1, nir_type_int32, 0);
   s->src[0] = nir_tex_src_for_ssa(nir_tex_src_lod, level);
   nir_builder_instr_insert(b, &s->instr);
   return nir_i2f32(b, nir_trim_vector(b, &s->def, dims));
}

static nir_def *
kk_num_levels(nir_builder *b, const nir_tex_instr *tex)
{
   nir_tex_instr *s =
      kk_new_tex(b, tex, nir_texop_query_levels, 0, nir_type_int32, 1);
   nir_builder_instr_insert(b, &s->instr);
   return &s->def;
}

static nir_def *
kk_implicit_lod(nir_builder *b, const nir_tex_instr *tex, nir_def *coord,
                nir_def *bias, unsigned dims)
{
   nir_tex_instr *s = kk_new_tex(b, tex, nir_texop_lod, 2, nir_type_float32, 2);
   s->coord_components = dims;
   s->src[0] =
      nir_tex_src_for_ssa(nir_tex_src_coord, nir_trim_vector(b, coord, dims));
   s->src[1] = nir_tex_src_for_ssa(nir_tex_src_bias, bias);
   nir_builder_instr_insert(b, &s->instr);
   return nir_channel(b, &s->def, 0);
}

struct kk_cube_face {
   nir_def *x_major, *y_major;
   nir_def *x, *y, *z, *ma;
};

static nir_def *
kk_cube_uv(nir_builder *b, nir_def *dir, struct kk_cube_face *f)
{
   f->x = nir_channel(b, dir, 0);
   f->y = nir_channel(b, dir, 1);
   f->z = nir_channel(b, dir, 2);
   nir_def *ax = nir_fabs(b, f->x), *ay = nir_fabs(b, f->y),
           *az = nir_fabs(b, f->z);
   f->x_major = nir_iand(b, nir_fge(b, ax, ay), nir_fge(b, ax, az));
   f->y_major = nir_iand(b, nir_inot(b, f->x_major), nir_fge(b, ay, az));
   f->ma = nir_bcsel(b, f->x_major, ax, nir_bcsel(b, f->y_major, ay, az));

   nir_def *sc = nir_bcsel(
      b, f->x_major, nir_fneg(b, nir_fmul(b, f->z, nir_fsign(b, f->x))),
      nir_bcsel(b, f->y_major, f->x, nir_fmul(b, f->x, nir_fsign(b, f->z))));
   nir_def *tc = nir_bcsel(b, f->y_major, nir_fmul(b, f->z, nir_fsign(b, f->y)),
                           nir_fneg(b, f->y));
   nir_def *st = nir_fdiv(b, nir_vec2(b, sc, tc), f->ma);
   return nir_fmul_imm(b, nir_fadd_imm(b, st, 1.0), 0.5);
}

static nir_def *
kk_cube_dir(nir_builder *b, nir_def *uv, const struct kk_cube_face *f)
{
   nir_def *st = nir_fadd_imm(b, nir_fmul_imm(b, uv, 2.0), -1.0);
   nir_def *over = nir_fmax(b, nir_fadd_imm(b, nir_fabs(b, st), -1.0),
                            nir_imm_float(b, 0));
   nir_def *major = nir_fsub(b, nir_imm_float(b, 1.0),
                             nir_fadd(b, nir_channel(b, over, 0),
                                      nir_channel(b, over, 1)));
   st = nir_fmin(b, nir_fmax(b, st, nir_imm_float(b, -1.0)),
                 nir_imm_float(b, 1.0));
   nir_def *sc = nir_channel(b, st, 0);
   nir_def *tc = nir_channel(b, st, 1);

   nir_def *x = nir_bcsel(b, f->x_major, nir_fmul(b, major, nir_fsign(b, f->x)),
                          nir_bcsel(b, f->y_major, sc,
                                    nir_fmul(b, sc, nir_fsign(b, f->z))));
   nir_def *y = nir_bcsel(b, f->y_major, nir_fmul(b, major, nir_fsign(b, f->y)),
                          nir_fneg(b, tc));
   nir_def *z = nir_bcsel(
      b, f->x_major, nir_fneg(b, nir_fmul(b, sc, nir_fsign(b, f->x))),
      nir_bcsel(b, f->y_major, nir_fmul(b, tc, nir_fsign(b, f->y)),
                nir_fmul(b, major, nir_fsign(b, f->z))));
   return nir_vec3(b, x, y, z);
}

static nir_def *
kk_reduce(nir_builder *b, nir_def *is_min, nir_def *x, nir_def *y)
{
   return nir_bcsel(b, is_min, nir_fmin(b, x, y), nir_fmax(b, x, y));
}

static nir_def *
kk_reduce_level(nir_builder *b, const nir_tex_instr *tex, nir_def *coord,
                nir_def *offset, nir_def *level, nir_def *is_min,
                nir_def *unnormalized, unsigned dims)
{
   bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   struct kk_cube_face face;
   nir_def *size = kk_level_size(b, tex, nir_f2i32(b, level), dims);
   size = nir_bcsel(b, unnormalized,
                    nir_fadd_imm(b, nir_fmul_imm(b, size, 0.0), 1.0), size);
   nir_def *uv = cube ? kk_cube_uv(b, coord, &face)
                      : nir_trim_vector(b, coord, dims);
   nir_def *p = nir_fadd_imm(b, nir_fmul(b, uv, size), -0.5);
   if (offset)
      p = nir_fadd(b, p, nir_i2f32(b, nir_trim_vector(b, offset, dims)));

   nir_def *base = nir_ffloor(b, p);
   nir_def *frac = nir_fsub(b, p, base);

   nir_def *result = NULL;
   for (unsigned corner = 0; corner < (1u << dims); corner++) {
      nir_def *texel[3];
      nir_def *used = nir_imm_true(b);
      for (unsigned c = 0; c < dims; c++) {
         bool high = corner & BITFIELD_BIT(c);
         texel[c] = nir_fadd_imm(b, nir_channel(b, base, c), high ? 1.5 : 0.5);
         if (high)
            used = nir_iand(b, used, nir_fgt_imm(b, nir_channel(b, frac, c), 0));
      }

      nir_def *centre = nir_fdiv(b, nir_vec(b, texel, dims), size);
      unsigned spatial = dims;
      if (cube) {
         centre = kk_cube_dir(b, centre, &face);
         spatial = 3;
      }
      nir_def *comps[4];
      for (unsigned c = 0; c < spatial; c++)
         comps[c] = nir_channel(b, centre, c);
      for (unsigned c = spatial; c < tex->coord_components; c++)
         comps[c] = nir_channel(b, coord, c);
      nir_def *sample = kk_sample_lod(
         b, tex, nir_vec(b, comps, tex->coord_components), level);

      if (result == NULL)
         result = sample;
      else
         result = nir_bcsel(b, used, kk_reduce(b, is_min, result, sample),
                            result);
   }

   return result;
}

static bool
lower_sampler_minmax(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
       tex->op != nir_texop_txl && tex->op != nir_texop_txd)
      return false;

   if (tex->is_shadow || tex->is_sparse ||
       (tex->backend_flags & KK_TEXTURE_FLAG_MINMAX_DONE) ||
       nir_alu_type_get_base_type(tex->dest_type) != nir_type_float)
      return false;

   unsigned dims;
   switch (tex->sampler_dim) {
   case GLSL_SAMPLER_DIM_2D:
      dims = 2;
      break;
   case GLSL_SAMPLER_DIM_3D:
      dims = 3;
      break;
   case GLSL_SAMPLER_DIM_CUBE:
      dims = 2;
      break;
   default:
      return false;
   }

   if (nir_tex_instr_src_index(tex, nir_tex_src_texture_deref) < 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref) < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *coord = nir_get_tex_src(tex, nir_tex_src_coord);
   nir_def *offset = nir_get_tex_src(tex, nir_tex_src_offset);
   nir_def *bias = nir_get_tex_src(tex, nir_tex_src_bias);
   nir_def *lod = nir_get_tex_src(tex, nir_tex_src_lod);
   nir_def *min_lod = nir_get_tex_src(tex, nir_tex_src_min_lod);
   nir_def *ddx = nir_get_tex_src(tex, nir_tex_src_ddx);
   nir_def *ddy = nir_get_tex_src(tex, nir_tex_src_ddy);

   if (coord->bit_size != 32)
      coord = nir_f2f32(b, coord);
   if (offset)
      offset = nir_i2i32(b, offset);
   if (bias)
      bias = nir_f2f32(b, bias);
   if (lod)
      lod = nir_f2f32(b, lod);
   if (min_lod)
      min_lod = nir_f2f32(b, min_lod);
   if (ddx)
      ddx = nir_f2f32(b, ddx);
   if (ddy)
      ddy = nir_f2f32(b, ddy);

   nir_tex_instr *flags_instr =
      kk_new_tex(b, tex, nir_texop_sampler_minmax_kk, 0, nir_type_uint32, 1);
   nir_builder_instr_insert(b, &flags_instr->instr);
   nir_def *flags = &flags_instr->def;

   if (lod == NULL) {
      if (ddx) {
         nir_def *size =
            kk_level_size(b, tex, nir_imm_int(b, 0), dims);
         nir_def *dx = nir_fmul(b, nir_trim_vector(b, ddx, dims), size);
         nir_def *dy = nir_fmul(b, nir_trim_vector(b, ddy, dims), size);
         nir_def *rho = nir_fmax(b, nir_fdot(b, dx, dx), nir_fdot(b, dy, dy));
         lod = nir_fmul_imm(b, nir_flog2(b, rho), 0.5);
      } else {
         lod = kk_implicit_lod(
            b, tex, coord, bias ? bias : nir_imm_float(b, 0),
            tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE ? 3 : dims);
      }
   }

   nir_def *max_level =
      nir_i2f32(b, nir_iadd_imm(b, kk_num_levels(b, tex), -1));
   nir_def *clamped = nir_fmin(b, nir_fmax(b, lod, nir_imm_float(b, 0)),
                               max_level);
   if (min_lod)
      clamped = nir_fmax(b, clamped, min_lod);

   nir_def *unnormalized =
      nir_test_mask(b, flags, KK_SAMPLER_MINMAX_UNNORMALIZED);
   clamped = nir_bcsel(b, unnormalized, nir_imm_float(b, 0), clamped);

   nir_def *mode = nir_iand_imm(b, flags, KK_SAMPLER_MINMAX_MODE_MASK);
   nir_def *is_min = nir_ieq_imm(b, mode, KK_SAMPLER_MINMAX_MIN);
   nir_def *magnify = nir_fle_imm(b, lod, 0);
   nir_def *linear = nir_bcsel(
      b, magnify, nir_test_mask(b, flags, KK_SAMPLER_MINMAX_MAG_LINEAR),
      nir_test_mask(b, flags, KK_SAMPLER_MINMAX_MIN_LINEAR));
   nir_def *mip_linear = nir_test_mask(b, flags, KK_SAMPLER_MINMAX_MIP_LINEAR);

   nir_push_if(b, nir_iand(b, nir_ine_imm(b, mode, 0), linear));
   nir_def *emulated;
   {
      nir_def *level0 = nir_bcsel(b, mip_linear, nir_ffloor(b, clamped),
                                  nir_ffloor(b, nir_fadd_imm(b, clamped, 0.5)));
      level0 = nir_fmin(b, level0, max_level);
      nir_def *level1 = nir_fmin(b, nir_fadd_imm(b, level0, 1), max_level);
      nir_def *two_levels =
         nir_iand(b, mip_linear,
                  nir_iand(b, nir_flt(b, level0, clamped),
                           nir_flt(b, level0, level1)));

      nir_def *r0 =
         kk_reduce_level(b, tex, coord, offset, level0, is_min,
                         unnormalized, dims);
      nir_push_if(b, two_levels);
      nir_def *r1 =
         kk_reduce_level(b, tex, coord, offset, level1, is_min,
                         unnormalized, dims);
      nir_def *both = kk_reduce(b, is_min, r0, r1);
      nir_pop_if(b, NULL);
      emulated = nir_if_phi(b, both, r0);
   }
   nir_push_else(b, NULL);
   nir_instr *plain = nir_instr_clone(b->shader, &tex->instr);
   nir_instr_as_tex(plain)->backend_flags |= KK_TEXTURE_FLAG_MINMAX_DONE;
   nir_builder_instr_insert(b, plain);
   nir_pop_if(b, NULL);

   nir_def *result = nir_if_phi(b, emulated, &nir_instr_as_tex(plain)->def);
   nir_def_replace(&tex->def, result);
   return true;
}

bool
kk_nir_lower_sampler_minmax(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower_sampler_minmax,
                              nir_metadata_none, NULL);
}
