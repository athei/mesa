/*
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "kk_shader.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_builtin_builder.h"

#include "util/u_dynarray.h"

/* VK_EXT_non_seamless_cube_map
 *
 * Metal always filters cube maps across faces. A sampler made with
 * VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT samples the selected face
 * alone, with the sampler's address modes applied to the face's (s, t). That
 * is a 2D array sample at the face's coordinates, from a 2D array view of the
 * cube view's faces (the descriptor's cube_array_gpu_resource_id). The
 * sampler's flag in the descriptor picks one of the two samples at run time.
 *
 * Face selection and the face coordinates follow the cube map face selection
 * table of the Vulkan spec. An explicit LOD is used as is. An implicit LOD
 * becomes gradients of the face coordinates, from the derivatives of the
 * direction by the quotient rule, which give the 2D sample the LOD the spec
 * computes for the cube. A bias scales those gradients by 2^bias. LOD queries
 * and size queries do not depend on the seamless mode and are left alone, as
 * are sparse samples, which have no 2D array residency view.
 */

struct kk_cube_face {
   nir_def *x_major, *y_major;
   /* 1.0 or -1.0, the sign of each direction component */
   nir_def *sx, *sy, *sz;
};

/* The face's (sc, tc, ma) for v, which is the direction or one of its
 * derivatives, with the major axis and signs of the direction. For the
 * direction ma is the absolute value of the major axis. */
static void
kk_cube_project(nir_builder *b, const struct kk_cube_face *f, nir_def *v,
                nir_def **sc, nir_def **tc, nir_def **ma)
{
   nir_def *x = nir_channel(b, v, 0);
   nir_def *y = nir_channel(b, v, 1);
   nir_def *z = nir_channel(b, v, 2);

   /* +X: (-z, -y, x), -X: (z, -y, -x), +Y: (x, z, y), -Y: (x, -z, -y),
    * +Z: (x, -y, z), -Z: (-x, -y, -z) */
   *sc = nir_bcsel(b, f->x_major, nir_fneg(b, nir_fmul(b, f->sx, z)),
                   nir_bcsel(b, f->y_major, x, nir_fmul(b, f->sz, x)));
   *tc = nir_bcsel(b, f->y_major, nir_fmul(b, f->sy, z), nir_fneg(b, y));
   *ma = nir_bcsel(b, f->x_major, nir_fmul(b, f->sx, x),
                   nir_bcsel(b, f->y_major, nir_fmul(b, f->sy, y),
                             nir_fmul(b, f->sz, z)));
}

/* d(0.5 * (sc, tc) / ma + 0.5) for the derivative dv of the direction */
static nir_def *
kk_cube_face_gradient(nir_builder *b, const struct kk_cube_face *f,
                      nir_def *sc, nir_def *tc, nir_def *ma, nir_def *dv)
{
   nir_def *dsc, *dtc, *dma;
   kk_cube_project(b, f, dv, &dsc, &dtc, &dma);

   nir_def *st = nir_vec2(b, sc, tc);
   nir_def *dst = nir_vec2(b, dsc, dtc);
   nir_def *rcp_ma = nir_frcp(b, ma);
   nir_def *num = nir_fsub(b, nir_fmul(b, dst, ma), nir_fmul(b, st, dma));
   return nir_fmul(b, num, nir_fmul_imm(b, nir_fmul(b, rcp_ma, rcp_ma), 0.5));
}

static nir_def *
kk_sampler_flags(nir_builder *b, nir_tex_instr *tex)
{
   return nir_channel(b,
                      nir_build_texture_query(b, tex,
                                              nir_texop_sampler_minmax_kk, 3,
                                              nir_type_uint32, false, false),
                      0);
}

/* The number of 2D array layers of the face view, 6 per cube */
static nir_def *
kk_cube_array_layers(nir_builder *b, nir_tex_instr *tex)
{
   nir_tex_instr *txs = nir_tex_instr_create(b->shader, 2);
   txs->op = nir_texop_txs;
   txs->sampler_dim = GLSL_SAMPLER_DIM_2D;
   txs->is_array = true;
   txs->dest_type = nir_type_int32;
   txs->backend_flags = KK_TEXTURE_FLAG_CUBE_AS_ARRAY;
   txs->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, nir_get_tex_src(tex, nir_tex_src_texture_deref));
   txs->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   nir_def_init(&txs->instr, &txs->def, 3, 32);
   nir_builder_instr_insert(b, &txs->instr);
   return nir_channel(b, &txs->def, 2);
}

static bool
needs_lowering(const nir_shader *nir, nir_tex_instr *tex)
{
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE || tex->is_sparse ||
       !nir_tex_instr_need_sampler(tex))
      return false;

   switch (tex->op) {
   case nir_texop_tex:
   case nir_texop_txb:
      /* Implicit derivatives */
      if (nir->info.stage != MESA_SHADER_FRAGMENT)
         return false;
      break;
   case nir_texop_txl:
   case nir_texop_tg4:
      break;
   default:
      /* txd is lowered to txl for cubes, the queries do not depend on the
       * seamless mode */
      return false;
   }

   return nir_tex_instr_src_index(tex, nir_tex_src_texture_deref) >= 0 &&
          nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref) >= 0;
}

static void
lower(nir_builder *b, nir_tex_instr *tex)
{
   bool implicit_lod = tex->op == nir_texop_tex || tex->op == nir_texop_txb;

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *coord = nir_get_tex_src(tex, nir_tex_src_coord);
   unsigned coord_bits = coord->bit_size;
   if (coord_bits != 32)
      coord = nir_f2f32(b, coord);
   nir_def *dir = nir_trim_vector(b, coord, 3);

   /* Derivatives in uniform control flow, before the branch */
   nir_def *ddx = NULL, *ddy = NULL;
   if (implicit_lod) {
      ddx = nir_ddx(b, dir);
      ddy = nir_ddy(b, dir);
   }

   nir_def *flags = kk_sampler_flags(b, tex);
   nir_push_if(b, nir_test_mask(b, flags, KK_SAMPLER_NON_SEAMLESS_CUBE));
   nir_def *face_sample;
   {
      nir_def *x = nir_channel(b, dir, 0);
      nir_def *y = nir_channel(b, dir, 1);
      nir_def *z = nir_channel(b, dir, 2);
      nir_def *ax = nir_fabs(b, x), *ay = nir_fabs(b, y), *az = nir_fabs(b, z);
      nir_def *one = nir_imm_float(b, 1.0f), *minus_one = nir_imm_float(b, -1.0f);

      struct kk_cube_face f = {
         .x_major = nir_iand(b, nir_fge(b, ax, ay), nir_fge(b, ax, az)),
         .sx = nir_bcsel(b, nir_fge_imm(b, x, 0.0f), one, minus_one),
         .sy = nir_bcsel(b, nir_fge_imm(b, y, 0.0f), one, minus_one),
         .sz = nir_bcsel(b, nir_fge_imm(b, z, 0.0f), one, minus_one),
      };
      f.y_major = nir_iand(b, nir_inot(b, f.x_major), nir_fge(b, ay, az));

      nir_def *sc, *tc, *ma;
      kk_cube_project(b, &f, dir, &sc, &tc, &ma);
      nir_def *rcp_ma = nir_frcp(b, ma);
      nir_def *s = nir_fadd_imm(b, nir_fmul_imm(b, nir_fmul(b, sc, rcp_ma), 0.5f),
                                0.5f);
      nir_def *t = nir_fadd_imm(b, nir_fmul_imm(b, nir_fmul(b, tc, rcp_ma), 0.5f),
                                0.5f);

      /* Faces in the order +X, -X, +Y, -Y, +Z, -Z */
      nir_def *positive = nir_bcsel(b, f.x_major, f.sx,
                                    nir_bcsel(b, f.y_major, f.sy, f.sz));
      nir_def *face = nir_bcsel(b, f.x_major, nir_imm_float(b, 0.0f),
                                nir_bcsel(b, f.y_major, nir_imm_float(b, 2.0f),
                                          nir_imm_float(b, 4.0f)));
      face = nir_fadd(b, face,
                      nir_bcsel(b, nir_fgt_imm(b, positive, 0.0f),
                                nir_imm_float(b, 0.0f), one));

      if (tex->is_array) {
         /* The cube is clamp(RNE(a), 0, cubes - 1), its first face at 6 times
          * that */
         nir_def *last = nir_i2f32(
            b, nir_iadd_imm(b, kk_cube_array_layers(b, tex), -6));
         nir_def *cube = nir_fround_even(b, nir_channel(b, coord, 3));
         nir_def *first = nir_fmin(b, nir_fmax(b, nir_fmul_imm(b, cube, 6.0f),
                                               nir_imm_float(b, 0.0f)),
                                   last);
         face = nir_fadd(b, face, first);
      }

      nir_def *array_coord = nir_f2fN(b, nir_vec3(b, s, t, face), coord_bits);

      nir_def *dsdx = NULL, *dsdy = NULL;
      if (implicit_lod) {
         dsdx = kk_cube_face_gradient(b, &f, sc, tc, ma, ddx);
         dsdy = kk_cube_face_gradient(b, &f, sc, tc, ma, ddy);

         nir_def *bias = nir_get_tex_src(tex, nir_tex_src_bias);
         if (bias) {
            nir_def *scale = nir_fexp2(b, nir_f2f32(b, bias));
            dsdx = nir_fmul(b, dsdx, scale);
            dsdy = nir_fmul(b, dsdy, scale);
         }
         dsdx = nir_f2fN(b, dsdx, coord_bits);
         dsdy = nir_f2fN(b, dsdy, coord_bits);
      }

      /* Sources are rewritten once the clone is in the shader */
      nir_tex_instr *array =
         nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
      nir_builder_instr_insert(b, &array->instr);
      array->sampler_dim = GLSL_SAMPLER_DIM_2D;
      array->is_array = true;
      array->coord_components = 3;
      array->backend_flags |= KK_TEXTURE_FLAG_CUBE_AS_ARRAY;
      nir_src_rewrite(&array->src[nir_tex_instr_src_index(array,
                                                          nir_tex_src_coord)]
                          .src,
                      array_coord);

      if (implicit_lod) {
         nir_steal_tex_src(array, nir_tex_src_bias);
         array->op = nir_texop_txd;
         nir_tex_instr_add_src(array, nir_tex_src_ddx, dsdx);
         nir_tex_instr_add_src(array, nir_tex_src_ddy, dsdy);
      }

      face_sample = &array->def;
   }
   nir_push_else(b, NULL);
   nir_def *cube_sample;
   {
      nir_instr *cube = nir_instr_clone(b->shader, &tex->instr);
      nir_builder_instr_insert(b, cube);
      cube_sample = &nir_instr_as_tex(cube)->def;
   }
   nir_pop_if(b, NULL);

   nir_def_replace(&tex->def, nir_if_phi(b, face_sample, cube_sample));
}

bool
kk_nir_lower_non_seamless_cube(nir_shader *nir)
{
   bool progress = false;

   nir_foreach_function_impl(impl, nir) {
      /* Gather first: the lowering clones the instruction it lowers */
      struct util_dynarray texs;
      util_dynarray_init(&texs, NULL);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex &&
                needs_lowering(nir, nir_instr_as_tex(instr)))
               util_dynarray_append(&texs, nir_instr_as_tex(instr));
         }
      }

      nir_builder b = nir_builder_create(impl);
      util_dynarray_foreach(&texs, nir_tex_instr *, tex)
         lower(&b, *tex);

      progress |= nir_progress(util_dynarray_num_elements(&texs,
                                                          nir_tex_instr *) > 0,
                               impl, nir_metadata_none);
      util_dynarray_fini(&texs);
   }

   return progress;
}
