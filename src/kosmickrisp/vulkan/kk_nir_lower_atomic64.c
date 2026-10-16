/*
 * Copyright 2026 David Andrews
 * SPDX-License-Identifier: MIT
 */

#include "kk_cmd_buffer.h"
#include "kk_shader.h"

#include "nir_builder.h"

enum target_kind {
   TARGET_GLOBAL,
   TARGET_SHARED,
   TARGET_IMAGE,
};

struct target {
   enum target_kind kind;
   nir_def *addr;
   unsigned base;
   nir_intrinsic_instr *image;
};

static nir_def *
load_arena(nir_builder *b)
{
   nir_def *root = nir_load_buffer_ptr_kk(b, 1, 64, .binding = 0);
   return nir_build_load_global_constant(
      b, 1, 64,
      nir_iadd_imm(b, root,
                   offsetof(struct kk_root_descriptor_table, atomic64)),
      .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
}

static nir_def *
load_locked_flag(nir_builder *b, unsigned class)
{
   return nir_build_load_global(b, 1, 32,
                                nir_iadd_imm(b, load_arena(b), 4u * class),
                                .align_mul = 4, .access = ACCESS_ATOMIC);
}

static nir_def *
lock_key(nir_builder *b, const struct target *t)
{
   switch (t->kind) {
   case TARGET_GLOBAL:
      return nir_u2u32(b, nir_ushr_imm(b, t->addr, 3));
   case TARGET_SHARED:
      return nir_iadd_imm(b, nir_ushr_imm(b, t->addr, 3), t->base);
   case TARGET_IMAGE: {
      static const uint32_t mul[] = {0x9e3779b1u, 0x85ebca77u, 0xc2b2ae3du};
      nir_def *coord = nir_u2u32(b, t->image->src[1].ssa);
      unsigned comps = MIN2(nir_image_intrinsic_coord_components(t->image),
                            ARRAY_SIZE(mul));
      nir_def *key = nir_imm_int(b, 0);
      for (unsigned i = 0; i < comps; i++)
         key = nir_iadd(b, key, nir_imul_imm(b, nir_channel(b, coord, i), mul[i]));
      return nir_ixor(b, key, nir_ushr_imm(b, key, 15));
   }
   }
   UNREACHABLE("bad target");
}

static nir_def *
lock_address(nir_builder *b, const struct target *t)
{
   nir_def *key = lock_key(b, t);
   nir_def *slot = nir_iand_imm(b, key, KK_ATOMIC64_LOCK_COUNT - 1u);
   nir_def *offset =
      nir_iadd_imm(b, nir_ishl_imm(b, slot, 2), KK_ATOMIC64_LOCK_OFFSET);
   return nir_iadd(b, load_arena(b), nir_u2u64(b, offset));
}

static void
fence(nir_builder *b, const struct target *t)
{
   static const nir_variable_mode modes[] = {
      [TARGET_GLOBAL] = nir_var_mem_ssbo,
      [TARGET_SHARED] = nir_var_mem_shared,
      [TARGET_IMAGE] = nir_var_image,
   };
   nir_barrier(b, .execution_scope = SCOPE_NONE, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = modes[t->kind]);
}

static nir_def *
load_value(nir_builder *b, const struct target *t)
{
   if (t->kind == TARGET_GLOBAL)
      return nir_build_load_global(b, 1, 64, t->addr, .align_mul = 8);

   if (t->kind == TARGET_IMAGE) {
      const nir_intrinsic_instr *image = t->image;
      return nir_bindless_image_load(
         b, 1, 64, image->src[0].ssa, image->src[1].ssa, nir_imm_int(b, 0),
         nir_imm_int(b, 0), .image_dim = nir_intrinsic_image_dim(image),
         .image_array = nir_intrinsic_image_array(image),
         .format = nir_intrinsic_format(image),
         .access = nir_intrinsic_access(image) & ~ACCESS_ATOMIC,
         .dest_type = nir_type_uint64);
   }

   return nir_load_shared(b, 1, 64, t->addr, .base = t->base, .align_mul = 8);
}

static void
store_value(nir_builder *b, const struct target *t, nir_def *value)
{
   if (t->kind == TARGET_GLOBAL) {
      nir_build_store_global(b, value, t->addr, .align_mul = 8,
                             .write_mask = 1);
   } else if (t->kind == TARGET_IMAGE) {
      const nir_intrinsic_instr *image = t->image;
      nir_bindless_image_store(
         b, image->src[0].ssa, image->src[1].ssa, nir_imm_int(b, 0),
         nir_pad_vec4(b, value), nir_imm_int(b, 0),
         .image_dim = nir_intrinsic_image_dim(image),
         .image_array = nir_intrinsic_image_array(image),
         .format = nir_intrinsic_format(image),
         .access = nir_intrinsic_access(image) & ~ACCESS_ATOMIC,
         .src_type = nir_type_uint64);
   } else
      nir_store_shared(b, value, t->addr, .base = t->base, .align_mul = 8,
                       .write_mask = 1);
}

static nir_def *
apply(nir_builder *b, nir_atomic_op op, nir_def *old, nir_def *data,
      nir_def *swap)
{
   switch (op) {
   case nir_atomic_op_iadd:
      return nir_iadd(b, old, data);
   case nir_atomic_op_imin:
      return nir_imin(b, old, data);
   case nir_atomic_op_umin:
      return nir_umin(b, old, data);
   case nir_atomic_op_imax:
      return nir_imax(b, old, data);
   case nir_atomic_op_umax:
      return nir_umax(b, old, data);
   case nir_atomic_op_iand:
      return nir_iand(b, old, data);
   case nir_atomic_op_ior:
      return nir_ior(b, old, data);
   case nir_atomic_op_ixor:
      return nir_ixor(b, old, data);
   case nir_atomic_op_xchg:
      return data;
   case nir_atomic_op_cmpxchg:
      return nir_bcsel(b, nir_ieq(b, old, data), swap, old);
   default:
      UNREACHABLE("not an integer atomic operation");
   }
}

static nir_def *
critical_section(nir_builder *b, const struct target *t, nir_atomic_op op,
                 nir_def *data, nir_def *swap)
{
   fence(b, t);
   nir_def *old = load_value(b, t);
   fence(b, t);
   store_value(b, t, apply(b, op, old, data, swap));
   fence(b, t);
   return old;
}

static nir_def *
try_lock(nir_builder *b, nir_def *lock)
{
   nir_def *held = nir_global_atomic_swap(b, 32, lock, nir_imm_int(b, 0),
                                          nir_imm_int(b, 1),
                                          .atomic_op = nir_atomic_op_cmpxchg);
   return nir_ieq_imm(b, held, 0);
}

static nir_def *
unlock(nir_builder *b, nir_def *lock)
{
   return nir_global_atomic(b, 32, lock, nir_imm_int(b, 0),
                            .atomic_op = nir_atomic_op_xchg);
}

static void
report_timeout(nir_builder *b)
{
   nir_global_atomic(b, 32,
                     nir_iadd_imm(b, load_arena(b), 4u * KK_ATOMIC64_TIMEOUT),
                     nir_imm_int(b, 1), .atomic_op = nir_atomic_op_xchg);
}

static nir_def *
locked(nir_builder *b, const struct target *t, nir_atomic_op op, nir_def *data,
       nir_def *swap)
{
   nir_function_impl *impl = b->impl;
   nir_variable *result =
      nir_local_variable_create(impl, glsl_uint64_t_type(), "atomic64_old");
   nir_variable *state =
      nir_local_variable_create(impl, glsl_uint_type(), "atomic64_state");
   nir_variable *spins =
      nir_local_variable_create(impl, glsl_uint_type(), "atomic64_spins");

   nir_def *lock = lock_address(b, t);
   nir_store_var(b, state, nir_imm_int(b, 0), 1);
   nir_store_var(b, spins, nir_imm_int(b, 0), 1);
   nir_store_var(b, result, nir_imm_int64(b, 0), 1);

   nir_push_loop(b);
   {
      nir_def *s = nir_load_var(b, state);
      nir_break_if(b, nir_ieq_imm(b, s, 2));
      nir_push_if(b, nir_ieq_imm(b, s, 1));
      {
         nir_store_var(b, result, critical_section(b, t, op, data, swap), 1);
         nir_store_var(b, state,
                       nir_iadd_imm(b, nir_ior_imm(b, unlock(b, lock), 1), 1),
                       1);
      }
      nir_push_else(b, NULL);
      {
         nir_def *spun = nir_load_var(b, spins);
         nir_def *enter = try_lock(b, lock);
         nir_def *timeout = nir_iand(b, nir_inot(b, enter),
                                     nir_uge_imm(b, spun, KK_ATOMIC64_SPIN_LIMIT));
         nir_store_var(b, spins, nir_iadd_imm(b, spun, 1), 1);
         nir_store_var(b, state,
                       nir_bcsel(b, enter, nir_imm_int(b, 1),
                                 nir_bcsel(b, timeout, nir_imm_int(b, 2),
                                           nir_imm_int(b, 0))),
                       1);
         nir_push_if(b, timeout);
         report_timeout(b);
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, NULL);

   return nir_load_var(b, result);
}

static bool
is_atomic64(const nir_intrinsic_instr *intrin)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap:
      return intrin->def.bit_size == 64;
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant_offset:
      return intrin->def.bit_size == 64 &&
             (nir_intrinsic_access(intrin) & ACCESS_ATOMIC);
   case nir_intrinsic_load_shared:
   case nir_intrinsic_bindless_image_load:
      return intrin->def.bit_size == 64 &&
             (nir_intrinsic_access(intrin) & ACCESS_ATOMIC);
   case nir_intrinsic_store_global:
   case nir_intrinsic_store_shared:
      return intrin->src[0].ssa->bit_size == 64 &&
             (nir_intrinsic_access(intrin) & ACCESS_ATOMIC);
   case nir_intrinsic_bindless_image_store:
      return intrin->src[3].ssa->bit_size == 64 &&
             (nir_intrinsic_access(intrin) & ACCESS_ATOMIC);
   default:
      return false;
   }
}

static void
lower_atomic(nir_builder *b, nir_intrinsic_instr *intrin, uint8_t *locked_out)
{
   b->cursor = nir_before_instr(&intrin->instr);

   if (intrin->intrinsic == nir_intrinsic_load_global ||
       intrin->intrinsic == nir_intrinsic_load_global_constant_offset ||
       intrin->intrinsic == nir_intrinsic_load_shared ||
       intrin->intrinsic == nir_intrinsic_bindless_image_load) {
      enum target_kind kind =
         intrin->intrinsic == nir_intrinsic_load_shared ? TARGET_SHARED
         : intrin->intrinsic == nir_intrinsic_bindless_image_load
            ? TARGET_IMAGE
            : TARGET_GLOBAL;
      fence(b, &(struct target){.kind = kind});
      nir_intrinsic_set_access(intrin,
                               nir_intrinsic_access(intrin) & ~ACCESS_ATOMIC);
      return;
   }

   if (intrin->intrinsic == nir_intrinsic_store_global) {
      struct target t = {.kind = TARGET_GLOBAL, .addr = intrin->src[1].ssa};
      locked(b, &t, nir_atomic_op_xchg, intrin->src[0].ssa, NULL);
      *locked_out |= BITFIELD_BIT(KK_ATOMIC64_BUFFER);
      nir_instr_remove(&intrin->instr);
      return;
   }

   if (intrin->intrinsic == nir_intrinsic_store_shared) {
      struct target t = {.kind = TARGET_SHARED,
                         .addr = intrin->src[1].ssa,
                         .base = nir_intrinsic_base(intrin)};
      locked(b, &t, nir_atomic_op_xchg, intrin->src[0].ssa, NULL);
      nir_instr_remove(&intrin->instr);
      return;
   }

   if (intrin->intrinsic == nir_intrinsic_bindless_image_store) {
      struct target t = {.kind = TARGET_IMAGE, .image = intrin};
      locked(b, &t, nir_atomic_op_xchg, nir_channel(b, intrin->src[3].ssa, 0),
             NULL);
      *locked_out |= BITFIELD_BIT(KK_ATOMIC64_IMAGE);
      nir_instr_remove(&intrin->instr);
      return;
   }

   const nir_atomic_op op = nir_intrinsic_atomic_op(intrin);

   if (intrin->intrinsic == nir_intrinsic_bindless_image_atomic ||
       intrin->intrinsic == nir_intrinsic_bindless_image_atomic_swap) {
      const bool is_swap =
         intrin->intrinsic == nir_intrinsic_bindless_image_atomic_swap;
      struct target t = {.kind = TARGET_IMAGE, .image = intrin};
      nir_def *data = intrin->src[3].ssa;
      nir_def *swap = is_swap ? intrin->src[4].ssa : NULL;
      nir_def *atomic_texture = nir_ine_imm(b, intrin->src[2].ssa, 0);
      nir_src_rewrite(&intrin->src[2], nir_imm_int(b, 0));

      nir_instr *next = nir_instr_next(&intrin->instr);
      if (next && next->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(next)->intrinsic ==
             nir_intrinsic_bindless_image_fence_kk)
         nir_instr_remove(next);

      const bool native = !is_swap && nir_def_is_unused(&intrin->def) &&
                          (op == nir_atomic_op_umax || op == nir_atomic_op_umin);
      if (native) {
         nir_def *unlocked =
            nir_ieq_imm(b, load_locked_flag(b, KK_ATOMIC64_IMAGE), 0);
         nir_push_if(b, nir_iand(b, unlocked, atomic_texture));
         nir_instr_remove(&intrin->instr);
         nir_builder_instr_insert(b, &intrin->instr);
         nir_push_else(b, NULL);
         locked(b, &t, op, data, swap);
         nir_pop_if(b, NULL);
         return;
      }

      *locked_out |= BITFIELD_BIT(KK_ATOMIC64_IMAGE);
      nir_def_replace(&intrin->def, locked(b, &t, op, data, swap));
      return;
   }

   const bool is_swap = intrin->intrinsic == nir_intrinsic_global_atomic_swap ||
                        intrin->intrinsic == nir_intrinsic_shared_atomic_swap;
   const bool is_global = intrin->intrinsic == nir_intrinsic_global_atomic ||
                          intrin->intrinsic == nir_intrinsic_global_atomic_swap;

   struct target t = {
      .kind = is_global ? TARGET_GLOBAL : TARGET_SHARED,
      .addr = intrin->src[0].ssa,
      .base = is_global ? 0 : nir_intrinsic_base(intrin),
   };
   nir_def *data = intrin->src[1].ssa;
   nir_def *swap = is_swap ? intrin->src[2].ssa : NULL;

   const bool native = is_global && nir_def_is_unused(&intrin->def) &&
                       (op == nir_atomic_op_umax || op == nir_atomic_op_umin);
   if (native) {
      nir_push_if(b, nir_ieq_imm(b, load_locked_flag(b, KK_ATOMIC64_BUFFER), 0));
      nir_global_atomic(b, 64, t.addr, data, .atomic_op = op);
      nir_push_else(b, NULL);
      locked(b, &t, op, data, swap);
      nir_pop_if(b, NULL);
      nir_instr_remove(&intrin->instr);
      return;
   }

   if (is_global)
      *locked_out |= BITFIELD_BIT(KK_ATOMIC64_BUFFER);

   nir_def_replace(&intrin->def, locked(b, &t, op, data, swap));
}

static bool
has_atomic64(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                is_atomic64(nir_instr_as_intrinsic(instr)))
               return true;
         }
      }
   }
   return false;
}

bool
kk_nir_lower_atomic64(nir_shader *nir, uint8_t *locked_out)
{
   if (!has_atomic64(nir))
      return false;

   bool cleaned;
   do {
      cleaned = false;
      NIR_PASS(cleaned, nir, nir_lower_vars_to_ssa);
      NIR_PASS(cleaned, nir, nir_opt_copy_prop);
      NIR_PASS(cleaned, nir, nir_opt_constant_folding);
      NIR_PASS(cleaned, nir, nir_opt_dead_cf);
      NIR_PASS(cleaned, nir, nir_opt_dce);
   } while (cleaned);

   bool progress = false;

   nir_foreach_function_impl(impl, nir) {
      struct util_dynarray atomics;
      util_dynarray_init(&atomics, NULL);

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (is_atomic64(intrin))
               util_dynarray_append(&atomics, intrin);
         }
      }

      nir_builder b = nir_builder_create(impl);
      util_dynarray_foreach(&atomics, nir_intrinsic_instr *, intrin)
         lower_atomic(&b, *intrin, locked_out);

      if (util_dynarray_num_elements(&atomics, nir_intrinsic_instr *)) {
         progress = true;
         nir_progress(true, impl, nir_metadata_none);
      }
      util_dynarray_fini(&atomics);
   }

   return progress;
}
