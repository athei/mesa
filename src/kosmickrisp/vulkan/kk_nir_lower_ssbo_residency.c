/*
 * Copyright 2026 David Andrews
 * SPDX-License-Identifier: MIT
 */

#include "kk_cmd_buffer.h"
#include "kk_shader.h"

#include "nir_builder.h"
#include "nir_control_flow.h"

static bool
zero_descriptor_offset(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_vulkan_descriptor ||
       nir_intrinsic_desc_type(intr) != nir_descriptor_type_storage_buffer)
      return false;

   b->cursor = nir_after_instr(&intr->instr);
   nir_def *zeroed = nir_vector_insert_imm(b, &intr->def, nir_imm_int(b, 0), 3);
   nir_def_rewrite_uses_after(&intr->def, zeroed);
   return true;
}

static nir_def *
ssbo_descriptor(nir_deref_instr *deref)
{
   while (deref && deref->deref_type != nir_deref_type_cast)
      deref = nir_deref_instr_parent(deref);
   if (deref == NULL)
      return NULL;

   nir_scalar base = nir_scalar_resolved(deref->parent.ssa, 0);
   if (!nir_def_is_intrinsic(base.def))
      return NULL;

   nir_intrinsic_instr *desc = nir_def_as_intrinsic(base.def);
   if (desc->intrinsic != nir_intrinsic_load_vulkan_descriptor)
      return NULL;

   return &desc->def;
}

static nir_src *
global_address(nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
      return &intr->src[0];
   case nir_intrinsic_store_global:
      return &intr->src[1];
   default:
      return NULL;
   }
}

static bool
mask_global_address(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_src *addr = global_address(intr);
   if (addr == NULL)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(addr, nir_iand_imm(b, addr->ssa, KK_SPARSE_ADDRESS_MASK));
   return true;
}

static bool
has_guarded_store(nir_function_impl *impl);

static bool
mask_load_addresses(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      if (has_guarded_store(impl))
         continue;

      progress |= nir_function_intrinsics_pass(
         impl, mask_global_address, nir_metadata_control_flow, NULL);
   }
   return progress;
}

static bool
is_guarded_store(nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_store_deref:
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap:
      break;
   case nir_intrinsic_store_global:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
      return true;
   default:
      return false;
   }

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   return nir_deref_mode_is(deref, nir_var_mem_ssbo) &&
          ssbo_descriptor(deref) != NULL;
}

static nir_def *
load_residency_map(nir_builder *b)
{
   nir_def *root = nir_load_buffer_ptr_kk(b, 1, 64, .binding = 0);
   return nir_build_load_global_constant(
      b, 1, 64,
      nir_iadd_imm(b, root,
                   offsetof(struct kk_root_descriptor_table, sparse_residency)),
      .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
}

static nir_def *
ssbo_page_resident(nir_builder *b, nir_def *word, nir_def *addr,
                   unsigned offset_B)
{
   nir_def *in_page =
      nir_iand_imm(b, nir_channel(b, addr, 0), KK_SPARSE_PAGE_SIZE - 1);
   nir_def *offset = nir_iadd_imm(b, nir_channel(b, addr, 3), offset_B);
   nir_def *page = nir_ushr_imm(b, nir_iadd(b, in_page, offset),
                                util_logbase2(KK_SPARSE_PAGE_SIZE));

   nir_def *map = load_residency_map(b);
   nir_def *entry = nir_iadd(
      b, map, nir_u2u64(b, nir_iadd(b, nir_iadd_imm(b, word, -1), page)));
   return nir_ine_imm(b, nir_build_load_global(b, 1, 8, entry, .align_mul = 1),
                      0);
}

static nir_intrinsic_instr *
insert_guarded(nir_builder *b, nir_intrinsic_instr *intr, nir_def *resident,
               int component, nir_def *address)
{
   nir_push_if(b, resident);
   nir_intrinsic_instr *store =
      nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intr->instr));
   if (component >= 0)
      nir_intrinsic_set_write_mask(store, BITFIELD_BIT(component));
   nir_builder_instr_insert(b, &store->instr);
   if (address)
      nir_src_rewrite(global_address(store), address);
   nir_pop_if(b, NULL);
   return store;
}

static void
finish_guard(nir_builder *b, nir_intrinsic_instr *intr, nir_def *sparse_result)
{
   nir_push_else(b, NULL);
   nir_instr_remove(&intr->instr);
   nir_builder_instr_insert(b, &intr->instr);
   nir_pop_if(b, NULL);

   if (sparse_result) {
      nir_def *result = nir_if_phi(b, sparse_result, &intr->def);
      nir_foreach_use_including_if_safe(use, &intr->def) {
         if (nir_src_is_if(use) ||
             nir_src_use_instr(use) != nir_def_instr(result))
            nir_src_rewrite(use, result);
      }
   }
}

static void
guard_global_store(nir_builder *b, nir_intrinsic_instr *intr)
{
   bool has_dest = nir_intrinsic_infos[intr->intrinsic].has_dest;
   nir_def *raw = global_address(intr)->ssa;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *zero = has_dest ? nir_imm_zero(b, intr->def.num_components,
                                           intr->def.bit_size)
                            : NULL;
   nir_def *tag =
      nir_u2u32(b, nir_ushr_imm(b, raw, KK_SPARSE_TAG_SHIFT));

   nir_def *sparse_result = NULL;
   nir_push_if(b, nir_ine_imm(b, tag, 0));
   {
      nir_def *masked = nir_iand_imm(b, raw, KK_SPARSE_ADDRESS_MASK);
      nir_def *map = load_residency_map(b);
      nir_def *entry = nir_iadd(
         b, nir_iadd_imm(b, map, KK_SPARSE_TAG_TABLE_OFFSET),
         nir_u2u64(b, nir_imul_imm(b, tag, 16)));
      nir_def *base = nir_build_load_global_constant(
         b, 1, 64, entry, .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
      nir_def *first = nir_build_load_global_constant(
         b, 1, 64, nir_iadd_imm(b, entry, 8), .align_mul = 8,
         .access = ACCESS_CAN_SPECULATE);
      nir_def *offset = nir_isub(b, masked, base);

      bool store = intr->intrinsic == nir_intrinsic_store_global;
      unsigned comp_B = store ? intr->src[0].ssa->bit_size / 8 : 0;
      unsigned mask = store ? nir_intrinsic_write_mask(intr) : 1;
      u_foreach_bit(c, mask) {
         nir_def *page = nir_ushr_imm(b, nir_iadd_imm(b, offset, c * comp_B),
                                      util_logbase2(KK_SPARSE_PAGE_SIZE));
         nir_def *byte = nir_iadd(b, map,
                                  nir_iadd(b, nir_iadd_imm(b, first, -1), page));
         nir_def *resident = nir_ine_imm(
            b, nir_build_load_global(b, 1, 8, byte, .align_mul = 1), 0);

         nir_intrinsic_instr *guarded =
            insert_guarded(b, intr, resident, store ? (int)c : -1, masked);
         if (!store)
            sparse_result = nir_if_phi(b, &guarded->def, zero);
      }
   }
   finish_guard(b, intr, sparse_result);
}

static void
guard_store(nir_builder *b, nir_intrinsic_instr *intr,
            nir_address_format addr_format)
{
   if (global_address(intr)) {
      guard_global_store(b, intr);
      return;
   }

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_def *desc = ssbo_descriptor(deref);
   bool has_dest = nir_intrinsic_infos[intr->intrinsic].has_dest;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *zero = has_dest ? nir_imm_zero(b, intr->def.num_components,
                                           intr->def.bit_size)
                            : NULL;
   nir_def *word = nir_channel(b, desc, 3);

   nir_def *sparse_result = NULL;
   nir_push_if(b, nir_ine_imm(b, word, 0));
   {
      nir_deref_instr *cast = deref;
      while (cast->deref_type != nir_deref_type_cast)
         cast = nir_deref_instr_parent(cast);

      nir_def *addr = nir_explicit_io_address_from_deref(
         b, deref, cast->parent.ssa, addr_format);

      if (intr->intrinsic == nir_intrinsic_store_deref) {
         nir_def *value = intr->src[1].ssa;
         unsigned comp_B = value->bit_size / 8;
         u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
            insert_guarded(b, intr,
                           ssbo_page_resident(b, word, addr, c * comp_B), c,
                           NULL);
         }
      } else {
         nir_push_if(b, ssbo_page_resident(b, word, addr, 0));
         nir_intrinsic_instr *guarded =
            nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intr->instr));
         nir_builder_instr_insert(b, &guarded->instr);
         nir_pop_if(b, NULL);
         sparse_result = nir_if_phi(b, &guarded->def, zero);
      }
   }
   finish_guard(b, intr, sparse_result);
}

static bool
has_guarded_store(nir_function_impl *impl)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             is_guarded_store(nir_instr_as_intrinsic(instr)))
            return true;
      }
   }
   return false;
}

bool
kk_nir_lower_ssbo_residency(nir_shader *nir, nir_address_format addr_format,
                            bool guard)
{
   bool progress = nir_shader_intrinsics_pass(
      nir, zero_descriptor_offset, nir_metadata_control_flow, NULL);

   if (!guard)
      return progress;

   progress |= mask_load_addresses(nir);

   nir_foreach_function_impl(impl, nir) {
      if (!has_guarded_store(impl))
         continue;

      nir_builder b = nir_builder_create(impl);
      nir_cf_list body;
      b.cursor =
         nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));

      nir_def *map = load_residency_map(&b);
      nir_def *live = nir_build_load_global_constant(
         &b, 1, 32, map, .align_mul = 4, .access = ACCESS_CAN_SPECULATE);

      nir_if *nif = nir_push_if(&b, nir_ine_imm(&b, live, 0));
      {
         nir_cf_list guarded;
         nir_cf_node *parent = &nir_cursor_current_block(b.cursor)->cf_node;
         nir_cf_list_clone(&guarded, &body, parent, NULL);
         nir_cf_reinsert(&guarded, b.cursor);
         b.cursor = nir_after_cf_list(&nif->then_list);
      }
      nir_push_else(&b, nif);
      {
         nir_cf_reinsert(&body, b.cursor);
         b.cursor = nir_after_cf_list(&nif->else_list);
      }
      nir_pop_if(&b, nif);

      struct util_dynarray stores = UTIL_DYNARRAY_INIT;
      foreach_list_typed(nir_cf_node, node, node, &nif->then_list) {
         nir_foreach_block_in_cf_node(block, node) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;

               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (is_guarded_store(intr))
                  util_dynarray_append(&stores, intr);
               else
                  mask_global_address(&b, intr, NULL);
            }
         }
      }

      util_dynarray_foreach(&stores, nir_intrinsic_instr *, store)
         guard_store(&b, *store, addr_format);
      util_dynarray_fini(&stores);

      progress |= nir_progress(true, impl, nir_metadata_none);
   }
   return progress;
}
