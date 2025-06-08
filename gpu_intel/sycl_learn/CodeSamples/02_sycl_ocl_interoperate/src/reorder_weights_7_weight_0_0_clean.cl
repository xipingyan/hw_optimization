# 0 "./reorder_weights_7_weight_0_0.cl"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "./reorder_weights_7_weight_0_0.cl"




inline float _convert_as_bfloat16_float(ushort source) {
 uint u = 0;
 if ( (source>>15) ) {
 u = 1 << 31;
 }
 u += ( ( (source >> 7) & 0b11111111)) << 23;
 u += (source & 0b1111111) << 16;
 float* f = &u;
 return *f;
}




inline ushort _convert_bfloat16_as_ushort(float source) {
 uint* in = &source;
 ushort u = 0;
 if ( (*in>>31) ) {
 u = 1 << 15;
 }
 u += ( ( (*in >> 23) & 0b11111111)) << 7;
 u += (*in >> 16) & 0b1111111;
 return u;
}
# 93 "./reorder_weights_7_weight_0_0.cl"
inline uint get_b_fs_yx_fsv_index(uint b, uint f, uint y, uint x,
 uint x_size, uint y_size, uint f_size, uint b_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after, uint alignment) {
 const uint feature = f + f_pad_before;
 const uint fs = feature / alignment;
 const uint fsv = feature % alignment;
 const uint x_pitch = alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint fs_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = (b_pad_before + b) * b_pitch +
 fs * fs_pitch +
 (y_pad_before + y) * y_pitch +
 (x_pad_before + x) * x_pitch
 + fsv;
 return output_offset;
}
inline uint get_b_fs_yx_fsv_index_safe(uint b, uint f, uint y, uint x,
 uint x_size, uint y_size, uint f_size, uint b_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after, uint alignment) {
 const uint f_mod = f_pad_before + (f % f_size);
 const uint fs = f_mod / alignment;
 const uint fsv = f_mod % alignment;
 const uint x_pitch = alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint fs_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = (b_pad_before + (b % b_size)) * b_pitch +
 fs * fs_pitch +
 (y_pad_before + (y % y_size)) * y_pitch +
 (x_pad_before + (x % x_size)) * x_pitch
 + fsv;
 return output_offset;
}
# 145 "./reorder_weights_7_weight_0_0.cl"
inline uint get_fs_b_yx_fsv32_index(uint b, uint f, uint y, uint x,
 uint x_pad_before, uint x_size, uint x_pad_after,
 uint y_pad_before, uint y_size, uint y_pad_after,
 uint f_pad_before,
 uint size_b)
{
 const uint feature_tile_size = 32;
 const uint x_total_size = x_pad_before + x_size + x_pad_after;
 const uint y_total_size = y_pad_before + y_size + y_pad_after;
 const uint real_x = x + x_pad_before;
 const uint real_y = y + y_pad_before;
 const uint real_f = f + f_pad_before;
 const uint x_pitch = feature_tile_size;
 const uint y_pitch = x_pitch * x_total_size;
 const uint b_pitch = y_pitch * y_total_size;
 const uint f_tile_pitch = b_pitch * size_b;
 const uint feature_tile_number = real_f / feature_tile_size;
 const uint feature_local_number = real_f % feature_tile_size;
 size_t index = 0;
 index += feature_tile_number * f_tile_pitch;
 index += b * b_pitch;
 index += real_y * y_pitch;
 index += real_x * x_pitch;
 index += feature_local_number;
 return index;
}
inline uint get_fs_b_yx_fsv32_index_safe(uint b, uint f, uint y, uint x,
 uint x_pad_before, uint x_size, uint x_pad_after,
 uint y_pad_before, uint y_size, uint y_pad_after,
 uint f_pad_before, uint f_size,
 uint size_b)
{
 const uint feature_tile_size = 32;
 const uint x_total_size = x_pad_before + x_size + x_pad_after;
 const uint y_total_size = y_pad_before + y_size + y_pad_after;
 const uint real_x = (x % x_size) + x_pad_before;
 const uint real_y = (y % y_size) + y_pad_before;
 const uint real_f = (f % f_size) + f_pad_before;
 const uint x_pitch = feature_tile_size;
 const uint y_pitch = x_pitch * x_total_size;
 const uint b_pitch = y_pitch * y_total_size;
 const uint f_tile_pitch = b_pitch * size_b;
 const uint feature_tile_number = real_f / feature_tile_size;
 const uint feature_local_number = real_f % feature_tile_size;
 size_t index = 0;
 index += feature_tile_number * f_tile_pitch;
 index += b * b_pitch;
 index += real_y * y_pitch;
 index += real_x * x_pitch;
 index += feature_local_number;
 return index;
}
# 209 "./reorder_weights_7_weight_0_0.cl"
inline uint get_b_fs_zyx_fsv_index(uint b, uint f, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint f_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint z_pad_before, uint z_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after,
 uint alignment)
{
 const uint feature = f + f_pad_before;
 const uint fs = feature / alignment;
 const uint fsv = feature % alignment;
 const uint x_pitch = alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = (b_pad_before + b) * b_pitch +
 fs * fs_pitch +
 (z_pad_before + z) * z_pitch +
 (y_pad_before + y) * y_pitch +
 (x_pad_before + x) * x_pitch
 + fsv;
 return output_offset;
}
inline uint get_b_fs_zyx_fsv_index_safe(uint b, uint f, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint f_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint z_pad_before, uint z_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after,
 uint alignment) {
 const uint f_mod = f_pad_before + (f % f_size);
 const uint fs = f_mod / alignment;
 const uint fsv = f_mod % alignment;
 const uint x_pitch = alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = (b_pad_before + b) * b_pitch +
 fs * fs_pitch +
 (z_pad_before + (z % z_size)) * z_pitch +
 (y_pad_before + (y % y_size)) * y_pitch +
 (x_pad_before + (x % x_size)) * x_pitch
 + fsv;
 return output_offset;
}
inline uint get_bs_fs_zyx_bsv_fsv_index_safe(uint b, uint f, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint f_size, uint b_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint z_pad_before, uint z_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after, uint alignmentB, uint alignmentF) {
 const uint b_mod = b_pad_before + (b % b_size);
 const uint f_mod = f_pad_before + (f % f_size);
 const uint bs = b_mod / alignmentB;
 const uint bsv = b_mod % alignmentB;
 const uint fs = f_mod / alignmentF;
 const uint fsv = f_mod % alignmentF;
 const uint x_pitch = alignmentF * alignmentB;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
 const uint bs_pitch = fs_pitch * ((total_f_size + alignmentF - 1) / alignmentF);
 const uint output_offset = bs * bs_pitch +
 fs * fs_pitch +
 (z_pad_before + (z % z_size)) * z_pitch +
 (y_pad_before + (y % y_size)) * y_pitch +
 (x_pad_before + (x % x_size)) * x_pitch +
 (bsv * alignmentF)
 + fsv;
 return output_offset;
}
inline uint get_bs_fs_zyx_bsv_fsv_index(uint b, uint f, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint f_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint z_pad_before, uint z_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after,
 uint b_alignment, uint f_alignment) {
 const uint feature = f + f_pad_before;
 const uint fs = feature / f_alignment;
 const uint fsv = feature % f_alignment;
 const uint bs = (b + b_pad_before) / b_alignment;
 const uint bsv = (b + b_pad_before) % b_alignment;
 const uint bsv_pitch = f_alignment;
 const uint x_pitch = bsv_pitch * b_alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint bs_pitch = fs_pitch * ((total_f_size + f_alignment - 1) / f_alignment);
 const uint output_offset = bs * bs_pitch +
 fs * fs_pitch +
 (z_pad_before + z) * z_pitch +
 (y_pad_before + y) * y_pitch +
 (x_pad_before + x) * x_pitch +
 bsv * bsv_pitch
 + fsv;
 return output_offset;
}
# 366 "./reorder_weights_7_weight_0_0.cl"
inline uint get_os_is_zyx_isv_osv_index(uint o, uint i, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
 const uint isv = i % isv_size;
 const uint osv = o % osv_size;
 const uint is = i / isv_size;
 const uint os = o / osv_size;
 const uint x_pitch = osv_size * isv_size;
 const uint y_pitch = x_pitch * x_size;
 const uint z_pitch = y_pitch * y_size;
 const uint is_pitch = z_pitch * z_size;
 const uint os_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);
 const uint output_offset =
 osv +
 isv * osv_size +
 x * x_pitch +
 y * y_pitch +
 z * z_pitch +
 is * is_pitch +
 os * os_pitch;
 return output_offset;
}
inline uint get_is_os_zyx_isv_osv_index(uint o, uint i, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
 const uint isv = i % isv_size;
 const uint osv = o % osv_size;
 const uint is = i / isv_size;
 const uint os = o / osv_size;
 const uint x_pitch = osv_size * isv_size;
 const uint y_pitch = x_pitch * x_size;
 const uint z_pitch = y_pitch * y_size;
 const uint os_pitch = z_pitch * z_size;
 const uint is_pitch = os_pitch * ((o_size + osv_size - 1) / osv_size);
 const uint output_offset =
 osv +
 isv * osv_size +
 x * x_pitch +
 y * y_pitch +
 z * z_pitch +
 os * os_pitch +
 is * is_pitch;
 return output_offset;
}
inline uint get_os_is_zyx_osv_isv_index(uint o, uint i, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
 const uint isv = i % isv_size;
 const uint osv = o % osv_size;
 const uint is = i / isv_size;
 const uint os = o / osv_size;
 const uint x_pitch = osv_size * isv_size;
 const uint y_pitch = x_pitch * x_size;
 const uint z_pitch = y_pitch * y_size;
 const uint is_pitch = z_pitch * z_size;
 const uint os_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);
 const uint output_offset =
 isv +
 osv * isv_size +
 x * x_pitch +
 y * y_pitch +
 z * z_pitch +
 is * is_pitch +
 os * os_pitch;
 return output_offset;
}
inline uint get_g_os_is_zyx_osv_isv_index(uint g, uint o, uint i, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
 const uint isv = i % isv_size;
 const uint osv = o % osv_size;
 const uint is = i / isv_size;
 const uint os = o / osv_size;
 const uint x_pitch = osv_size * isv_size;
 const uint y_pitch = x_pitch * x_size;
 const uint z_pitch = y_pitch * y_size;
 const uint is_pitch = z_pitch * z_size;
 const uint os_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);
 const uint g_pitch = os_pitch * ((o_size + osv_size - 1) / osv_size);
 const uint output_offset =
 isv +
 osv * isv_size +
 x * x_pitch +
 y * y_pitch +
 z * z_pitch +
 is * is_pitch +
 os * os_pitch +
 g * g_pitch;
 return output_offset;
}







inline uint get_os_is_zyx_isv8_osv16_isv2_index(uint g, uint o, uint i, uint z, uint y, uint x, uint x_size, uint y_size, uint z_size,
 uint g_size, uint o_size, uint i_size, uint offset)
{
 const uint group_offset = g * o_size * i_size * z_size * y_size * x_size;
 const uint xyz_offset = (x + y * x_size + z * x_size * y_size)* 8*16*2;
 const uint i2_val = i % 2;
 const uint i2_slice = i / 2;
 const uint i8_v = i2_slice % 8;
 const uint i8_s = i2_slice / 8;
 const uint i2_offset = i2_val;
 const uint o_offset = (o % 16)*2 + (o / 16) * 16 * i_size * x_size * y_size * z_size;
 const uint i8_offset = 8*16*2* x_size*y_size*z_size * i8_s + 16*2*i8_v;
 const size_t idx = offset + group_offset + xyz_offset + i2_offset + i8_offset + o_offset;
 return idx;
}
inline uint get_os_zyxi_osv16_index(uint o, uint i, uint z, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size, uint z_size)
{
 const size_t idx = o%16 + (o / 16)*i_size*x_size*y_size*z_size*16 +
 16 *(i+ x*i_size + y*i_size*x_size + z*i_size*x_size*y_size);
 return idx;
}
# 501 "./reorder_weights_7_weight_0_0.cl"
inline uint get_gi_yxs_os_yxsv2_osv_index(uint g, uint o, uint i, uint y, uint x, uint x_size, uint g_pitch, uint i_pitch,
 uint y_pitch, uint x_pitch, uint offset, uint sub_group_size)
{
 const uint aligned_ofm_line = x_pitch;
 const uint ifm_height_pitch = (i_pitch/aligned_ofm_line);
 const uint dst_height = i*ifm_height_pitch + y*x_size + x;
 const uint base_filter_index = y*x_size + x;
 const uint aligned_height = dst_height & 0xfffffffe;
 const uint base_filter_odd = (base_filter_index & 0x1);
 uint slice_id = o / sub_group_size;
 uint id_in_slice = o % sub_group_size;
 uint slice_pitch = 2*sub_group_size;
 uint offset_in_slice = (int)(sub_group_size*base_filter_odd);
 const uint in_line = (slice_pitch*slice_id + offset_in_slice + id_in_slice);
 size_t idx = offset + aligned_height*aligned_ofm_line + in_line;
 idx += g * g_pitch;
 return idx;
}

inline uint get_giy_xs_os_xsv2_osv_index(uint g, uint o, uint i, uint y, uint x, uint x_size, uint g_pitch,
 uint i_pitch, uint y_pitch, uint x_pitch, uint offset, uint sub_group_size)
{
 const uint aligned_ofm_line = x_pitch;
 const uint ifm_height_pitch = (i_pitch/aligned_ofm_line);
 const uint aligned_x_line = y_pitch / x_pitch;
 const uint dst_height = i*ifm_height_pitch + y*aligned_x_line + x;
 const uint base_filter_index = x;
 const uint aligned_height = dst_height & 0xfffffffe;
 const uint base_filter_odd = (base_filter_index & 0x1);
 uint slice_id = o / sub_group_size;
 uint id_in_slice = o % sub_group_size;
 uint slice_pitch = 2*sub_group_size;
 uint offset_in_slice = (int)(sub_group_size*base_filter_odd);
 const bool last_line_in_base_filter = (x == (x_size - 1));
 if (last_line_in_base_filter && base_filter_odd == 0)
 {
 const uint element_in_slice = 32;
 slice_id = o / element_in_slice;
 id_in_slice = o % element_in_slice;
 slice_pitch = 2*element_in_slice;
 offset_in_slice = 0;
 }
 const uint in_line = (slice_pitch*slice_id + offset_in_slice + id_in_slice);
 size_t idx = offset + aligned_height*aligned_ofm_line + in_line;
 idx += g * g_pitch;
 return idx;
}

inline uint get_is_os_zyx_isa8_osv8_isv2_index(uint o, uint i, uint z, uint y, uint x, uint size_x,
 uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv2_idx = i % 2;
 const uint osv_idx = o % 8;
 const uint isv1_idx = (i / 2) % 8;
 const uint is_idx = i / 16;
 const uint os_idx = o / 8;
 const uint of_8_aligned = ((size_ofm + 7) / 8);
 size_t idx = offset +
 isv2_idx +
 osv_idx * 2 +
 isv1_idx * 8 * 2 +
 x * 8 * 8 * 2 +
 y * size_x * 8 * 8 * 2 +
 z * size_y * size_x * 8 * 8 * 2 +
 os_idx * size_z * size_y * size_x * 8 * 8 * 2 +
 is_idx * of_8_aligned * size_z * size_y * size_x * 8 * 8 * 2;
 return idx;
}
inline uint get_g_os_is_zyx_isa_osv_isv_index(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset,
 uint isa, uint osv, uint isv)
{
 const uint isv2_idx = i % isv;
 const uint osv_idx = o % osv;
 const uint isv1_idx = (i / isv) % isa;
 const uint is_idx = i / (isa * isv);
 const uint os_idx = o / osv;
 const uint if_aligned = ((size_ifm + (isa * isv) - 1) / (isa * isv));
 const uint of_aligned = ((size_ofm + (osv - 1)) / osv);
 size_t idx = offset +
 isv2_idx +
 osv_idx * isv +
 isv1_idx * osv * isv +
 x * isa * osv * isv +
 y * size_x * isa * osv * isv +
 z * size_y * size_x * isa * osv * isv +
 is_idx * size_z * size_y * size_x * isa * osv * isv +
 os_idx * if_aligned * size_z * size_y * size_x * isa * osv * isv +
 g * of_aligned * if_aligned * size_z * size_y * size_x * isa * osv * isv;
 return idx;
}

inline uint get_g_os_is_yx_isa8_osv8_isv4_index(uint g, uint o, uint i, uint y, uint x, uint size_x,
 uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv2_idx = i % 4;
 const uint osv_idx = o % 8;
 const uint isv1_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint os_idx = o / 8;
 const uint if_32_aligned = ((size_ifm + 31) / 32);
 const uint of_8_aligned = ((size_ofm + 7) / 8);
 size_t idx = offset +
 isv2_idx +
 osv_idx * 4 +
 isv1_idx * 8 * 4 +
 x * 8 * 8 * 4 +
 y * size_x * 8 * 8 * 4 +
 is_idx * size_y * size_x * 4 * 8 * 8 +
 os_idx * if_32_aligned * size_y * size_x * 4 * 8 * 8 +
 g * of_8_aligned * if_32_aligned * size_y * size_x * 4 * 8 * 8;
 return idx;
}

inline uint get_is_os_yx_isa8_osv8_isv2_index(uint o, uint i, uint y, uint x, uint size_x,
 uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv2_idx = i % 2;
 const uint osv_idx = o % 8;
 const uint isv1_idx = (i / 2) % 8;
 const uint is_idx = i / 16;
 const uint os_idx = o / 8;
 const uint of_8_aligned = ((size_ofm + 7) / 8);
 size_t idx = offset +
 isv2_idx +
 osv_idx * 2 +
 isv1_idx * 8 * 2 +
 x * 8 * 8 * 2 +
 y * size_x * 8 * 8 * 2 +
 os_idx * size_y * size_x * 2 * 8 * 8 +
 is_idx * of_8_aligned * size_y * size_x * 2 * 8 * 8;
 return idx;
}
inline uint get_is_os_yx_isa8_osv8_isv4_index(uint o, uint i, uint y, uint x, uint size_x,
 uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv2_idx = i % 4;
 const uint osv_idx = o % 8;
 const uint isv1_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint os_idx = o / 8;
 const uint of_8_aligned = ((size_ofm + 7) / 8);
 size_t idx = offset +
 isv2_idx +
 osv_idx * 4 +
 isv1_idx * 8 * 4 +
 x * 8 * 8 * 4 +
 y * size_x * 8 * 8 * 4 +
 os_idx * size_y * size_x * 4 * 8 * 8 +
 is_idx * of_8_aligned * size_y * size_x * 4 * 8 * 8;
 return idx;
}
inline uint get_is_os_yx_osa8_isv16_osv4_index(uint o, uint i, uint y, uint x, uint size_x,
 uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint osv2_idx = o % 4;
 const uint isv_idx = i % 16;
 const uint osv1_idx = (o / 4) % 8;
 const uint os_idx = o / 32;
 const uint is_idx = i / 16;
 const uint of_32_aligned = ((size_ofm + 31) / 32);
 size_t idx = offset +
 osv2_idx +
 isv_idx * 4 +
 osv1_idx * 16 * 4 +
 x * 8 * 16 * 4 +
 y * size_x * 8 * 16 * 4 +
 os_idx * size_y * size_x * 4 * 16 * 8 +
 is_idx * of_32_aligned * size_y * size_x * 4 * 16 * 8;
 return idx;
}
inline uint get_os_is_zyx_isa8_osv8_isv4_index(uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z,
 uint size_ifm, uint size_ofm, uint offset)
{
 const uint ifm_slices = (size_ifm + 31)/32;
 const uint isv2_idx = i % 4;
 const uint osv_idx = o % 8;
 const uint isv1_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint os_idx = o / 8;
 size_t idx = offset + isv2_idx + 4 * (osv_idx + 8 * isv1_idx);
 idx += x * 4 * 8 * 8;
 idx += y * size_x * 4 * 8 * 8;
 idx += z * size_y * size_x * 4 * 8 * 8;
 idx += is_idx * size_z * size_y * size_x * 4 * 8 * 8;
 idx += os_idx * ifm_slices * size_z * size_y * size_x * 4 * 8 * 8;
 return idx;
}

inline uint get_os_is_yx_isa8_osv16_isv4_index(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint f_32_aligned = ((size_ifm + 31)/32) * 32;
 const uint isv2_idx = i % 4;
 const uint osv_idx = o % 16;
 const uint isv1_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint os_idx = o / 16;
 size_t idx = offset + isv2_idx + 4 * (osv_idx + 16 * isv1_idx);
 idx += x * 4 * 8 * 16;
 idx += y * size_x * 4 * 8 * 16;
 idx += is_idx * size_y * size_x * 4 * 8 * 16;
 idx += os_idx * (f_32_aligned/32) * size_y * size_x * 4 * 8 * 16;
 return idx;
}

inline uint get_os_is_zyx_isa8_osv16_isv4_index(uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z,
 uint size_ifm, uint size_ofm, uint offset)
{
 const uint ifm_slices = (size_ifm + 31)/32;
 const uint isv2_idx = i % 4;
 const uint osv_idx = o % 16;
 const uint isv1_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint os_idx = o / 16;
 size_t idx = offset + isv2_idx + 4 * (osv_idx + 16 * isv1_idx);
 idx += x * 4 * 8 * 16;
 idx += y * size_x * 4 * 8 * 16;
 idx += z * size_y * size_x * 4 * 8 * 16;
 idx += is_idx * size_z * size_y * size_x * 4 * 8 * 16;
 idx += os_idx * ifm_slices * size_z * size_y * size_x * 4 * 8 * 16;
 return idx;
}

inline uint get_os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4_index(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
 const uint isv_idx = i % 4;
 const uint isa_idx = (i / 4) % 8;
 const uint is_idx = (i / 32);
 const uint osv_idx = o_swizzled % 8;
 const uint osa_idx = (o_swizzled / 8) % 4;
 const uint os_idx = (o / 32);
 const uint f_32_aligned = ((size_ifm + 31)/32);
 size_t idx = offset +
 isv_idx +
 osv_idx * 4 +
 isa_idx * 8 * 4 +
 osa_idx * 8 * 32 +
 x * 32 * 32 +
 y * size_x * 32 * 32 +
 is_idx * 32 * 32 * size_x * size_y +
 os_idx * 32 * 32 * f_32_aligned * size_x * size_y;
 return idx;
}
inline uint get_os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4_index(uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z,
 uint size_ifm, uint size_ofm, uint offset)
{
 const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
 const uint isv_idx = i % 4;
 const uint isa_idx = (i / 4) % 8;
 const uint is_idx = (i / 32);
 const uint osv_idx = o_swizzled % 8;
 const uint osa_idx = (o_swizzled / 8) % 4;
 const uint os_idx = (o / 32);
 const uint f_32_aligned = ((size_ifm + 31)/32);
 size_t idx = offset +
 isv_idx +
 osv_idx * 4 +
 isa_idx * 8 * 4 +
 osa_idx * 8 * 32 +
 x * 32 * 32 +
 y * size_x * 32 * 32 +
 z * size_x * size_y * 32 * 32 +
 is_idx * 32 * 32 * size_x * size_y * size_z +
 os_idx * 32 * 32 * f_32_aligned * size_x * size_y * size_z;
 return idx;
}
inline uint get_g_is_os_yx_osa4_isa8_osv8_isv4(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv_idx = i % 4;
 const uint isa_idx = (i / 4) % 8;
 const uint is_idx = (i / 32);
 const uint osv_idx = o % 8;
 const uint osa_idx = (o / 8) % 4;
 const uint os_idx = (o / 32);
 const uint ifm_32_aligned = ((size_ifm + 31) / 32);
 const uint ofm_32_aligned = ((size_ofm + 31) / 32);
 size_t idx = offset +
 isv_idx +
 osv_idx * 4 +
 isa_idx * 8 * 4 +
 osa_idx * 8 * 32 +
 x * 32 * 32 +
 y * size_x * 32 * 32 +
 z * size_y * size_x * 32 * 32 +
 os_idx * 32 * 32 * size_x * size_y * size_z +
 is_idx * 32 * 32 * ofm_32_aligned * size_x * size_y * size_z +
 g * 32 * 32 * ifm_32_aligned * ofm_32_aligned * size_x * size_y * size_z;
 return idx;
}
inline uint get_g_os_is_yx_osa4_isa8_osv8_isv4(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv_idx = i % 4;
 const uint isa_idx = (i / 4) % 8;
 const uint is_idx = (i / 32);
 const uint osv_idx = o % 8;
 const uint osa_idx = (o / 8) % 4;
 const uint os_idx = (o / 32);
 const uint ifm_32_aligned = ((size_ifm + 31)/32);
 const uint ofm_32_aligned = ((size_ofm + 31)/32);
 size_t idx = offset +
 isv_idx +
 osv_idx * 4 +
 isa_idx * 8 * 4 +
 osa_idx * 8 * 32 +
 x * 32 * 32 +
 y * size_x * 32 * 32 +
 z * size_y * size_x * 32 * 32 +
 is_idx * 32 * 32 * size_x * size_y * size_z +
 os_idx * 32 * 32 * ifm_32_aligned * size_x * size_y * size_z +
 g * 32 * 32 * ifm_32_aligned * ofm_32_aligned * size_x * size_y * size_z;
 return idx;
}
inline uint get_g_os_is_yx_osa4_isa8_osv8_isv2(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv_idx = i % 2;
 const uint isa_idx = (i / 2) % 8;
 const uint is_idx = (i / 16);
 const uint osv_idx = o % 8;
 const uint osa_idx = (o / 8) % 4;
 const uint os_idx = (o / 32);
 const uint ifm_16_aligned = ((size_ifm + 15)/16);
 const uint ofm_32_aligned = ((size_ofm + 31)/32);
 size_t idx = offset +
 isv_idx +
 osv_idx * 2 +
 isa_idx * 8 * 2 +
 osa_idx * 8 * 16 +
 x * 32 * 16 +
 y * size_x * 32 * 16 +
 z * size_y * size_x * 32 * 16 +
 is_idx * 32 * 16 * size_x * size_y * size_z +
 os_idx * 32 * 16 * ifm_16_aligned * size_x * size_y * size_z +
 g * 32 * 16 * ifm_16_aligned * ofm_32_aligned * size_x * size_y * size_z;
 return idx;
}
inline uint get_g_os_is_yx_osa2_isa8_osv8_isv2(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv_idx = i % 2;
 const uint isa_idx = (i / 2) % 8;
 const uint is_idx = (i / 16);
 const uint osv_idx = o % 8;
 const uint osa_idx = (o / 8) % 2;
 const uint os_idx = (o / 16);
 const uint ifm_16_aligned = ((size_ifm + 15)/16);
 const uint ofm_16_aligned = ((size_ofm + 15)/16);
 size_t idx = offset +
 isv_idx +
 osv_idx * 2 +
 isa_idx * 8 * 2 +
 osa_idx * 8 * 16 +
 x * 16 * 16 +
 y * size_x * 16 * 16 +
 z * size_y * size_x * 16 * 16 +
 is_idx * 16 * 16 * size_x * size_y * size_z +
 os_idx * 16 * 16 * ifm_16_aligned * size_x * size_y * size_z +
 g * 16 * 16 * ifm_16_aligned * ofm_16_aligned * size_x * size_y * size_z;
 return idx;
}
inline uint get_g_is_os_yx_isa2_osa8_isv8_osv2(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 return get_g_os_is_yx_osa2_isa8_osv8_isv2(g, i, o, z, y, x, size_x, size_y, size_z, size_ofm, size_ifm, offset);
}
inline uint get_g_is_os_yx_isa4_osa8_isv8_osv4(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 return get_g_os_is_yx_osa4_isa8_osv8_isv4(g, i, o, z, y, x, size_x, size_y, size_z, size_ofm, size_ifm, offset);
}




inline uint get_is_o32_yx_isv32_swizzled_by_4_index(uint o, uint i, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size)
{
 const uint o_aligned_to_32 = ((o_size + 31) / 32) * 32;
 const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
 const uint i_aligned_to_32 = ((i_size + 31) / 32) * 32;
 const uint i_val = i % 32;
 const uint i_slice = i / 32;
 const size_t idx = i_val + 32* (x + x_size * (y + y_size * (o_swizzled + o_aligned_to_32 * i_slice) ) );
 return idx;
}

inline uint get_g_os_is_yx_osv_isv(uint g, uint o, uint i, uint y, uint x,
 uint i_size,
 uint o_size,
 uint x_size,
 uint y_size,
 uint osv_size,
 uint isv_size)
{
 return get_g_os_is_zyx_osv_isv_index(g, o, i, 0, y, x,
 x_size, y_size, 1, i_size, o_size, osv_size, isv_size);
}





inline uint get_os_is_yx_osv32_isv4_swizzled_by_2(uint o, uint i, uint y, uint x,
 uint o_size,
 uint i_size,
 uint y_size,
 uint x_size)
{
 const uint osv = 32;
 const uint os = o / osv;
 const uint ofm_block = (o % osv) % 2;
 const uint ofm_in_block = (o % osv) / 2;
 const uint tile = 4;
 const uint ifm_aligned = ((i_size + tile - 1) / tile) * tile;
 const uint ifm_tile = i / tile;
 const uint id = i - ifm_tile * tile;
 uint idx = os * ifm_aligned * y_size * x_size * osv
 + ifm_tile * y_size * x_size * osv * tile
 + y * x_size * osv * tile
 + x * osv * tile
 + ofm_block * 16 * tile
 + ofm_in_block * tile
 + id;
 return idx;
}
inline uint get_os_is_osv32_isv32_swizzled_by_4_index(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint size_ifm_a = ((size_ifm + 31)/32) * 32;
 const uint o_hi = o / 32;
 const uint o_lo = o % 32;
 const uint i_hi = i / 32;
 const uint i_lo = i % 32;
 const uint o_lo1 = o_lo % 4;
 const uint o_lo2 = (o_lo / 4) % 8;
 const uint i_lo1 = i_lo % 4;
 const uint i_lo2 = i_lo / 4;
 const uint idx_in_group = o_lo2 * 4 + o_lo1 * (32 * 8) + i_lo2 * 32 + i_lo1;
 const uint group_idx = o_hi * (size_ifm_a / 32) + i_hi;
 return group_idx * (32 * 32) + idx_in_group;
}
inline uint get_os_i_yxs_osv_yxsv4_index(uint o, uint i, uint y, uint x, uint i_size, uint size_x, uint size_y, uint osv) {
 const uint yxsv = 4;
 uint yx = y * size_x + x;
 uint yx_size_aligned = (size_x * size_y + yxsv - 1) / yxsv * yxsv;
 uint os_index = o / osv;
 uint yxs_index = yx / yxsv;
 uint osv_index = o % osv;
 uint yxsv_index = yx % yxsv;
 uint index = 0;
 index += yxsv_index;
 index += osv_index * yxsv;
 index += yxs_index * yxsv * osv;
 index += i * osv * yx_size_aligned;
 index += os_index * osv * yx_size_aligned * i_size;
 return index;
}
# 972 "./reorder_weights_7_weight_0_0.cl"
inline uint get_gs_oi_yxs_gsv_yxsv4_index(uint g, uint o, uint i, uint y, uint x, uint o_size, uint i_size, uint size_x, uint size_y, const uint gsv) {
 const uint yxsv = 4;
 uint yx = y * size_x + x;
 uint yx_size_aligned = (size_x * size_y + yxsv - 1) / yxsv * yxsv;
 uint gs_index = g / gsv;
 uint yxs_index = yx / yxsv;
 uint gsv_index = g % gsv;
 uint yxsv_index = yx % yxsv;
 uint index = 0;
 index += yxsv_index;
 index += gsv_index * yxsv;
 index += yxs_index * yxsv * gsv;
 index += o * i * gsv * yx_size_aligned;
 index += gs_index * gsv * yx_size_aligned * o_size * i_size;
 return index;
}




inline uint get_g_os_zyx_is_osv_isv_index(uint g, uint o, uint i, uint z, uint y, uint x,
 uint g_size, uint o_size, uint i_size, uint z_size, uint y_size, uint x_size,
 uint osv, uint isv) {
 uint is_size = (i_size + isv - 1) / isv;
 uint os_size = (o_size + osv - 1) / osv;
 uint isv_index = i % isv;
 uint osv_index = o % osv;
 uint is_index = i / isv;
 uint os_index = o / osv;
 uint isv_pitch = 1;
 uint osv_pitch = isv_pitch * isv;
 uint is_pitch = osv_pitch * osv;
 uint x_pitch = is_pitch * is_size;
 uint y_pitch = x_pitch * x_size;
 uint z_pitch = y_pitch * y_size;
 uint os_pitch = z_pitch * z_size;
 uint g_pitch = os_pitch * os_size;
 uint index = 0;
 index += isv_index * isv_pitch;
 index += osv_index * osv_pitch;
 index += is_index * is_pitch;
 index += x * x_pitch;
 index += y * y_pitch;
 index += z * z_pitch;
 index += os_index * os_pitch;
 index += g * g_pitch;
 return index;
}







inline uint get_g_os_y_is_x_osv_isv_index(uint g, uint o, uint i, uint y, uint x,
 uint x_size, uint y_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
 const uint isv = i % isv_size;
 const uint osv = o % osv_size;
 const uint is = i / isv_size;
 const uint os = o / osv_size;
 const uint x_pitch = osv_size * isv_size;
 const uint is_pitch = x_pitch * x_size;
 const uint y_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);
 const uint os_pitch = y_pitch * y_size;
 const uint g_pitch = os_pitch * ((o_size + osv_size - 1) / osv_size);
 const uint output_offset =
 isv +
 osv * isv_size +
 x * x_pitch +
 is * is_pitch +
 y * y_pitch +
 os * os_pitch +
 g * g_pitch;
 return output_offset;
}

inline uint get_g_os_zy_is_x_osv_isv_index(uint g, uint o, uint i, uint z, uint y, uint x,
 uint o_size, uint i_size, uint z_size, uint y_size, uint x_size,
 uint osv, uint isv) {
 uint is_size = (i_size + isv - 1) / isv;
 uint os_size = (o_size + osv - 1) / osv;
 uint isv_index = i % isv;
 uint osv_index = o % osv;
 uint is_index = i / isv;
 uint os_index = o / osv;
 uint isv_pitch = 1;
 uint osv_pitch = isv_pitch * isv;
 uint x_pitch = osv_pitch * osv;
 uint is_pitch = x_pitch * x_size;
 uint y_pitch = is_pitch * is_size;
 uint z_pitch = y_pitch * y_size;
 uint os_pitch = z_pitch * z_size;
 uint g_pitch = os_pitch * os_size;
 uint index = 0;
 index += isv_index * isv_pitch;
 index += osv_index * osv_pitch;
 index += is_index * is_pitch;
 index += x * x_pitch;
 index += y * y_pitch;
 index += z * z_pitch;
 index += os_index * os_pitch;
 index += g * g_pitch;
 return index;
}


inline int imad_SW(int acc, uchar4 input, char4 weight) __attribute__((overloadable)) {
 acc += input[0] * weight[0];
 acc += input[1] * weight[1];
 acc += input[2] * weight[2];
 acc += input[3] * weight[3];
 return acc;
}
inline int imad_SW(int acc, char4 input, char4 weight) __attribute__((overloadable)) {
 acc += input[0] * weight[0];
 acc += input[1] * weight[1];
 acc += input[2] * weight[2];
 acc += input[3] * weight[3];
 return acc;
}
inline int imad_SW(int acc, char4 input, uchar4 weight) __attribute__((overloadable)) {
 acc += input[0] * weight[0];
 acc += input[1] * weight[1];
 acc += input[2] * weight[2];
 acc += input[3] * weight[3];
 return acc;
}
inline int imad_SW(int acc, uchar4 input, uchar4 weight) __attribute__((overloadable)) {
 acc += input[0] * weight[0];
 acc += input[1] * weight[1];
 acc += input[2] * weight[2];
 acc += input[3] * weight[3];
 return acc;
}


typedef struct __attribute__ ((packed)) int4x2_t { char s0; } int4x2_t;
typedef struct __attribute__ ((packed)) int4x4_t { int4x2_t s0; int4x2_t s1; } int4x4_t;
typedef struct __attribute__ ((packed)) int4x8_t { int4x2_t s0; int4x2_t s1; int4x2_t s2; int4x2_t s3; } int4x8_t;
typedef struct __attribute__ ((packed)) int4x16_t { int4x2_t s0; int4x2_t s1; int4x2_t s2; int4x2_t s3; int4x2_t s4; int4x2_t s5; int4x2_t s6; int4x2_t s7; } int4x16_t;
typedef struct __attribute__ ((packed)) uint4x2_t { uchar s0; } uint4x2_t;
typedef struct __attribute__ ((packed)) uint4x4_t { uint4x2_t s0; uint4x2_t s1; } uint4x4_t;
typedef struct __attribute__ ((packed)) uint4x8_t { uint4x2_t s0; uint4x2_t s1; uint4x2_t s2; uint4x2_t s3; } uint4x8_t;
typedef struct __attribute__ ((packed)) uint4x16_t { uint4x2_t s0; uint4x2_t s1; uint4x2_t s2; uint4x2_t s3; uint4x2_t s4; uint4x2_t s5; uint4x2_t s6; uint4x2_t s7; } uint4x16_t;
inline uchar2 cvt_uint4x2_to_uint8x2(uint4x2_t v) __attribute__((overloadable)) {
 const uchar v0 = v.s0 & 0x0F;
 const uchar v1 = (v.s0 & 0xF0) >> 4;
 return (uchar2)(v0, v1);
}
inline char2 cvt_uint4x2_to_int8x2(uint4x2_t v) __attribute__((overloadable)) {
 const char v0 = convert_char(v.s0 & 0x0F);
 const char v1 = convert_char((v.s0 & 0xF0) >> 4);
 return (char2)(v0, v1);
}
inline char2 cvt_int4x2_to_int8x2(int4x2_t v) __attribute__((overloadable)) {
 const char s_bit = (v.s0 & convert_char(0x08));
 const char mask = s_bit > 0 ? convert_char(0xF0) : convert_char(0x00);
 const char v0 = (v.s0 & convert_char(0x0F)) | mask;
 const char v1 = v.s0 >> 4;
 return (char2)(v0, v1);
}
inline uchar2 unpack_to_uchar(uint4x2_t v) __attribute__((overloadable)) {
 return cvt_uint4x2_to_uint8x2(v);
}
inline char2 unpack_to_char(int4x2_t v) __attribute__((overloadable)) {
 return cvt_int4x2_to_int8x2(v);
}
inline char2 unpack_to_char(uint4x2_t v) __attribute__((overloadable)) {
 return convert_char2(cvt_uint4x2_to_uint8x2(v));
}
inline char4 unpack_to_char(int4x4_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 return (char4)(v0.s0, v0.s1, v1.s0, v1.s1);
}
inline char4 unpack_to_char(uint4x4_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 return (char4)(v0.s0, v0.s1, v1.s0, v1.s1);
}
inline uchar4 unpack_to_uchar(uint4x4_t v) __attribute__((overloadable)) {
 uchar2 v0 = unpack_to_uchar(v.s0);
 uchar2 v1 = unpack_to_uchar(v.s1);
 return (uchar4)(v0.s0, v0.s1, v1.s0, v1.s1);
}
inline char4 unpack_transposed_to_char(int4x4_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 return (char4)(v0.s0, v1.s0, v0.s1, v1.s1);
}
inline char4 unpack_transposed_to_char(uint4x4_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 return (char4)(v0.s0, v1.s0, v0.s1, v1.s1);
}
inline uchar4 unpack_transposed_to_uchar(uint4x4_t v) __attribute__((overloadable)) {
 uchar2 v0 = unpack_to_uchar(v.s0);
 uchar2 v1 = unpack_to_uchar(v.s1);
 return (uchar4)(v0.s0, v1.s0, v0.s1, v1.s1);
}
inline uchar8 unpack_to_uchar(uint4x8_t v) __attribute__((overloadable)) {
 uchar2 v0 = unpack_to_uchar(v.s0);
 uchar2 v1 = unpack_to_uchar(v.s1);
 uchar2 v2 = unpack_to_uchar(v.s2);
 uchar2 v3 = unpack_to_uchar(v.s3);
 return (uchar8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}
inline char8 unpack_to_char(int4x8_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 char2 v2 = unpack_to_char(v.s2);
 char2 v3 = unpack_to_char(v.s3);
 return (char8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}
inline char8 unpack_to_char(uint4x8_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 char2 v2 = unpack_to_char(v.s2);
 char2 v3 = unpack_to_char(v.s3);
 return (char8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}
inline char8 unpack_transposed_to_char(int4x8_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 char2 v2 = unpack_to_char(v.s2);
 char2 v3 = unpack_to_char(v.s3);
 return (char8)(v0.s0, v1.s0, v2.s0, v3.s0, v0.s1, v1.s1, v2.s1, v3.s1);
}
inline char8 unpack_transposed_to_char(uint4x8_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 char2 v2 = unpack_to_char(v.s2);
 char2 v3 = unpack_to_char(v.s3);
 return (char8)(v0.s0, v1.s0, v2.s0, v3.s0, v0.s1, v1.s1, v2.s1, v3.s1);
}
inline uchar8 unpack_transposed_to_uchar(uint4x8_t v) __attribute__((overloadable)) {
 uchar2 v0 = unpack_to_uchar(v.s0);
 uchar2 v1 = unpack_to_uchar(v.s1);
 uchar2 v2 = unpack_to_uchar(v.s2);
 uchar2 v3 = unpack_to_uchar(v.s3);
 return (uchar8)(v0.s0, v1.s0, v2.s0, v3.s0, v0.s1, v1.s1, v2.s1, v3.s1);
}
inline float2 unpack_to_float(uint4x2_t v) __attribute__((overloadable)) {
 return convert_float2(cvt_uint4x2_to_uint8x2(v));
}
inline float2 unpack_to_float(int4x2_t v) __attribute__((overloadable)) {
 return convert_float2(cvt_int4x2_to_int8x2(v));
}
inline float4 unpack_to_float(uint4x4_t v) __attribute__((overloadable)) {
 float2 f0 = unpack_to_float(v.s0);
 float2 f1 = unpack_to_float(v.s1);
 return (float4)(f0.s0, f0.s1, f1.s0, f1.s1);
}
inline float4 unpack_to_float(int4x4_t v) __attribute__((overloadable)) {
 float2 f0 = unpack_to_float(v.s0);
 float2 f1 = unpack_to_float(v.s1);
 return (float4)(f0.s0, f0.s1, f1.s0, f1.s1);
}
inline float8 unpack_to_float(uint4x8_t v) __attribute__((overloadable)) {
 float2 f0 = unpack_to_float(v.s0);
 float2 f1 = unpack_to_float(v.s1);
 float2 f2 = unpack_to_float(v.s2);
 float2 f3 = unpack_to_float(v.s3);
 return (float8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}
inline float8 unpack_to_float(int4x8_t v) __attribute__((overloadable)) {
 float2 f0 = unpack_to_float(v.s0);
 float2 f1 = unpack_to_float(v.s1);
 float2 f2 = unpack_to_float(v.s2);
 float2 f3 = unpack_to_float(v.s3);
 return (float8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}
# 1519 "./reorder_weights_7_weight_0_0.cl"
typedef struct half5 { half s0; half s1; half s2; half s3; half s4; } half5;
typedef struct half6 { half s0; half s1; half s2; half s3; half s4; half s5; } half6;
typedef struct half7 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; } half7;
typedef struct half9 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; } half9;
typedef struct half10 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; half s9; } half10;
typedef struct half11 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; half s9; half sa; } half11;
typedef struct half12 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; half s9; half sa; half sb;} half12;
typedef struct half13 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; half s9; half sa; half sb; half sc;} half13;
typedef struct half14 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; half s9; half sa; half sb; half sc; half se;} half14;
typedef struct half15 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; half s9; half sa; half sb; half sc; half se; half sf;} half15;
typedef struct half0 { half s0; } half0;
typedef struct float5 { float s0; float s1; float s2; float s3; float s4; } float5;
typedef struct float6 { float s0; float s1; float s2; float s3; float s4; float s5; } float6;
typedef struct float7 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; } float7;
typedef struct float9 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; float s7; float s8; } float9;
typedef struct float10 { float s0; float s1; float s2; float s3; float s4; float s5;
 float s6; float s7; float s8; float s9;} float10;
typedef struct float11 { float s0; float s1; float s2; float s3; float s4; float s5;
 float s6; float s7; float s8; float s9; float sa;} float11;
typedef struct float12 { float s0; float s1; float s2; float s3; float s4; float s5;
 float s6; float s7; float s8; float s9; float sa; float sb; } float12;
typedef struct float13 { float s0; float s1; float s2; float s3; float s4; float s5;
 float s6; float s7; float s8; float s9; float sa; float sb; float sc;} float13;
typedef struct float14 { float s0; float s1; float s2; float s3; float s4; float s5;
 float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; } float14;
typedef struct float15 { float s0; float s1; float s2; float s3; float s4; float s5;
 float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; float se; } float15;
typedef struct float0 { float s0; } float0;
# 1599 "./reorder_weights_7_weight_0_0.cl"
__constant size_t _INPUT0_SIZES_reorder_weights_7_weight_0_0 [] = { 1,1,384,192,1,1,1,1,1, };
# 1649 "./reorder_weights_7_weight_0_0.cl"
__constant size_t _OUTPUT_SIZES_reorder_weights_7_weight_0_0 [] = { 1,1,384,192,1,1,1,1,1, };
# 1698 "./reorder_weights_7_weight_0_0.cl"
inline uint8 _reshape_2_to_4_reorder_weights_7_weight_0_0(uint o, uint i, uint y, uint x, uint dst_size_y, uint dst_size_x)
{
 uint _i = i / (dst_size_y*dst_size_x);
 uint _yx = i % (dst_size_y*dst_size_x);
 uint _y = _yx / dst_size_x;
 uint _x = _yx % dst_size_x;
 return (uint8)(0, o, _i, 0, 0, _y,_x, 0);
}
inline uint8 _reshape_4_to_2_reorder_weights_7_weight_0_0(uint o, uint i, uint y, uint x, uint src_size_y, uint src_size_x)
{
 uint _i = i*src_size_y*src_size_x + y*src_size_x + x;
 return (uint8)(0, o, _i, 0, 0, 0, 0, 0);
}
inline uint8 _reshape_4_to_5_reorder_weights_7_weight_0_0(uint o, uint i, uint y, uint x,
 uint src_size_f, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_z, uint dst_size_y, uint dst_size_x)
{
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_f = src_pitch_y * src_size_y;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x + y * src_pitch_y + i * src_pitch_f + o * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_z = flat_idx % dst_size_z;
 flat_idx /= dst_size_z;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(0, dst_b, dst_f, 0, dst_z, dst_y, dst_x, 0);
}
inline uint8 _reshape_5_to_4_reorder_weights_7_weight_0_0(uint o, uint i, uint z, uint y, uint x,
 uint src_size_f, uint src_size_z, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_y, uint dst_size_x)
{
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_z = src_pitch_y * src_size_y;
 const uint src_pitch_f = src_pitch_z * src_size_z;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x + y * src_pitch_y + z * src_pitch_z + i * src_pitch_f + o * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(0, dst_b, dst_f, 0, 0, dst_y, dst_x, 0);
}
inline uint8 _reshape_4_to_6_reorder_weights_7_weight_0_0(uint o, uint i, uint y, uint x,
 uint src_size_f, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x)
{
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_f = src_pitch_y * src_size_y;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x + y * src_pitch_y + i * src_pitch_f + o * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_z = flat_idx % dst_size_z;
 flat_idx /= dst_size_z;
 uint dst_w = flat_idx % dst_size_w;
 flat_idx /= dst_size_w;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(0, dst_b, dst_f, dst_w, dst_z, dst_y, dst_x, 0);
}
inline uint8 _reshape_6_to_4_reorder_weights_7_weight_0_0(uint o, uint i, uint w, uint z, uint y, uint x,
 uint src_size_f, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_y, uint dst_size_x)
{
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_z = src_pitch_y * src_size_y;
 const uint src_pitch_w = src_pitch_z * src_size_z;
 const uint src_pitch_f = src_pitch_w * src_size_w;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x + y * src_pitch_y + z * src_pitch_z + w * src_pitch_w + i * src_pitch_f + o * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(0, dst_b, dst_f, 0, 0, dst_y, dst_x, 0);
}
inline uint8 _reshape_6_to_5_reorder_weights_7_weight_0_0(uint o, uint i, uint w, uint z, uint y, uint x,
 uint src_size_f, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_z, uint dst_size_y, uint dst_size_x)
{
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_z = src_pitch_y * src_size_y;
 const uint src_pitch_w = src_pitch_z * src_size_z;
 const uint src_pitch_f = src_pitch_w * src_size_w;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x + y * src_pitch_y + z * src_pitch_z + w * src_pitch_w + i * src_pitch_f + o * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_z = flat_idx % dst_size_z;
 flat_idx /= dst_size_z;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(0, dst_b, dst_f, 0, dst_z, dst_y, dst_x, 0);
}
inline uint8 _reshape_grouped_reorder_weights_7_weight_0_0(uint g, uint o, uint i, uint z, uint y, uint x, uint src_size_ofm, uint dst_size_ofm)
{
 const uint flat_ofm = g * src_size_ofm + o;
 const uint dst_ofm = flat_ofm % dst_size_ofm;
 const uint dst_g = flat_ofm / dst_size_ofm;
 const uint dst_ifm = i;
 const uint dst_z = z;
 const uint dst_y = y;
 const uint dst_x = x;
 return (uint8)(dst_g, dst_ofm, dst_ifm, 0, dst_z, dst_y, dst_x, 0);
}
inline uint8 _reshape_weights_reorder_weights_7_weight_0_0(
 uint o, uint i, uint w, uint z, uint y, uint x,
 uint src_size_f, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x,
 uint src_dims, uint dst_dims)
{
 if (src_dims == 4 && dst_dims == 2)
 {
 return _reshape_4_to_2_reorder_weights_7_weight_0_0(o,i,y,x,src_size_y,src_size_x);
 }
 else if (src_dims == 2 && dst_dims == 4)
 {
 return _reshape_2_to_4_reorder_weights_7_weight_0_0(o,i,y,x,dst_size_y,dst_size_x);
 }
 else if (src_dims == 4 && dst_dims == 6)
 {
 return _reshape_4_to_6_reorder_weights_7_weight_0_0(o, i, y, x, src_size_f, src_size_y, src_size_x, dst_size_f, dst_size_w, dst_size_z, dst_size_y, dst_size_x);
 }
 else if (src_dims == 6 && dst_dims == 4)
 {
 return _reshape_6_to_4_reorder_weights_7_weight_0_0(o, i, w, z, y, x, src_size_f, src_size_w, src_size_z, src_size_y, src_size_x, dst_size_f, dst_size_y, dst_size_x);
 }
 else if (src_dims == 4 && dst_dims == 5)
 {
 return _reshape_4_to_5_reorder_weights_7_weight_0_0(o, i, y, x, src_size_f, src_size_y, src_size_x, dst_size_f, dst_size_z, dst_size_y, dst_size_x);
 }
 else if (src_dims == 5 && dst_dims == 4)
 {
 return _reshape_5_to_4_reorder_weights_7_weight_0_0(o, i, z, y, x, src_size_f, src_size_z, src_size_y, src_size_x, dst_size_f, dst_size_y, dst_size_x);
 }
 else if (src_dims == 6 && dst_dims == 5)
 {
 return _reshape_6_to_5_reorder_weights_7_weight_0_0(o, i, w, z, y, x, src_size_f, src_size_w, src_size_z, src_size_y, src_size_x, dst_size_f, dst_size_z, dst_size_y, dst_size_x);
 }
 return (uint8)(0, o, i, w, z, y, x, 0);
}
inline uint8 _reshape_dims_with_groups_reorder_weights_7_weight_0_0(
 uint g, uint o, uint i, uint w, uint z, uint y, uint x,
 uint src_size_ofm, uint src_size_ifm, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
 uint dst_size_ofm, uint dst_size_ifm, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x,
 uint src_dims, uint dst_dims, uint src_size_groups, uint dst_size_groups)
{
 if (src_dims == 5 && dst_dims == 4)
 {
 return _reshape_grouped_reorder_weights_7_weight_0_0(g, o, i, 0, y, x, src_size_ofm, dst_size_ofm);
 }
 else if (src_dims == 6 && dst_dims == 5)
 {
 return _reshape_grouped_reorder_weights_7_weight_0_0(g, o, i, z, y, x, src_size_ofm, dst_size_ofm);
 }
 else if (src_dims == 6 && dst_dims == 4)
 {
 return _reshape_grouped_reorder_weights_7_weight_0_0(g, o, i, 0, y, x, src_size_ofm, dst_size_ofm);
 }
 return (uint8)(g, o, i, w, z, y, x, 0);
}


inline uint _get_input_index_reorder_weights_7_weight_0_0(uint g, uint o, uint i, uint z, uint y, uint x)
{

 return 0 + (x)*1 + (y)*1 + (i)*1 + (o)*384 + (0)*1;
# 1988 "./reorder_weights_7_weight_0_0.cl"
}
inline uint _get_output_index_reorder_weights_7_weight_0_0(uint g, uint o, uint i, uint z, uint y, uint x)
{
//i=343,x=0,y=191,z=0
 return 0 + (x)*1 + (y)*1 + (i)*1 + (o)*384 + (0)*1;
# 2124 "./reorder_weights_7_weight_0_0.cl"
}
# 2140 "./reorder_weights_7_weight_0_0.cl"
__kernel void reorder_weights_7_weight_0_0(const __global float* input, __global half* output)
{




 const unsigned g = 0;
 const unsigned o = (uint)get_global_id(0);

 const unsigned i = (uint)get_global_id(1);





 const unsigned x = (uint)get_global_id(2) % 1;
 const unsigned y = (uint)get_global_id(2) / 1;
 const unsigned z = 0;
# 2167 "./reorder_weights_7_weight_0_0.cl"
 uint8 ir = _reshape_weights_reorder_weights_7_weight_0_0( o, i, 0, z, y, x, 384, 1, 1, 1, 1, 384, 1, 1, 1, 1, 4, 4);

 uint input_idx = _get_input_index_reorder_weights_7_weight_0_0(ir.s0,ir.s1,ir.s2,ir.s4,ir.s5,ir.s6);

 uint output_idx = _get_output_index_reorder_weights_7_weight_0_0(g, o, i, z, y, x);






 output[output_idx] = convert_half(input[input_idx]);


//printf("o=%d,i=%d, output[%d]=%f\n", o, i, output_idx, output[output_idx]);
}
