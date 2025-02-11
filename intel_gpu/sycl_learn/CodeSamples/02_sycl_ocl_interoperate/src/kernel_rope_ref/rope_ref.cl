# 0 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"




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
# 93 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
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
# 145 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
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
# 209 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
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
# 366 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
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
# 501 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
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
# 972 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
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
# 1519 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
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
# 1716 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
__constant size_t _INPUT1_SIZES_rms_gpu_bfyx_opt_16553883649600634465_0_0__sa [] = { 1,896,1,1,1,1,1,1,1, };
# 1817 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"

__kernel void rms_gpu_bfyx_opt_16553883649600634465_0_0__sa(
 __global const int* shape_info,
 const __global half* input,
 const __global half* gamma,
 __global half* output)
{
 const uint data_idx = get_global_id(1);
 const uint in_data_idx = get_global_id(0);
 const uint workers_per_data = get_local_size(0);
 const uint data_size = 896;
 const uint items_num = data_size / workers_per_data;
 const uint leftovers = data_size % workers_per_data;
 const uint data_offset = data_idx * data_size;
 const uint subgroup_offset = get_sub_group_id() * get_sub_group_size() * items_num;
 float data[7];
 float rms = 0.0f;
 __local float slm_buf[1024];
 uint i = 0;
 if (workers_per_data > 16)
 {
 for (; i < items_num - (items_num % 8); i += 8)
 {
 float8 vec_tmp = convert_float8(as_half8(_sub_group_block_read_us8((const __global ushort*)(input) + (data_offset + subgroup_offset + i * get_sub_group_size()))));




 __attribute__((opencl_unroll_hint)) for (int j = 0; j < 8; j++)
 {
 float tmp = vec_tmp[j];
 rms += native_powr(tmp, 2);
 data[i + j] = tmp;
 }

 }
 }
 for (; i < items_num; i++)
 {
 float tmp = convert_float(input[data_offset + subgroup_offset + get_sub_group_local_id() + i * get_sub_group_size()]);
 rms += native_powr(tmp, 2);
 data[i] = tmp;
 }
 if (in_data_idx < leftovers)
 {
 float tmp = convert_float(input[data_offset + workers_per_data * items_num + in_data_idx]);
 rms += native_powr(tmp, 2);
 data[items_num] = tmp;
 }
 rms = sub_group_reduce_add(rms);
 if (get_sub_group_local_id() == 0)
 slm_buf[get_sub_group_id()] = rms;
 barrier(CLK_LOCAL_MEM_FENCE);
 for (uint offset = get_num_sub_groups() / 2; offset > 0; offset /= 2) {
 if (in_data_idx < offset) {
 slm_buf[in_data_idx] += slm_buf[in_data_idx + offset];
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 }
 if (in_data_idx == 0) {
 rms = slm_buf[0] / data_size;
 slm_buf[0] = native_powr(sqrt(rms + convert_float(as_float(0x358637bd))), -1);
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 rms = slm_buf[0];
 i = 0;
 if ((workers_per_data > 16) && ((2 * (1*896*1*1*1*1)) & 0xF == 0))
 {
 for (; i < items_num - (items_num % 8); i += 8)
 {
 float8 vec_gamma = convert_float8(as_half8(_sub_group_block_read_us8((const __global ushort*)(gamma) + (subgroup_offset + i * get_sub_group_size()))));
 half8 vec_tmp;



 __attribute__((opencl_unroll_hint)) for (int j = 0; j < 8; j++)
 vec_tmp[j] = convert_half(rms * data[i + j] * vec_gamma[j]);

 _sub_group_block_write_us8( (__global ushort*)(output) + (data_offset + subgroup_offset + i * get_sub_group_size()), as_ushort8(vec_tmp));
 }
 }
 for (; i < items_num; i++)
 {
 float temp = convert_float(gamma[subgroup_offset + get_sub_group_local_id() + i * get_sub_group_size()]);
 output[data_offset + subgroup_offset + get_sub_group_local_id() + i * get_sub_group_size()] = convert_half(rms * data[i] * temp);
 }
 if (in_data_idx < leftovers)
 {
 float temp = convert_float(gamma[workers_per_data * items_num + in_data_idx]);
 output[data_offset + workers_per_data * items_num + in_data_idx] = convert_half(rms * data[items_num] * temp);
 }
}
# 2947 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
inline uint _get_input_index_rope_ref_11982042700243959200_0_0__sa(__global const int* shape_info, uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {

 return ((1*0) + (64*(shape_info[8])) + ((64*(14 + (shape_info[8] + shape_info[9])))*0) + ((64*(14 + (shape_info[8] + shape_info[9]))*1)*0) + ((64*(14 + (shape_info[8] + shape_info[9]))*1*1*1*1)*0) + ((64*(14 + (shape_info[8] + shape_info[9]))*1*1*1*1*(shape_info[1] + 0))*0)) + (x)*1 + (y)*64 + (f)*(64*(14 + (shape_info[8] + shape_info[9]))*1*1*1*1) + (b)*(64*(14 + (shape_info[8] + shape_info[9]))*1*1*1*1*(shape_info[1] + 0));
# 2961 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
}
inline uint _get_input_index_rope_ref_11982042700243959200_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
 return _get_input_index_rope_ref_11982042700243959200_0_0__sa(shape_info, b, f, 0, 0, w, z, y, x);
}
inline uint _get_output_index_rope_ref_11982042700243959200_0_0__sa(__global const int* shape_info, uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {

 return ((1*0) + (64*0) + ((64*(shape_info[32] + 0))*0) + ((64*(shape_info[32] + 0)*1)*0) + ((64*(shape_info[32] + 0)*1*1*1*1)*0) + ((64*(shape_info[32] + 0)*1*1*1*1*14)*0)) + (x)*1 + (y)*64 + (f)*(64*(shape_info[32] + 0)*1*1*1*1) + (b)*(64*(shape_info[32] + 0)*1*1*1*1*14);
# 2979 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
}
inline uint _get_output_index_rope_ref_11982042700243959200_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
 return _get_output_index_rope_ref_11982042700243959200_0_0__sa(shape_info, b, f, 0, 0, w, z, y, x);
}
# 3060 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
__kernel void rope_ref_11982042700243959200_0_0__sa(
 __global const int* shape_info,
 const __global half* input,
 const __global half* cos,
 const __global half* sin,



 __global half* output)
{
 const uint b = get_global_id(0);
 const uint h = get_global_id(1);
 const uint p = (uint)get_global_id(2) / 32;
 const uint r = (uint)get_global_id(2) % 32;

 uint input_idx = ((1*0) + (64*(shape_info[8])) + ((64*(14 + (shape_info[8] + shape_info[9])))*0) + ((64*(14 + (shape_info[8] + shape_info[9]))*1)*0) + ((64*(14 + (shape_info[8] + shape_info[9]))*1*1*1*1)*0) + ((64*(14 + (shape_info[8] + shape_info[9]))*1*1*1*1*(shape_info[1] + 0))*0)) + (0)*1 + (h)*64 + (p)*(64*(14 + (shape_info[8] + shape_info[9]))*1*1*1*1) + (b)*(64*(14 + (shape_info[8] + shape_info[9]))*1*1*1*1*(shape_info[1] + 0));






 uint cos_sin_b = b < (shape_info[10] ) ? b : 0;
 uint cos_sin_h = h < 1 ? h : 0;
 uint cos_sin_p = p;
# 3097 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 cos_sin_p = cos_sin_p < (shape_info[16] ) ? cos_sin_p : 0;

 uint cos_sin_idx = ((1*0) + (64*0) + ((64*(shape_info[16] + 0))*0) + ((64*(shape_info[16] + 0)*1)*0) + ((64*(shape_info[16] + 0)*1*1*1*1)*0) + ((64*(shape_info[16] + 0)*1*1*1*1*1)*0)) + (0)*1 + (cos_sin_p)*64 + (cos_sin_h)*(64*(shape_info[16] + 0)*1*1*1*1) + (cos_sin_b)*(64*(shape_info[16] + 0)*1*1*1*1*1);
 uint cos_idx = cos_sin_idx;
 uint sin_idx = cos_sin_idx;




 uint output_idx = ((1*0) + (64*0) + ((64*(shape_info[32] + 0))*0) + ((64*(shape_info[32] + 0)*1)*0) + ((64*(shape_info[32] + 0)*1*1*1*1)*0) + ((64*(shape_info[32] + 0)*1*1*1*1*14)*0)) + (0)*1 + (p)*64 + (h)*(64*(shape_info[32] + 0)*1*1*1*1) + (b)*(64*(shape_info[32] + 0)*1*1*1*1*14);
 half in1 = input[input_idx + r];
 half in2 = input[input_idx + 32 + r];
 output[output_idx + r] = cos[cos_idx + r] * in1 - sin[sin_idx + r] * in2;
 output[output_idx + 32 + r] = cos[cos_idx + 32 + r] * in2 +
 sin[sin_idx + 32 + r] * in1;
}
# 4321 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
inline uint _get_input0_index_nt_sdpa_opt_single_token_7001936805389565166_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) {

 return ((1*0) + (64*0) + ((64*(shape_info[6] + 0))*0) + ((64*(shape_info[6] + 0)*1)*0) + ((64*(shape_info[6] + 0)*1*1*1*1)*0) + ((64*(shape_info[6] + 0)*1*1*1*1*14)*0)) + (x)*1 + (y)*64 + (z)*(64*(shape_info[6] + 0)) + (w)*(64*(shape_info[6] + 0)*1) + (f)*(64*(shape_info[6] + 0)*1*1*1*1) + (b)*(64*(shape_info[6] + 0)*1*1*1*1*14);
# 4335 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
}
inline uint _get_input0_index_sdpa_opt_single_token_7001936805389565166_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) {

 return _get_input0_index_nt_sdpa_opt_single_token_7001936805389565166_0_0__sa(shape_info, b,f,w,z,y,x);



}
inline uint _get_input1_index_nt_sdpa_opt_single_token_7001936805389565166_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) {

 f /= 7;;


 return ((1*0) + (64*(shape_info[16])) + ((64*(shape_info[14] + (shape_info[16] + shape_info[17])))*0) + ((64*(shape_info[14] + (shape_info[16] + shape_info[17]))*1)*0) + ((64*(shape_info[14] + (shape_info[16] + shape_info[17]))*1*1*1*1)*0) + ((64*(shape_info[14] + (shape_info[16] + shape_info[17]))*1*1*1*1*2)*0)) + (x)*1 + (y)*64 + (z)*(64*(shape_info[14] + (shape_info[16] + shape_info[17]))) + (w)*(64*(shape_info[14] + (shape_info[16] + shape_info[17]))*1) + (f)*(64*(shape_info[14] + (shape_info[16] + shape_info[17]))*1*1*1*1) + (b)*(64*(shape_info[14] + (shape_info[16] + shape_info[17]))*1*1*1*1*2);
# 4360 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
}
inline uint _get_input1_index_sdpa_opt_single_token_7001936805389565166_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) {

 return _get_input1_index_nt_sdpa_opt_single_token_7001936805389565166_0_0__sa(shape_info, b,f,w,z,y,x);



}
inline uint _get_input2_index_nt_sdpa_opt_single_token_7001936805389565166_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) {

 f /= 7;;


 return ((1*0) + (64*(shape_info[26])) + ((64*(shape_info[24] + (shape_info[26] + shape_info[27])))*0) + ((64*(shape_info[24] + (shape_info[26] + shape_info[27]))*1)*0) + ((64*(shape_info[24] + (shape_info[26] + shape_info[27]))*1*1*1*1)*0) + ((64*(shape_info[24] + (shape_info[26] + shape_info[27]))*1*1*1*1*2)*0)) + (x % 64)*1 + (y % (shape_info[24] ))*64 + (z % 1)*(64*(shape_info[24] + (shape_info[26] + shape_info[27]))) + (w % 1)*(64*(shape_info[24] + (shape_info[26] + shape_info[27]))*1) + (f % 2)*(64*(shape_info[24] + (shape_info[26] + shape_info[27]))*1*1*1*1) + (b % (shape_info[18] ))*(64*(shape_info[24] + (shape_info[26] + shape_info[27]))*1*1*1*1*2);
# 4385 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
}
inline uint _get_input2_index_sdpa_opt_single_token_7001936805389565166_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) {

 return _get_input2_index_nt_sdpa_opt_single_token_7001936805389565166_0_0__sa(shape_info, b,f,w,z,y,x);



}
# 4421 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"

__attribute__((reqd_work_group_size(1, 1, 64 * 2)))
__kernel void sdpa_opt_single_token_7001936805389565166_0_0__sa(
 __global const int* shape_info,
 const __global half* query_input,
 const __global half* key_input,
 const __global half* value_input,

 const __global half* attn_mask,




 __global half* output,







 __global float* exp_sums,
 __global float* max_logits,
 __global half* tmp_out
)
{
 const uint batch_idx = get_global_id(0);
 const uint b0_idx = batch_idx / 14;
 const uint b1_idx = batch_idx % 14;
 const uint target_seq_idx = get_global_id(1);
 const uint lid = get_local_id(2);

 const uint head_size_idx = lid % 64;
# 4462 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 const uint sgid = get_sub_group_id();
 const uint sglid = get_sub_group_local_id();
 const uint partition_idx = get_group_id(2);
 const uint num_of_partitions = get_num_groups(2);
 const uint wi_num_per_partition = get_local_size(2);
 const uint start_partition_idx = partition_idx * 256;
 const uint partition_seq_len =
 ((partition_idx + 1) < num_of_partitions) ? (256)
 : ((shape_info[14] ) - partition_idx * 256);
 __local half query_local[64 * 1];
 __local float qk_local[256 * 1];
 __local float qk_max_vals[(64 * 2 / 16) * 1];
 __local float qk_sum_vals[(64 * 2 / 16) * 1];
 {
 float qk_max[1] = {-FLT_MAX};
 for (uint i = 0; i < 1; i++) {
 qk_max[i] = -FLT_MAX;
 }
 {



 const half scale_val = 1.0h / sqrt(convert_half(64));

 {

 uint query_local_offset = sgid * 16 + sglid;
 const uint seq_idx_end = 1;

 uint query_offset = _get_input0_index_sdpa_opt_single_token_7001936805389565166_0_0__sa(shape_info, b0_idx, b1_idx, 0, 0, target_seq_idx, (sgid * 16));
 uint query_offset_next_seq = _get_input0_index_sdpa_opt_single_token_7001936805389565166_0_0__sa(shape_info, b0_idx, b1_idx, 0, 0, target_seq_idx + 1, (sgid * 16));
 const uint query_pitch = query_offset_next_seq - query_offset;





 if (sgid < 64 / 16) {



 for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {

 half val = as_half(_sub_group_block_read_us((const __global ushort*)(query_input) + (query_offset)));
 query_local[query_local_offset] = val * scale_val;
 query_local_offset += 16 * (64 * 2 / 16);
 query_offset += query_pitch;
 }
 }


 barrier(CLK_LOCAL_MEM_FENCE);
 }
 for (uint seq_len = sgid; seq_len < partition_seq_len; seq_len += (64 / 16) * 2) {



 const uint b_idx = b0_idx;


 uint key_offset = _get_input1_index_sdpa_opt_single_token_7001936805389565166_0_0__sa(shape_info, b_idx, b1_idx, 0, 0, start_partition_idx + seq_len, 0);



 float acc[1] = {0.0f};







 uint head_idx_index = 0;

 for (; head_idx_index + (8 * 16) <= 64; head_idx_index += 16 * 8) {





 half8 key_vec_packed = as_half8(_sub_group_block_read_us8((const __global ushort*)(key_input) + (key_offset + head_idx_index)));;





 half8 key_vals = key_vec_packed;

 uint query_offset = head_idx_index + sglid;
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 1; seq_idx++) {
 half8 query_vals_reg;
 __attribute__((opencl_unroll_hint)) for(uint i = 0; i < 8; i++) {
 query_vals_reg[i] = query_local[query_offset + i * 16];
 }
 __attribute__((opencl_unroll_hint)) for(uint i = 0; i < 8; i++) {
 acc[seq_idx] = mad(convert_float(query_vals_reg[i]), convert_float(key_vals[i]), acc[seq_idx]);
 }
 query_offset += 64;
 }
 }

 for (; head_idx_index + (4 * 16) <= 64; head_idx_index += 16 * 4) {





 half4 key_vec_packed = as_half4(_sub_group_block_read_us4((const __global ushort*)(key_input) + (key_offset + head_idx_index)));;





 half4 key_vals = key_vec_packed;

 uint query_offset = head_idx_index + sglid;
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 1; seq_idx++) {
 half4 query_vals_reg;
 __attribute__((opencl_unroll_hint)) for(uint i = 0; i < 4; i++) {
 query_vals_reg[i] = query_local[query_offset + i * 16];
 }
 __attribute__((opencl_unroll_hint)) for(uint i = 0; i < 4; i++) {
 acc[seq_idx] = mad(convert_float(query_vals_reg[i]), convert_float(key_vals[i]), acc[seq_idx]);
 }
 query_offset += 64;
 }
 }

 for (; head_idx_index + (2 * 16) <= 64; head_idx_index += 16 * 2) {





 half2 key_vec_packed = as_half2(_sub_group_block_read_us2((const __global ushort*)(key_input) + (key_offset + head_idx_index)));;





 half2 key_vals = key_vec_packed;

 uint query_offset = head_idx_index + sglid;
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 1; seq_idx++) {
 half2 query_vals_reg;
 __attribute__((opencl_unroll_hint)) for(uint i = 0; i < 2; i++) {
 query_vals_reg[i] = query_local[query_offset + i * 16];
 }
 __attribute__((opencl_unroll_hint)) for(uint i = 0; i < 2; i++) {
 acc[seq_idx] = mad(convert_float(query_vals_reg[i]), convert_float(key_vals[i]), acc[seq_idx]);
 }
 query_offset += 64;
 }
 }

 for (; head_idx_index + (1 * 16) <= 64; head_idx_index += 16 * 1) {





 half key_vec_packed = as_half(_sub_group_block_read_us((const __global ushort*)(key_input) + (key_offset + head_idx_index)));;





 half key_vals = key_vec_packed;

 uint query_offset = head_idx_index + sglid;
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 1; seq_idx++) {
 half query_vals_reg;
 __attribute__((opencl_unroll_hint)) for(uint i = 0; i < 1; i++) {
 query_vals_reg = query_local[query_offset + i * 16];
 }
 acc[seq_idx] = mad(convert_float(query_vals_reg), convert_float(key_vals), acc[seq_idx]);
 query_offset += 64;
 }
 }
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 1; seq_idx++) {
 acc[seq_idx] = sub_group_reduce_add(acc[seq_idx]);
 qk_local[seq_idx * 256 + seq_len] = acc[seq_idx];
 }
 }
 {
 barrier(CLK_LOCAL_MEM_FENCE);
 float qk_val[1];
 const uint seq_idx_end = 1;
 for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
 for (uint seq_len = sgid * 16 + sglid; seq_len < partition_seq_len; seq_len += (64 * 2)) {
 qk_val[seq_idx] = qk_local[seq_idx * 256 + seq_len];




 const uint attn_mask_offset = ((1*0) + ((shape_info[35] + 0)*0) + (((shape_info[35] + 0)*(shape_info[34] + 0))*0) + (((shape_info[35] + 0)*(shape_info[34] + 0)*1)*0) + (((shape_info[35] + 0)*(shape_info[34] + 0)*1*1*1*1)*0) + (((shape_info[35] + 0)*(shape_info[34] + 0)*1*1*1*1*(shape_info[29] + 0))*0)) + (start_partition_idx + seq_len % (shape_info[35] ))*1 + (target_seq_idx + seq_idx % (shape_info[34] ))*(shape_info[35] + 0) + (b1_idx % (shape_info[29] ))*((shape_info[35] + 0)*(shape_info[34] + 0)*1*1*1*1) + (b0_idx % (shape_info[28] ))*((shape_info[35] + 0)*(shape_info[34] + 0)*1*1*1*1*(shape_info[29] + 0));
 qk_val[seq_idx] += attn_mask[attn_mask_offset];

 qk_max[seq_idx] = fmax(qk_max[seq_idx], convert_float(qk_val[seq_idx]));
 qk_local[seq_idx * 256 + seq_len] = qk_val[seq_idx];
 }
 }
 }
 }
 {
 const uint seq_idx_end = 1;
 for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
 qk_max[seq_idx] = sub_group_reduce_max(qk_max[seq_idx]);
 }
 if (sglid == 0) {
 for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
 qk_max_vals[seq_idx * (64 * 2 / 16) + sgid] = qk_max[seq_idx];
 }
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 for (uint seq_idx = 0; seq_idx < 1; seq_idx++) {
 qk_max[seq_idx] = -FLT_MAX;
 if (sglid < (64 * 2 / 16))
 qk_max[seq_idx] = qk_max_vals[seq_idx * (64 * 2 / 16) + sglid];
 qk_max[seq_idx] = sub_group_reduce_max(qk_max[seq_idx]);
 }
 float exp_sum[1] = {0.0f};
 const uint qk_num_per_wi = (((partition_seq_len) + ((64 * 2 / 16) * 16) - 1)/((64 * 2 / 16) * 16));
 for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
 const uint local_data_idx = qk_idx * ((64 * 2 / 16) * 16) + lid;
 if (local_data_idx < partition_seq_len) {
 for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
 float qk_new = native_exp(convert_float(qk_local[seq_idx * 256 + local_data_idx]) - qk_max[seq_idx]);
 qk_local[seq_idx * 256 + local_data_idx] = convert_half(qk_new);
 exp_sum[seq_idx] += qk_new;
 }
 }
 }
 for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
 exp_sum[seq_idx] = sub_group_reduce_add(exp_sum[seq_idx]);
 if (sglid == 0)
 qk_sum_vals[seq_idx * (64 * 2 / 16) + sgid] = exp_sum[seq_idx];
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 1; seq_idx++) {
 exp_sum[seq_idx] = 0.0f;
 if (sglid < (64 * 2 / 16))
 exp_sum[seq_idx] = qk_sum_vals[seq_idx * (64 * 2 / 16) + sglid];
 exp_sum[seq_idx] = sub_group_reduce_add(exp_sum[seq_idx]);
 }
 for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
 const uint local_data_idx = qk_idx * ((64 * 2 / 16) * 16) + lid;
 if (local_data_idx < partition_seq_len) {
 for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
 float qk_new = convert_float(qk_local[seq_idx * 256 + local_data_idx]) / exp_sum[seq_idx];
 qk_local[seq_idx * 256 + local_data_idx] = convert_half(qk_new);
 }
 }
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 {
 if (num_of_partitions > 1 && lid == 0) {
 for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
 const uint exp_sums_offset = b0_idx * (14 * (shape_info[6] ) * num_of_partitions) +
 b1_idx * ((shape_info[6] ) * num_of_partitions) +
 (seq_idx + target_seq_idx) * (num_of_partitions) +
 partition_idx;
 exp_sums[exp_sums_offset] = exp_sum[seq_idx];
 const uint max_logits_offset = exp_sums_offset;
 max_logits[max_logits_offset] = qk_max[seq_idx];
 }
 }
 }
 }
 }
 {
 half acc[1] = {0.0h};


 uint value_offset = _get_input2_index_sdpa_opt_single_token_7001936805389565166_0_0__sa(shape_info, b0_idx, b1_idx, 0, 0, 0, 0);
 uint value_offset_next_seq = _get_input2_index_sdpa_opt_single_token_7001936805389565166_0_0__sa(shape_info, b0_idx, b1_idx, 0, 0, 1, 0);
 const uint value_pitch = value_offset_next_seq - value_offset;





 const uint seq_len_start = (sgid / (64 / 16)) * (256 / 2 / 16);
 const uint seq_len_end = min(seq_len_start + (256 / 2 / 16), partition_seq_len / 16);




 for (uint seq_len = seq_len_start; seq_len < seq_len_end; seq_len++) {




 const uint b_idx = b0_idx;

 uint value_offset = _get_input2_index_sdpa_opt_single_token_7001936805389565166_0_0__sa(shape_info, b_idx, b1_idx, 0, 0, start_partition_idx + (seq_len * 16), head_size_idx);
# 4769 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 half qk_val[1];
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 1; seq_idx++) {
 qk_val[seq_idx] = qk_local[seq_idx * 256 + seq_len * 16 + sglid];
 }
 __attribute__((opencl_unroll_hint)) for (uint i = 0; i < 16; i++) {



 const half value_packed = as_half(_sub_group_block_read_us((const __global ushort*)(value_input) + (value_offset)));






 half value_val = value_packed;

 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 1; seq_idx++) {
 acc[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], i), value_val, acc[seq_idx]);
 }

 value_offset += value_pitch;

 }
 }

 if (sgid >= 64 / 16) {

 for (uint seq_len = (partition_seq_len / 16) * 16; seq_len < partition_seq_len; seq_len++) {



 const uint b_idx = b0_idx;


 const uint value_offset = _get_input2_index_sdpa_opt_single_token_7001936805389565166_0_0__sa(shape_info, b_idx, b1_idx, 0, 0, start_partition_idx + seq_len, head_size_idx);
# 4815 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 half qk_val[1];
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 1; seq_idx++) {
 qk_val[seq_idx] = qk_local[seq_idx * 256 + seq_len];
 }
 const half value_packed = as_half(_sub_group_block_read_us((const __global ushort*)(value_input) + (value_offset)));





 const half value_val = value_packed;

 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 1; seq_idx++) {
 acc[seq_idx] = mad(qk_val[seq_idx], value_val, acc[seq_idx]);
 }
 }

 }


 if ((partition_seq_len > (256 / 2)) || (partition_seq_len % 16 != 0)) {
 if (sgid >= 64 / 16) {
 query_local[head_size_idx] = acc[0];
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 if (sgid < 64 / 16) {
 acc[0] += query_local[head_size_idx];
 }
 }


 if (sgid < 64 / 16) {

 if (num_of_partitions > 1) {
 const uint seq_idx_end = 1;
 for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
 const uint tmp_out_offset = b0_idx * (14 * (shape_info[6] ) * num_of_partitions * 64) +
 b1_idx * ((shape_info[6] ) * num_of_partitions * 64) +
 (target_seq_idx + seq_idx) * (num_of_partitions * 64) +
 partition_idx * (64) +
 head_size_idx;
 tmp_out[tmp_out_offset] = acc[seq_idx];
 }
 } else {
 const uint seq_idx_end = 1;
 for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
 const uint output_offset = ((1*0) + (64*0) + ((64*(shape_info[50] + 0))*0) + ((64*(shape_info[50] + 0)*1)*0) + ((64*(shape_info[50] + 0)*1*1*1*1)*0) + ((64*(shape_info[50] + 0)*1*1*1*1*14)*0)) + (head_size_idx)*1 + (target_seq_idx + seq_idx)*64 + (b1_idx)*(64*(shape_info[50] + 0)*1*1*1*1) + (b0_idx)*(64*(shape_info[50] + 0)*1*1*1*1*14);
 output[output_offset] = acc[seq_idx];
 }
 }

 }

 }
}
# 7448 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
inline uint _get_input0_index_nt_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) {

 return ((1*0) + (64*0) + ((64*(shape_info[6] + 0))*0) + ((64*(shape_info[6] + 0)*1)*0) + ((64*(shape_info[6] + 0)*1*1*1*1)*0) + ((64*(shape_info[6] + 0)*1*1*1*1*14)*0)) + (x)*1 + (y)*64 + (z)*(64*(shape_info[6] + 0)) + (w)*(64*(shape_info[6] + 0)*1) + (f)*(64*(shape_info[6] + 0)*1*1*1*1) + (b)*(64*(shape_info[6] + 0)*1*1*1*1*14);
# 7462 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
}
inline uint _get_input0_index_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) {

 return _get_input0_index_nt_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info, b,f,w,z,y,x);



}
inline uint _get_input1_index_nt_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) {

 f /= 7;;


 return ((1*0) + (64*(shape_info[16])) + ((64*(shape_info[14] + (shape_info[16] + shape_info[17])))*0) + ((64*(shape_info[14] + (shape_info[16] + shape_info[17]))*1)*0) + ((64*(shape_info[14] + (shape_info[16] + shape_info[17]))*1*1*1*1)*0) + ((64*(shape_info[14] + (shape_info[16] + shape_info[17]))*1*1*1*1*2)*0)) + (x)*1 + (y)*64 + (z)*(64*(shape_info[14] + (shape_info[16] + shape_info[17]))) + (w)*(64*(shape_info[14] + (shape_info[16] + shape_info[17]))*1) + (f)*(64*(shape_info[14] + (shape_info[16] + shape_info[17]))*1*1*1*1) + (b)*(64*(shape_info[14] + (shape_info[16] + shape_info[17]))*1*1*1*1*2);
# 7487 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
}
inline uint _get_input1_index_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) {

 return _get_input1_index_nt_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info, b,f,w,z,y,x);



}
inline uint _get_input2_index_nt_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) {

 f /= 7;;


 return ((1*0) + (64*(shape_info[26])) + ((64*(shape_info[24] + (shape_info[26] + shape_info[27])))*0) + ((64*(shape_info[24] + (shape_info[26] + shape_info[27]))*1)*0) + ((64*(shape_info[24] + (shape_info[26] + shape_info[27]))*1*1*1*1)*0) + ((64*(shape_info[24] + (shape_info[26] + shape_info[27]))*1*1*1*1*2)*0)) + (x % 64)*1 + (y % (shape_info[24] ))*64 + (z % 1)*(64*(shape_info[24] + (shape_info[26] + shape_info[27]))) + (w % 1)*(64*(shape_info[24] + (shape_info[26] + shape_info[27]))*1) + (f % 2)*(64*(shape_info[24] + (shape_info[26] + shape_info[27]))*1*1*1*1) + (b % (shape_info[18] ))*(64*(shape_info[24] + (shape_info[26] + shape_info[27]))*1*1*1*1*2);
# 7512 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
}
inline uint _get_input2_index_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(__global const int* shape_info, uint b, uint f, uint w, uint z, uint y, uint x) {

 return _get_input2_index_nt_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info, b,f,w,z,y,x);



}
# 8025 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
inline half16 _load_attn_mask_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(__global const int* shape_info,
 uint b0_idx,
 uint b1_idx,
 uint target_seq_idx,
 uint source_seq_idx
 , const __global half* attn_mask


 ) {
 half16 mask_vec = 0.0h;

 const uint attn_mask_offset = ((1*0) + ((shape_info[35] + 0)*0) + (((shape_info[35] + 0)*(shape_info[34] + 0))*0) + (((shape_info[35] + 0)*(shape_info[34] + 0)*1)*0) + (((shape_info[35] + 0)*(shape_info[34] + 0)*1*1*1*1)*0) + (((shape_info[35] + 0)*(shape_info[34] + 0)*1*1*1*1*(shape_info[29] + 0))*0)) + (source_seq_idx % (shape_info[35] ))*1 + (target_seq_idx % (shape_info[34] ))*(shape_info[35] + 0) + (b1_idx % (shape_info[29] ))*((shape_info[35] + 0)*(shape_info[34] + 0)*1*1*1*1) + (b0_idx % (shape_info[28] ))*((shape_info[35] + 0)*(shape_info[34] + 0)*1*1*1*1*(shape_info[29] + 0));
 if (target_seq_idx >= (uint)(shape_info[6] )) {
 __attribute__((opencl_unroll_hint)) for (uint i = 0; i < 16; i++) {
 mask_vec[i] = NAN;
 }
 } else {
 if (source_seq_idx + 16 <= (uint)(shape_info[14] )) {
 __attribute__((opencl_unroll_hint)) for (uint i = 0; i < 16; i++) {
 const half mask_val = attn_mask[attn_mask_offset + i];
 mask_vec[i] = mask_val;
 }
 } else {
 const uint max_mask_offset = min(source_seq_idx + 16, (uint)(shape_info[14] ));
 for (uint i = 0; i < 16; i++) {
 const half mask_val = source_seq_idx + i < max_mask_offset ? attn_mask[attn_mask_offset + i] : NAN;
 mask_vec[i] = mask_val;
 }
 }
 }
# 8083 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 const half scale_val = convert_half(as_float(0x41000000));


 mask_vec *= scale_val;

 return mask_vec;
}








__kernel void sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(
 __global const int* shape_info,
 const __global half* query_input,
 const __global half* key_input,
 const __global half* value_input,




 const __global half* attn_mask,







 __global half* output,
# 8128 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 __global float* exp_sums,
 __global float* max_logits,
 __global half* tmp_out

)
{
# 8146 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 __local half slm_query[64 * 16];
 __local half slm_qk_vals[128 * 16];
 __local float slm_qk_max_vals[(64 * 2 / 16) * 16];
 __local float slm_exp_sum_vals[(64 * 2 / 16) * 16];
 __local float slm_exp_sum_cur[16];
 __local float slm_max_val_cur[16];
 __local float slm_exp_sum_prev[16];
 __local float slm_max_val_prev[16];
 {
# 8165 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 uint query_offset = _get_input0_index_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info, (((uint)get_global_id(0)) / 14), (((uint)get_global_id(0)) % 14), 0, 0, ((uint)get_global_id(1) * 16), (((uint)get_local_id(2) % 64)));
 uint query_offset_next_seq = _get_input0_index_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info, (((uint)get_global_id(0)) / 14), (((uint)get_global_id(0)) % 14), 0, 0, ((uint)get_global_id(1) * 16) + 1, (((uint)get_local_id(2) % 64)));
 const uint query_pitch = query_offset_next_seq - query_offset;




 const uint cur_target_seq_len_size = min((shape_info[6] ) - ((uint)get_global_id(1) * 16), (uint)16);

 uint query_local_offset = ((uint)get_local_id(2) % 64) * 16;







 const half scale_val = 1.0h;

 if (cur_target_seq_len_size != 16) {
 if ((uint)get_sub_group_id() * 16 < 64) {
 for (uint seq_idx = 0; seq_idx < cur_target_seq_len_size; seq_idx++) {
 half val = as_half(_sub_group_block_read_us((const __global ushort*)(query_input) + (query_offset)));
 slm_query[query_local_offset] = val * scale_val;
 query_offset += query_pitch;
 query_local_offset++;
 }
 }
 } else {

 if (((uint)get_sub_group_id() < ((64 * 2 / 16) / 2))) {
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < (16 / 2); seq_idx++) {
 half val = as_half(_sub_group_block_read_us((const __global ushort*)(query_input) + (query_offset)));
 slm_query[query_local_offset] = val * scale_val;
 query_offset += query_pitch;
 query_local_offset++;
 }
 } else {
 query_local_offset += (16 / 2);
 query_offset += query_pitch * (16 / 2);
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < (16 / 2); seq_idx++) {
 half val = as_half(_sub_group_block_read_us((const __global ushort*)(query_input) + (query_offset)));
 slm_query[query_local_offset] = val * scale_val;
 query_offset += query_pitch;
 query_local_offset++;
 }
 }
# 8229 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 }
 {

 if ((uint)get_sub_group_id() == 0 && (uint)get_sub_group_local_id() < 16) {
 slm_max_val_prev[(uint)get_sub_group_local_id()] = -FLT_MAX;
 slm_exp_sum_prev[(uint)get_sub_group_local_id()] = 0.0f;
 }



 }
 half16 output_acc = 0.0h;
 __attribute__((opencl_unroll_hint(1)))
 for (uint start_partition_idx = 0; start_partition_idx < (shape_info[14] ); start_partition_idx += 128) {
 float qk_max = -FLT_MAX;
 const uint seq_len = start_partition_idx + (uint)get_sub_group_id() * 16;
 const uint partition_seq_len = min((uint)(shape_info[14] ) - start_partition_idx, (uint)128);
# 8265 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 const uint b_idx = (((uint)get_global_id(0)) / 14);

 uint key_offset = _get_input1_index_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info, (((uint)get_global_id(0)) / 14), (((uint)get_global_id(0)) % 14), 0, 0, seq_len, 0);
 uint key_offset_next_seq = _get_input1_index_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info, (((uint)get_global_id(0)) / 14), (((uint)get_global_id(0)) % 14), 0, 0, seq_len + 1, 0);
 const uint key_pitch = key_offset_next_seq - key_offset;






 int seq_len_calc_size = min((int)((shape_info[14] )) - (int)seq_len, (int)16);
 half16 qk_acc;
 qk_acc = _load_attn_mask_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info,
 (((uint)get_global_id(0)) / 14),
 (((uint)get_global_id(0)) % 14),



 ((uint)get_global_id(1) * 16) + (uint)get_sub_group_local_id(),

 seq_len
 , attn_mask

 );
 if (seq_len_calc_size >= 16) {







 __attribute__((opencl_unroll_hint(1)))
 for (uint head_idx_index = 0; head_idx_index < 64; head_idx_index += 16) {


 half16 queries_vec;
 uint query_local_offset = (head_idx_index * 16) + (uint)get_sub_group_local_id();
 __attribute__((opencl_unroll_hint)) for (uint q_row_idx = 0; q_row_idx < 16; q_row_idx++) {
 queries_vec[q_row_idx] = slm_query[query_local_offset];
 query_local_offset += 16;
 }
 __attribute__((opencl_unroll_hint)) for (uint key_row_idx = 0; key_row_idx < 16; key_row_idx++) {



 const half key_packed = as_half(_sub_group_block_read_us((const __global ushort*)(key_input) + (key_offset + key_row_idx * key_pitch + head_idx_index)));;






 half key_vals = key_packed;

 __attribute__((opencl_unroll_hint)) for (uint i = 0; i < 16; i++) {
 qk_acc[key_row_idx] = mad(sub_group_broadcast(key_vals, i), queries_vec[i], qk_acc[key_row_idx]);
 }
 }
 }
 } else if (seq_len_calc_size > 0) {







 __attribute__((opencl_unroll_hint(1)))
 for (uint head_idx_index = 0; head_idx_index < 64; head_idx_index += 16) {
# 8347 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 half16 queries_vec;
 uint query_local_offset = (head_idx_index * 16) + (uint)get_sub_group_local_id();
 __attribute__((opencl_unroll_hint)) for (uint q_row_idx = 0; q_row_idx < 16; q_row_idx++) {
 queries_vec[q_row_idx] = slm_query[query_local_offset];
 query_local_offset += 16;
 }

 half16 key_vec = 0;
 __attribute__((opencl_unroll_hint)) for (uint key_row_idx = 0; key_row_idx < seq_len_calc_size; key_row_idx++) {



 key_vec[key_row_idx] = convert_half(as_half(_sub_group_block_read_us((const __global ushort*)(key_input) + (key_offset + key_row_idx * key_pitch + head_idx_index))));






 }

 __attribute__((opencl_unroll_hint)) for (uint key_row_idx = 0; key_row_idx < 16; key_row_idx++) {
# 8386 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 __attribute__((opencl_unroll_hint)) for (uint i = 0; i < 16; i++) {
 qk_acc[key_row_idx] = mad(sub_group_broadcast(key_vec[key_row_idx], i), queries_vec[i], qk_acc[key_row_idx]);
 }
 }
 }
 }
 {
 __attribute__((opencl_unroll_hint)) for (uint i = 0; i < 16; i++) {




 const half scale_val = convert_half(as_float(0x3dffffff));

 qk_acc[i] *= scale_val;





 qk_acc[i] = fmin(fmax(qk_acc[i], -HALF_MAX), HALF_MAX);
 qk_max = fmax(qk_max, convert_float(qk_acc[i]));
 }
 }
 {
 slm_qk_max_vals[(uint)get_sub_group_id() * 16 + (uint)get_sub_group_local_id()] = qk_max;
 qk_max = -FLT_MAX;
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 {
 float qk_max_new = -FLT_MAX;
 for (uint i = 0; i < (64 * 2 / 16); i++) {
 float qk_max_val = slm_qk_max_vals[i * 16 + (uint)get_sub_group_local_id()];
 qk_max_new = fmax(qk_max_new, qk_max_val);
 }
 if ((uint)get_sub_group_id() == 0) {
 slm_max_val_cur[(uint)get_sub_group_local_id()] = qk_max_new;
 }
 float exp_sum_new = 0.0f;
 for (uint i = 0; i < 16; i++) {
 qk_acc[i] = native_exp(convert_float(qk_acc[i]) - qk_max_new);
 exp_sum_new += qk_acc[i];
 }
 {
 slm_exp_sum_vals[(uint)get_sub_group_id() * 16 + (uint)get_sub_group_local_id()] = exp_sum_new;
 }
 exp_sum_new = 0.0f;
 barrier(CLK_LOCAL_MEM_FENCE);
 for (uint i = 0; i < (64 * 2 / 16); i++) {
 float exp_sum = slm_exp_sum_vals[i * 16 + (uint)get_sub_group_local_id()];
 exp_sum_new += exp_sum;
 }
 for (uint i = 0; i < 16; i++) {
 qk_acc[i] = qk_acc[i] / exp_sum_new;
 }
 if ((uint)get_sub_group_id() == 0) {
 slm_exp_sum_cur[(uint)get_sub_group_local_id()] = exp_sum_new;
 }
 for (uint i = 0; i < 16; i++) {
 slm_qk_vals[(uint)get_sub_group_local_id() * 128 + (uint)get_sub_group_id() * 16 + i] = qk_acc[i];
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 }
 {
 half16 acc_output_res = 0.0h;




 uint value_offset_base = _get_input2_index_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info, (((uint)get_global_id(0)) / 14), (((uint)get_global_id(0)) % 14), 0, 0, 0, 0);
 uint value_offset_next_seq = _get_input2_index_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info, (((uint)get_global_id(0)) / 14), (((uint)get_global_id(0)) % 14), 0, 0, 1, 0);
 const uint value_pitch = value_offset_next_seq - value_offset_base;




 if (partition_seq_len == 128) {
 uint seq_len_start = ((uint)get_sub_group_id() / ((64 * 2 / 16) / 2)) * (128 / 2);
 for (uint seq_len = seq_len_start; seq_len < seq_len_start + (128 / 2); seq_len += 16) {
# 8481 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 const uint b_idx = (((uint)get_global_id(0)) / 14);

 uint value_offset = _get_input2_index_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info, (((uint)get_global_id(0)) / 14), (((uint)get_global_id(0)) % 14), 0, 0, start_partition_idx + (seq_len), ((uint)get_local_id(2) % 64));





 half16 qk_val;
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 16; seq_idx++) {
 qk_val[seq_idx] = slm_qk_vals[seq_idx * 128 + seq_len + (uint)get_sub_group_local_id()];
 }







 __attribute__((opencl_unroll_hint)) for (uint i = 0; i < 16; i++) {



 const half value_packed = as_half(_sub_group_block_read_us((const __global ushort*)(value_input) + (value_offset)));






 half value_val = value_packed;

 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 16; seq_idx++) {
 acc_output_res[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], i), value_val, acc_output_res[seq_idx]);
 }

 value_offset += value_pitch;

 }
 }
 } else {
 const uint seq_len_start = ((uint)get_sub_group_id() / ((64 * 2 / 16) / 2)) * (128 / 2);
 uint seq_len_end = 0;
 if (seq_len_start < partition_seq_len)
 seq_len_end = seq_len_start + min(partition_seq_len - seq_len_start, (uint)(128 / 2));;
 for (uint seq_len = seq_len_start / 16; seq_len < seq_len_end / 16; seq_len++) {
# 8543 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 const uint b_idx = (((uint)get_global_id(0)) / 14);

 uint value_offset = _get_input2_index_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info, (((uint)get_global_id(0)) / 14), (((uint)get_global_id(0)) % 14), 0, 0, start_partition_idx + (seq_len * 16), ((uint)get_local_id(2) % 64));
# 8558 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 half16 qk_val;
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 16; seq_idx++) {
 qk_val[seq_idx] = slm_qk_vals[seq_idx * 128 + seq_len * 16 + (uint)get_sub_group_local_id()];
 }
 __attribute__((opencl_unroll_hint)) for (uint i = 0; i < 16; i++) {



 const half value_packed = as_half(_sub_group_block_read_us((const __global ushort*)(value_input) + (value_offset)));






 half value_val = value_packed;

 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 16; seq_idx++) {
 acc_output_res[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], i), value_val, acc_output_res[seq_idx]);
 }

 value_offset += value_pitch;

 }
 }
 const uint seq_len_leftovers_start = ((seq_len_end / 16) * 16);
 if (seq_len_leftovers_start != seq_len_end) {
 uint qk_offset = min(seq_len_leftovers_start + (uint)get_sub_group_local_id(), seq_len_end - 1);
 half16 qk_val;
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 16; seq_idx++) {
 qk_val[seq_idx] = slm_qk_vals[qk_offset];
 qk_offset += 128;
 }
# 8607 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 const uint b_idx = (((uint)get_global_id(0)) / 14);

 uint value_offset = _get_input2_index_sdpa_opt_multi_tokens_7001936805389565166_0_0__sa(shape_info, (((uint)get_global_id(0)) / 14), (((uint)get_global_id(0)) % 14), 0, 0, start_partition_idx + seq_len_leftovers_start, ((uint)get_local_id(2) % 64));
# 8622 "output_sources/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl"
 for (uint seq_len_idx = 0; seq_len_idx < partition_seq_len - seq_len_leftovers_start; seq_len_idx++) {



 const half value_packed = as_half(_sub_group_block_read_us((const __global ushort*)(value_input) + (value_offset)));






 half value_val = value_packed;

 for (uint seq_idx = 0; seq_idx < 16; seq_idx++) {
 acc_output_res[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], seq_len_idx), value_val, acc_output_res[seq_idx]);
 }

 value_offset += value_pitch;

 }
 }
 }
 {
 float exp_sum_prev = slm_exp_sum_prev[(uint)get_sub_group_local_id()];
 float exp_sum_cur = slm_exp_sum_cur[(uint)get_sub_group_local_id()];
 float max_val_prev = slm_max_val_prev[(uint)get_sub_group_local_id()];
 float max_val_cur = slm_max_val_cur[(uint)get_sub_group_local_id()];
 barrier(CLK_LOCAL_MEM_FENCE);





 const uint seq_idx_end = min((shape_info[6] ) - ((uint)get_global_id(1) * 16), (uint)16);

 for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
 float total_max = fmax(sub_group_broadcast(max_val_prev, seq_idx), sub_group_broadcast(max_val_cur, seq_idx));
 float updated_exp_sum_prev = sub_group_broadcast(exp_sum_prev, seq_idx) * native_exp(sub_group_broadcast(max_val_prev, seq_idx) - total_max);
 float updated_exp_sum_cur = sub_group_broadcast(exp_sum_cur, seq_idx) * native_exp(sub_group_broadcast(max_val_cur, seq_idx) - total_max);
 float updated_total_exp_sum = updated_exp_sum_prev + updated_exp_sum_cur;
 if (start_partition_idx > 0) {
 half updated_prev_res = convert_float(output_acc[seq_idx]) * updated_exp_sum_prev / updated_total_exp_sum;;
 acc_output_res[seq_idx] *= updated_exp_sum_cur / updated_total_exp_sum;
 acc_output_res[seq_idx] += updated_prev_res;
 }
 output_acc[seq_idx] = acc_output_res[seq_idx];
 if ((uint)get_sub_group_id() == 0 && (uint)get_sub_group_local_id() == 0) {
 slm_exp_sum_prev[seq_idx] = updated_total_exp_sum;
 slm_max_val_prev[seq_idx] = total_max;
 }
 }
 }
 }
 }
 if ((uint)get_sub_group_id() >= ((64 * 2 / 16) / 2)) {
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 16; seq_idx++) {
 slm_qk_vals[seq_idx * 128 + (uint)get_local_id(2)] = output_acc[seq_idx];
 }
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 if ((uint)get_sub_group_id() < ((64 * 2 / 16) / 2)) {
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 16; seq_idx++) {
 __attribute__((opencl_unroll_hint)) for (uint i = 1; i < 2; i++) {
 output_acc[seq_idx] += slm_qk_vals[seq_idx * 128 + (i * 64) + ((uint)get_local_id(2) % 64)];
 }
 }






 uint output_offset = ((1*0) + (64*0) + ((64*(shape_info[50] + 0))*0) + ((64*(shape_info[50] + 0)*1)*0) + ((64*(shape_info[50] + 0)*1*1*1*1)*0) + ((64*(shape_info[50] + 0)*1*1*1*1*14)*0)) + ((uint)get_sub_group_id() * 16)*1 + (((uint)get_global_id(1) * 16))*64 + ((((uint)get_global_id(0)) % 14))*(64*(shape_info[50] + 0)*1*1*1*1) + ((((uint)get_global_id(0)) / 14))*(64*(shape_info[50] + 0)*1*1*1*1*14);
 const uint output_pitch = 64;





 if (get_global_id(1) == get_global_size(1) - 1) {
 const uint seq_idx_end = min((uint)(shape_info[6] ) - ((uint)get_global_id(1) * 16), (uint)16);

 for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
 _sub_group_block_write_us( (__global ushort*)(output) + (output_offset), as_ushort(output_acc[seq_idx]));
 output_offset += output_pitch;
 }
 } else {
 __attribute__((opencl_unroll_hint)) for (uint seq_idx = 0; seq_idx < 16; seq_idx++) {
 _sub_group_block_write_us( (__global ushort*)(output) + (output_offset), as_ushort(output_acc[seq_idx]));
 output_offset += output_pitch;
 }
 }
 }
}
