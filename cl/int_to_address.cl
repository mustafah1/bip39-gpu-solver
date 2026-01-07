

// Batch kernel - generates permutations on GPU
__constant uchar TARGET_ADDRESS[25] = {0x05, 0x74, 0xa3, 0x98, 0xff, 0x7b, 0xd2, 0x28, 0x70, 0x8c, 0x73, 0xde, 0xd2, 0x8a, 0xa5, 0xb2, 0x22, 0x61, 0xb0, 0x86, 0x43, 0x8e, 0xe5, 0x6e, 0xd2};
__constant ushort PERM_WORDS[12] = {112, 146, 238, 608, 759, 905, 1251, 1348, 1437, 1559, 1597, 1841};
__constant ulong FACTORIALS[13] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600};

__kernel void int_to_address(ulong start_k,
                             __global uchar * target_mnemonic, __global uchar * found_idx,
                             __global const secp256k1_ge_storage* prec_table,
                             uint batch_len) {
  ulong idx = get_global_id(0);
  if (idx >= batch_len) {
    return;
  }

  ulong k = start_k + idx;
  ushort remaining[12];
  ushort indices[12];
  for (int i = 0; i < 12; i++) {
    remaining[i] = PERM_WORDS[i];
  }
  for (int i = 12; i >= 1; i--) {
    ulong f = FACTORIALS[i - 1];
    ulong j = k / f;
    k = k % f;
    indices[12 - i] = remaining[j];
    for (ulong m = j; m < (ulong)(i - 1); m++) {
      remaining[m] = remaining[m + 1];
    }
  }

  ulong mnemonic_hi = 0;
  ulong mnemonic_lo = 0;
  for (int i = 0; i < 12; i++) {
    ulong w = (ulong)indices[i] & 2047;
    mnemonic_hi = (mnemonic_hi << 11) | (mnemonic_lo >> 53);
    mnemonic_lo = (mnemonic_lo << 11) | w;
  }
  mnemonic_hi = (mnemonic_hi << 4) | (mnemonic_lo >> 60);
  mnemonic_lo = (mnemonic_lo << 4);

  uchar bytes[16];
  bytes[15] = mnemonic_lo & 0xFF;
  bytes[14] = (mnemonic_lo >> 8) & 0xFF;
  bytes[13] = (mnemonic_lo >> 16) & 0xFF;
  bytes[12] = (mnemonic_lo >> 24) & 0xFF;
  bytes[11] = (mnemonic_lo >> 32) & 0xFF;
  bytes[10] = (mnemonic_lo >> 40) & 0xFF;
  bytes[9] = (mnemonic_lo >> 48) & 0xFF;
  bytes[8] = (mnemonic_lo >> 56) & 0xFF;
  
  bytes[7] = mnemonic_hi & 0xFF;
  bytes[6] = (mnemonic_hi >> 8) & 0xFF;
  bytes[5] = (mnemonic_hi >> 16) & 0xFF;
  bytes[4] = (mnemonic_hi >> 24) & 0xFF;
  bytes[3] = (mnemonic_hi >> 32) & 0xFF;
  bytes[2] = (mnemonic_hi >> 40) & 0xFF;
  bytes[1] = (mnemonic_hi >> 48) & 0xFF;
  bytes[0] = (mnemonic_hi >> 56) & 0xFF;

  uchar mnemonic_hash[32];
  sha256_bytes(bytes, 16, mnemonic_hash);
  uchar checksum = (mnemonic_hash[0] >> 4) & 0x0F;
  if (((uchar)indices[11] & 0x0F) != checksum) {
    return;
  }

  uchar ipad_key[128];
  uchar opad_key[128];
  for(int x=0;x<128;x++){
    ipad_key[x] = 0x36;
    opad_key[x] = 0x5c;
  }

  int mnemonic_length = 0;
  for (int i=0; i < 12; i++) {
    int word_index = indices[i];
    int word_length = word_lengths[word_index];
    for(int j=0;j<word_length;j++) {
      uchar b = words[word_index][j];
      ipad_key[mnemonic_length] ^= b;
      opad_key[mnemonic_length] ^= b;
      mnemonic_length++;
    }
    if (i < 11) {
      ipad_key[mnemonic_length] ^= 32;
      opad_key[mnemonic_length] ^= 32;
      mnemonic_length++;
    }
  }

  uchar seed[64] = { 0 };
  uchar sha512_result[64] = { 0 };
  uchar key_previous_concat[192] = { 0 };
  uchar salt[12] = { 109, 110, 101, 109, 111, 110, 105, 99, 0, 0, 0, 1 };
  for(int x=0;x<128;x++){
    key_previous_concat[x] = ipad_key[x];
  }
  for(int x=0;x<12;x++){
    key_previous_concat[x+128] = salt[x];
  }

  sha512_bytes(key_previous_concat, 140, sha512_result);
  copy_pad_previous(opad_key, sha512_result, key_previous_concat);
  sha512_bytes(key_previous_concat, 192, sha512_result);
  xor_seed_with_round(seed, sha512_result);

  #pragma unroll 1
  for(int x=1;x<2048;x++){
    copy_pad_previous(ipad_key, sha512_result, key_previous_concat);
    sha512_bytes(key_previous_concat, 192, sha512_result);
    copy_pad_previous(opad_key, sha512_result, key_previous_concat);
    sha512_bytes(key_previous_concat, 192, sha512_result);
    xor_seed_with_round(seed, sha512_result);
  }

  uchar network = BITCOIN_MAINNET;
  extended_private_key_t master_private;
  extended_public_key_t master_public;

  new_master_from_seed(network, seed, &master_private);
  public_from_private(&master_private, &master_public, prec_table);

  extended_private_key_t target_key;
  extended_public_key_t target_public_key;
  hardened_private_child_from_private(&master_private, &target_key, 49);
  hardened_private_child_from_private(&target_key, &target_key, 0);
  hardened_private_child_from_private(&target_key, &target_key, 0);
  normal_private_child_from_private(&target_key, &target_key, 0, prec_table);
  normal_private_child_from_private(&target_key, &target_key, 0, prec_table);
  public_from_private(&target_key, &target_public_key, prec_table);

  uchar raw_address[25] = {0};
  p2shwpkh_address_for_public_key(&target_public_key, raw_address);
 
  bool found_target = 1;
  for(int i=0;i<25;i++) {
    if(raw_address[i] != TARGET_ADDRESS[i]){
      found_target = 0;
    }
  }

  if(found_target == 1) {
    found_idx[0] = 0x01;
    // Store the index that was found
    found_idx[1] = (idx >> 24) & 0xFF;
    found_idx[2] = (idx >> 16) & 0xFF;
    found_idx[3] = (idx >> 8) & 0xFF;
    found_idx[4] = idx & 0xFF;
    int out_idx = 0;
    for (int i=0; i < 12; i++) {
      int word_index = indices[i];
      int word_length = word_lengths[word_index];
      for(int j=0;j<word_length;j++) {
        target_mnemonic[out_idx] = words[word_index][j];
        out_idx++;
      }
      if (i < 11) {
        target_mnemonic[out_idx] = 32;
        out_idx++;
      }
    }
    target_mnemonic[out_idx] = 0;
  }
}
