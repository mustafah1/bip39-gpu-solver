#define uint32_t uint
#define uint64_t ulong
#define uint8_t uchar
#define NULL 0

// Use void* for generic memory operations to avoid pointer type issues
static void memset_bytes(void *str, int c, size_t n){
  uchar *p = (uchar *)str;
  for(int i=0;i<n;i++){
    p[i] = c;
  }
}

static void memcpy_bytes(void *dest, const void *src, size_t n){
  uchar *d = (uchar *)dest;
  const uchar *s = (const uchar *)src;
  for(int i=0;i<n;i++){
    d[i] = s[i];
  }
}

static void memcpy_offset_bytes(void *dest, const void *src, int offset, int bytes_to_copy){
  uchar *d = (uchar *)dest;
  const uchar *s = (const uchar *)src;
  for(int i=0;i<bytes_to_copy;i++){
    d[i] = s[offset+i];
  }
}

// Legacy names for compatibility - just call the new versions
#define memset(str, c, n) memset_bytes((void*)(str), (c), (n))
#define memcpy(dest, src, n) memcpy_bytes((void*)(dest), (const void*)(src), (n))
#define memcpy_offset(dest, src, offset, bytes) memcpy_offset_bytes((void*)(dest), (const void*)(src), (offset), (bytes))

static void memzero(void *const pnt, const size_t len) {
  volatile unsigned char *volatile pnt_ = (volatile unsigned char *volatile)pnt;
  size_t i = (size_t)0U;

  while (i < len) {
    pnt_[i++] = 0U;
  }
}

static void memczero(void *s, size_t len, int flag) {
    unsigned char *p = (unsigned char *)s;
    volatile int vflag = flag;
    unsigned char mask = -(unsigned char) vflag;
    while (len) {
        *p &= ~mask;
        p++;
        len--;
    }
}

void copy_pad_previous(void *pad, void *previous, void *joined) {
  uchar *p = (uchar *)pad;
  uchar *prev = (uchar *)previous;
  uchar *j = (uchar *)joined;
  for(int x=0;x<128;x++){
    j[x] = p[x];
  }
  for(int x=0;x<64;x++){
    j[x+128] = prev[x];
  }
}

void print_byte_array_hex(uchar *arr, int len) {
  for (int i = 0; i < len; i++) {
    printf("%02x", arr[i]);
  }
  printf("\n\n");
}

void xor_seed_with_round(void *seed, void *round) {
  uchar *s = (uchar *)seed;
  uchar *r = (uchar *)round;
  for(int x=0;x<64;x++){
    s[x] = s[x] ^ r[x];
  }
}

void print_seed(uchar *seed){
  printf("seed: ");
  print_byte_array_hex(seed, 64);
}

void print_raw_address(uchar *address){
  printf("address: ");
  print_byte_array_hex(address, 25);
}

void print_mnemonic(uchar *mnemonic) {
  printf("mnemonic: ");
  for(int i=0;i<120;i++){
    printf("%c", mnemonic[i]);
  }
  printf("\n");
}

void print_byte_array(uchar *arr, int len) {
  printf("[");
  for(int x=0;x<len;x++){
    printf("%u", arr[x]);
    if(x < len-1){
      printf(", ");
    }
  }
  printf("]\n");
}