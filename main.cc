/*
    Standalone program to check .field file checksums

    Copyright (C) 2021  Christoph Lehner (christoph.lehner@ur.de)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

*/
#include <stdint.h>
#include <stdio.h>
#include <zlib.h>
#include <vector>
#include <assert.h>

// threading from Grid
#define DO_PRAGMA_(x) _Pragma (#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)
#define thread_num(a) omp_get_thread_num()
#define thread_max(a) omp_get_max_threads()
#define thread_for( i, num, ... )                           DO_PRAGMA(omp parallel for schedule(static)) for ( uint64_t i=0;i<num;i++) { __VA_ARGS__ } ;

// checksum from GPT
static uint32_t cgpt_crc32(unsigned char* data, int64_t len, uint32_t start_crc = 0) {

  off_t step = 1024*1024*1024;

  if (len == 0)
    return start_crc;

  if (len <= step) {

    //uint32_t ref = crc32(start_crc,data,len);

    // parallel version
    int64_t block_size = 512*1024;
    int64_t blocks = ( len % block_size == 0 ) ? ( len / block_size ) : ( len / block_size + 1 );
    std::vector<uint32_t> pcrcs(blocks);
    thread_for(iblk, blocks, {
	int64_t block_start = block_size * iblk;
	int64_t block_len = std::min(block_size, len - block_start);
	pcrcs[iblk] = crc32(iblk == 0 ? start_crc : 0,&data[block_start],block_len);
    });

    // crc
    uint32_t crc = pcrcs[0];
    // reduce
    for (int iblk=1;iblk<blocks;iblk++) {
      int64_t block_start = block_size * iblk;
      int64_t block_len = std::min(block_size, len - block_start);
      crc = crc32_combine(crc,pcrcs[iblk],block_len);
    }

    //assert(crc == ref);

    return crc;

  } else {

    // crc32 of zlib was incorrect for very large sizes, so do it block-wise
    uint32_t crc = start_crc;
    off_t blk = 0;
    
    while (len > step) {
      crc = cgpt_crc32(&data[blk],step,crc);
      blk += step;
      len -= step;
    }
    
    return cgpt_crc32(&data[blk],len,crc);
  }
}

// main file
int main(int argc, char* argv[]) {
  for (int i=1;i<argc;i++) {
    char* fn = argv[i];
    printf("Checking %s\n", fn);
    FILE* f = fopen(fn,"rb");
    if (!f) {
      fprintf(stderr,"File not found\n");
      return 1;
    }

    while (!feof(f)) {
      uint32_t n_tag, crc32, nd;
      if (fread(&n_tag,4,1,f)==0)
	break;
      char* tag = new char[n_tag+1];
      assert(tag);
      assert(fread(tag,n_tag,1,f)==1);
      tag[n_tag] = '\0';
      assert(fread(&crc32,4,1,f)==1);
      assert(fread(&nd,4,1,f)==1);
      int* dim = new int[2*nd];
      assert(dim);
      assert(fread(dim,4*2*nd,1,f)==1);
      uint64_t size;
      assert(fread(&size,8,1,f)==1);
      printf("Checking CRC32 of %s (crc32 = %X, nd = %d, size = %g GB) ... ",tag,crc32,nd, (double)size / 1e9);
      unsigned char* data = new unsigned char[size];
      assert(data);
      assert(fread(data,size,1,f)==1);
      
      uint32_t crc32_check = cgpt_crc32(data, size);
      if (crc32_check == crc32) {
	printf("OK\n");
      } else {
	printf("ERR (%X)\n",crc32_check);
	return 2;
      }

      delete[] data;
      delete[] dim;
      delete[] tag;
    }
    fclose(f);
  }
  return 0;
}
