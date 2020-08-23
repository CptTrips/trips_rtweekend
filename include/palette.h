// palette.h
// Library for writing bmp files

#ifndef PALETTE_H
#define PALETTE_H

#include <windows.h>

void win_write(char* msg, DWORD msg_len, HANDLE file_handle);
char* generate_header(int filesize, int w, int h);
void write_int_to_header(char* header, int number, int start);
void buffer_to_data(char* pixel_buffer, int h, int w, char* data, int pad_len, int byte_w);
void bmp_write(char* pixel_buffer, int h, int w, const char* filename);


#endif
