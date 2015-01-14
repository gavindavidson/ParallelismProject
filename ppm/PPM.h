// based on code by rpf (ruifelgueiras.pt.vu) from stackoverflow
// http://stackoverflow.com/questions/2693631/read-ppm-file-and-store-it-in-an-array-coded-with-c/2699908#2699908
// Changed to have a union type

#include <stdlib.h>

#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;
typedef PPMPixel RGB;
//typedef union RGB
//{
//  PPMPixel bytes;
//  u_int32_t uint;
//} RGB;

typedef struct {
     int x, y;
     RGB *data;
} PPMImage;

#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

PPMImage *readPPM(const char *filename);

void writePPM(const char *filename, PPMImage *img);

void changeColorPPM(PPMImage *img);
