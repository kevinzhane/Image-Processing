#include <stdio.h>
#include <stdlib.h>


int main()
{
    // open the file with 'rb' & 'wb'
    FILE *streamIn = fopen("images/cameraman.bmp","rb");
    FILE *fo = fopen("images/cameraman_copy.bmp","wb");

    // check the file is exist
    if (streamIn == (FILE*)0)
    {
        printf("Unable to open file\n");
    }

    // create  the array (image header) to hold the variable
    unsigned char header[54];
    unsigned char colorTable[1024];

    // use loop to read the extract data from streamIn
    for (int i = 0; i < 54 ; i++)
    {
        header[i] = getc(streamIn);
    }

    // read the information from header number
    int width = *(int *)&header[18];
    int height = *(int *)&header[22];
    int bitDepth = *(int *)&header[28];

    // check the bitmap color table
    if(bitDepth<=8)
    {
        fread(colorTable,sizeof(unsigned char), 1024,streamIn);
    }

    // create the buffer to hole the image pixel data
    fwrite(header,sizeof(unsigned char),54,fo);
    unsigned char buf[height * width];
    fread(buf,sizeof(unsigned char),(height*width), streamIn);

    if(bitDepth <=8)
    {
       fwrite(colorTable,sizeof(unsigned char),1024,fo);
    }

        // write the data to fo   
       fwrite(buf,sizeof(unsigned char),(height *width),fo);
       fclose(fo);
       fclose(streamIn);

       printf("Success !\n");
       printf("Width : %d\n", width);
       printf("Height : %d\n",height);

    return 0;





}

