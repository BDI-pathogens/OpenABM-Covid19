#include <stdio.h>
#include <stdlib.h>

void fileOpen(FILE** file, char fileName[]){
    *file = fopen(fileName, "r");
    while(*file == NULL) {
         ;
         }
      char chunk[128];
      while(fgets(chunk, sizeof(chunk), *file) != NULL) {
            fputs(chunk, stdout);
            fputs("\n", stdout);  // marker string used to show where the content of the chunk array has ended
 
      fclose(*file);
    }
}

void fileWrite(FILE** fp, char fileName[], char string[]){
   int i;
   *fp = fopen (fileName,"w");
   fprintf (*fp, "%s", string);
   fclose (*fp);

}

int main()
{
   /* write a temporary file for python use */
   #define LEN 256
   FILE *fp;
   char output[] = "./temp_python.txt";
   char string[] = "initial output";
   fileWrite(&fp, output, string);

   /* read the temporary file created in python 
   FILE *mainFile;
   char name[] = "./temp_python.txt";
   fileOpen(&mainFile, name);*/

   return 0;
}
